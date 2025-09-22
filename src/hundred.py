#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulação de 100 geodésicas (partículas de teste) no espaço-tempo de Schwarzschild
(restritas ao plano equatorial θ = π/2). Integração numérica com RK45 (solve_ivp).

Teoria (resumo):
  Métrica (c=G=1): ds^2 = -(1-2M/r) dt^2 + (1-2M/r)^(-1) dr^2 + r^2 dφ^2  (em θ=π/2).
  Equações geodésicas (forma de 2ª ordem): d²x^μ/dτ² + Γ^μ_{αβ} (dx^α/dτ)(dx^β/dτ) = 0.
  Símbolos de Christoffel não-nulos usados (θ=π/2):
    f = 1 - 2M/r
    Γ^t_{tr} = Γ^t_{rt} =  M/(r^2 f)
    Γ^r_{tt} =  M f / r^2
    Γ^r_{rr} = -M/(r^2 f)
    Γ^r_{φφ} = -f r
    Γ^φ_{rφ} = Γ^φ_{φ r} = 1/r

Inicialização via integrais de movimento (energia E e momento angular L):
  dt/dτ = E / f(r),   dφ/dτ = L / r^2,
  (dr/dτ)^2 + f(r)*(1 + L^2/r^2) = E^2.
Mantemos o sistema completo de 2ª ordem para permitir inversões radiais (turning points).

Saídas:
  figures/run_<timestamp>_orbit.png
  results/run_<timestamp>_trajectory.csv   (particle_id, tau, t, r, phi, x, y)
  logs/run_<timestamp>_benchmark.json      (tempo/memória, integrador, passos, contagens)
"""
from __future__ import annotations

import csv
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import psutil
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from tqdm import tqdm


# -----------------------------
# Configuração da simulação
# -----------------------------
M = 1.0                        # massa do buraco negro (unidades geométricas)
N_PARTICLES = 100              # número de partículas
TAU_MAX = 3000.0               # limite de tempo próprio de integração
R_ESCAPE = 200.0 * M           # raio para considerar "escape"
R_H_EPS = 1e-6 * M             # margem acima do horizonte para evitar singularidade
RTOL = 1e-9
ATOL = 1e-9
MAX_STEP = 0.5                 # passo máximo para RK45 (ajuste fino de estabilidade/tempo)
R0_MIN = 6.1 * M               # início acima de 6M (fora da ISCO)
R0_MAX = 30.0 * M
L_MIN = 2.8 * M                # em torno do limiar 2*sqrt(3)~3.464M e acima
L_MAX = 6.0 * M
E_PERTURB_MIN = -0.02          # perturbação relativa em E para evitar "exatas circulares"
E_PERTURB_MAX = +0.08
RANDOM_SEED = 12345            # reprodutibilidade


@dataclass
class ParticleResult:
    particle_id: int
    termination_reason: str
    steps_taken: int
    tau: np.ndarray
    t: np.ndarray
    r: np.ndarray
    phi: np.ndarray


# -----------------------------
# Utilidades de caminho (root = dois níveis acima deste arquivo: clean/src -> ROOT)
# -----------------------------
def get_root_dirs() -> Tuple[Path, Path, Path, Path]:
    script_path = Path(__file__).resolve()
    root_dir = script_path.parents[2]  # .../ROOT
    figures_dir = root_dir / "figures"
    logs_dir = root_dir / "logs"
    results_dir = root_dir / "results"
    for d in (figures_dir, logs_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)
    return root_dir, figures_dir, logs_dir, results_dir


# -----------------------------
# Dinâmica: RHS das geodésicas (2ª ordem → forma de 1ª ordem ampliada)
# Estado y = [t, r, phi, vt, vr, vphi]
# -----------------------------
def rhs_schwarzschild(_tau: float, y: np.ndarray, M: float) -> np.ndarray:
    t, r, phi, vt, vr, vphi = y
    # Evitar divisão por zero se o integrador ultrapassar o horizonte por erro numérico
    r_eff = r if r > (2.0 * M + 1e-12) else (2.0 * M + 1e-12)

    f = 1.0 - 2.0 * M / r_eff

    # Christoffel (equatorial)
    gamma_t_tr = M / (r_eff * r_eff * f)          # Γ^t_{tr}
    gamma_r_tt = M * f / (r_eff * r_eff)          # Γ^r_{tt}
    gamma_r_rr = -M / (r_eff * r_eff * f)         # Γ^r_{rr}
    gamma_r_phph = -f * r_eff                     # Γ^r_{φφ}
    gamma_ph_rph = 1.0 / r_eff                    # Γ^φ_{rφ}

    # Equações de 1ª ordem
    dtau_dt = vt
    dtau_dr = vr
    dtau_dphi = vphi

    # 2ª ordem → derivadas das velocidades
    dvt_dtau = -2.0 * gamma_t_tr * vt * vr
    dvr_dtau = -(gamma_r_tt * vt * vt + gamma_r_rr * vr * vr + gamma_r_phph * vphi * vphi)
    dvphi_dtau = -2.0 * gamma_ph_rph * vr * vphi

    return np.array([dtau_dt, dtau_dr, dtau_dphi, dvt_dtau, dvr_dtau, dvphi_dtau], dtype=float)


# -----------------------------
# Eventos (parada por horizonte e por "escape")
# -----------------------------
def make_event_horizon(M: float):
    def event_horizon(_tau: float, y: np.ndarray) -> float:
        r = y[1]
        return r - (2.0 * M + R_H_EPS)
    event_horizon.terminal = True
    event_horizon.direction = -1.0
    return event_horizon

def make_event_escape(M: float):
    def event_escape(_tau: float, y: np.ndarray) -> float:
        r = y[1]
        return r - R_ESCAPE
    event_escape.terminal = True
    event_escape.direction = 1.0
    return event_escape


# -----------------------------
# Amostragem de condições iniciais físicas
# -----------------------------
def sample_initial_conditions(i: int, rng: np.random.Generator, M: float) -> Tuple[np.ndarray, float, float]:
    """
    Retorna y0 = [t0, r0, phi0, vt0, vr0, vphi0], e também (E, L).
    """
    r0 = rng.uniform(R0_MIN, R0_MAX)
    phi0 = rng.uniform(0.0, 2.0 * math.pi)
    L = rng.uniform(L_MIN, L_MAX)

    f0 = 1.0 - 2.0 * M / r0
    # Energia efetiva para dr/dτ = 0 no raio inicial:
    E_circ_like = math.sqrt(max(f0 * (1.0 + (L * L) / (r0 * r0)), 0.0))
    # Perturba um pouco para gerar dinâmica radial (entra/escapa/óbitas excêntricas)
    eps = rng.uniform(E_PERTURB_MIN, E_PERTURB_MAX)
    E = max(E_circ_like * (1.0 + eps), 0.0)

    vt0 = E / f0                     # dt/dτ no ponto inicial
    vphi0 = L / (r0 * r0)            # dφ/dτ
    # Sinal radial: metade para dentro, metade para fora (alternado por id)
    sign = -1.0 if (i % 2 == 0) else +1.0
    radicand = max(E * E - f0 * (1.0 + (L * L) / (r0 * r0)), 0.0)
    vr0 = sign * math.sqrt(radicand) # dr/dτ

    y0 = np.array([0.0, r0, phi0, vt0, vr0, vphi0], dtype=float)
    return y0, E, L


# -----------------------------
# Integra uma partícula
# -----------------------------
def simulate_one_particle(i: int, rng: np.random.Generator, M: float) -> ParticleResult:
    y0, E, L = sample_initial_conditions(i, rng, M)

    # Eventos
    event_h = make_event_horizon(M)
    event_e = make_event_escape(M)

    sol = solve_ivp(
        fun=lambda tau, y: rhs_schwarzschild(tau, y, M),
        t_span=(0.0, TAU_MAX),
        y0=y0,
        method="RK45",
        rtol=RTOL,
        atol=ATOL,
        max_step=MAX_STEP,
        events=[event_h, event_e],
        dense_output=False,
        vectorized=False
    )

    # Motivo de parada
    reason = "tau_max_reached"
    if sol.status == 1:
        # Parou em evento
        hit_h = len(sol.t_events[0]) > 0
        hit_e = len(sol.t_events[1]) > 0
        if hit_h and not hit_e:
            reason = "horizon_hit"
        elif hit_e and not hit_h:
            reason = "escaped"
        else:
            reason = "mixed_event"
    elif sol.status == -1:
        reason = "integration_failure"

    steps = max(len(sol.t) - 1, 0)

    return ParticleResult(
        particle_id=i,
        termination_reason=reason,
        steps_taken=steps,
        tau=sol.t,
        t=sol.y[0, :],
        r=sol.y[1, :],
        phi=sol.y[2, :]
    )


# -----------------------------
# Execução principal
# -----------------------------
def main():
    # Diretórios de saída
    root_dir, figures_dir, logs_dir, results_dir = get_root_dirs()

    # Timestamp
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = figures_dir / f"run_{ts}_orbit.png"
    csv_path = results_dir / f"run_{ts}_trajectory.csv"
    json_path = logs_dir / f"run_{ts}_benchmark.json"

    # Medição de recursos
    proc = psutil.Process(os.getpid())
    mem_start_mb = proc.memory_info().rss / (1024 ** 2)
    t0 = time.perf_counter()

    # Semente
    rng = np.random.default_rng(RANDOM_SEED)

    # Simulação das N partículas
    results: List[ParticleResult] = []
    for i in tqdm(range(N_PARTICLES), desc="Integrando partículas", ncols=80):
        res = simulate_one_particle(i, rng, M)
        results.append(res)

    # Pós-processamento: escrever CSV longo (todas as partículas)
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_id", "tau", "t", "r", "phi", "x", "y"])
        for res in results:
            x = res.r * np.cos(res.phi)
            y = res.r * np.sin(res.phi)
            for k in range(len(res.tau)):
                writer.writerow([
                    res.particle_id,
                    float(res.tau[k]),
                    float(res.t[k]),
                    float(res.r[k]),
                    float(res.phi[k]),
                    float(x[k]),
                    float(y[k]),
                ])

    # Figura: órbitas no plano (x,y) + horizonte
    plt.figure(figsize=(6, 6))
    # Horizonte (círculo de raio 2M)
    th = np.linspace(0, 2.0 * np.pi, 512)
    plt.plot(2.0 * M * np.cos(th), 2.0 * M * np.sin(th), linewidth=2, label="Horizonte r=2M")
    # Trajetórias
    for res in results:
        x = res.r * np.cos(res.phi)
        y = res.r * np.sin(res.phi)
        plt.plot(x, y, linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.title("Órbitas de 100 partículas de teste (Schwarzschild, plano equatorial)")
    plt.xlabel("x")
    plt.ylabel("y")
    # Limites automáticos, mas garanta que o horizonte apareça
    plt.axhline(0.0, linewidth=0.5)
    plt.axvline(0.0, linewidth=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Coleta final de métricas
    runtime_seconds = time.perf_counter() - t0
    mem_end_mb = proc.memory_info().rss / (1024 ** 2)

    total_steps = int(sum(r.steps_taken for r in results))
    reasons = [r.termination_reason for r in results]
    reason_counts: Dict[str, int] = {}
    for rr in reasons:
        reason_counts[rr] = reason_counts.get(rr, 0) + 1
    termination_reason_summary = "mixed" if len(reason_counts) > 1 else reasons[0]

    # JSON de benchmark (formato pedido + extras úteis)
    benchmark = {
        "timestamp": ts,
        "runtime_seconds": runtime_seconds,
        "memory_start_mb": mem_start_mb,
        "memory_end_mb": mem_end_mb,
        "memory_delta_mb": mem_end_mb - mem_start_mb,
        "particles_simulated": N_PARTICLES,
        "integrator": "RK45",
        "steps_taken": total_steps,
        "termination_reason": termination_reason_summary,
        "termination_reason_counts": reason_counts,
        "mass_M": M,
        "tau_max": TAU_MAX,
        "r_escape": R_ESCAPE,
        "rtol": RTOL,
        "atol": ATOL,
        "max_step": MAX_STEP,
        "seed": RANDOM_SEED,
        "paths": {
            "figure": str(fig_path.relative_to(root_dir)),
            "csv": str(csv_path.relative_to(root_dir)),
            "json": str(json_path.relative_to(root_dir)),
        },
    }

    with open(json_path, "w") as f:
        json.dump(benchmark, f, indent=4)

    # Saída mínima no terminal
    print(f"[OK] CSV:    {csv_path}")
    print(f"[OK] Figura: {fig_path}")
    print(f"[OK] JSON:   {json_path}")
    print(f"[INFO] Runtime: {runtime_seconds:.3f}s | ΔMem: {benchmark['memory_delta_mb']:.2f} MB | Passos: {total_steps}")
    print(f"[INFO] Finais: {reason_counts}")


if __name__ == "__main__":
    main()
