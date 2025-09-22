
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulação de 100 geodésicas (partículas de teste) no espaço-tempo de Schwarzschild
(restritas ao plano equatorial θ = π/2), paralelizada em múltiplos núcleos de CPU.
Integração numérica com RK45 (solve_ivp).

As saídas são segregadas por modo de execução e timestamp:
  figures/cpupar/cpupar_<YYYYMMDD_HHMMSS>_orbit.png
  results/cpupar/cpupar_<YYYYMMDD_HHMMSS>_trajectory.csv
  logs/cpupar/cpupar_<YYYYMMDD_HHMMSS>_benchmark.json
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
from multiprocessing import Pool

# -----------------------------
# Modo de execução (para nomear/organizar saídas)
# -----------------------------
RUN_MODE = "cpupar"

# -----------------------------
# Configuração da simulação
# -----------------------------
M = 1.0
N_PARTICLES = 100
TAU_MAX = 3000.0
R_ESCAPE = 200.0 * M
R_H_EPS = 1e-6 * M
RTOL = 1e-9
ATOL = 1e-9
MAX_STEP = 0.5
R0_MIN = 6.1 * M
R0_MAX = 30.0 * M
L_MIN = 2.8 * M
L_MAX = 6.0 * M
E_PERTURB_MIN = -0.02
E_PERTURB_MAX = +0.08
RANDOM_SEED = 12345

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
# Diretórios raiz (com subpastas por modo)
# -----------------------------
def get_root_dirs(run_mode: str) -> Tuple[Path, Path, Path, Path]:
    script_path = Path(__file__).resolve()
    # .../bhpar/clean/src -> parents[2] == .../bhpar
    root_dir = script_path.parents[2]
    figures_dir = root_dir / "figures" / run_mode
    logs_dir = root_dir / "logs" / run_mode
    results_dir = root_dir / "results" / run_mode
    for d in (figures_dir, logs_dir, results_dir):
        d.mkdir(parents=True, exist_ok=True)
    return root_dir, figures_dir, logs_dir, results_dir

# -----------------------------
# RHS das geodésicas de Schwarzschild (plano equatorial)
# -----------------------------
def rhs_schwarzschild(_tau: float, y: np.ndarray, M: float) -> np.ndarray:
    t, r, phi, vt, vr, vphi = y
    r_eff = r if r > (2.0 * M + 1e-12) else (2.0 * M + 1e-12)
    f = 1.0 - 2.0 * M / r_eff

    gamma_t_tr   = M / (r_eff * r_eff * f)
    gamma_r_tt   = M * f / (r_eff * r_eff)
    gamma_r_rr   = -M / (r_eff * r_eff * f)
    gamma_r_phph = -f * r_eff
    gamma_ph_rph = 1.0 / r_eff

    dtau_dt   = vt
    dtau_dr   = vr
    dtau_dphi = vphi

    dvt_dtau   = -2.0 * gamma_t_tr * vt * vr
    dvr_dtau   = -(gamma_r_tt * vt * vt + gamma_r_rr * vr * vr + gamma_r_phph * vphi * vphi)
    dvphi_dtau = -2.0 * gamma_ph_rph * vr * vphi

    return np.array([dtau_dt, dtau_dr, dtau_dphi, dvt_dtau, dvr_dtau, dvphi_dtau], dtype=float)

# -----------------------------
# Eventos de parada
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
# Amostragem de condições iniciais
# -----------------------------
def sample_initial_conditions(i: int, rng: np.random.Generator, M: float) -> Tuple[np.ndarray, float, float]:
    r0 = rng.uniform(R0_MIN, R0_MAX)
    phi0 = rng.uniform(0.0, 2.0 * math.pi)
    L = rng.uniform(L_MIN, L_MAX)

    f0 = 1.0 - 2.0 * M / r0
    E_circ_like = math.sqrt(max(f0 * (1.0 + (L * L) / (r0 * r0)), 0.0))
    eps = rng.uniform(E_PERTURB_MIN, E_PERTURB_MAX)
    E = max(E_circ_like * (1.0 + eps), 0.0)

    vt0 = E / f0
    vphi0 = L / (r0 * r0)
    sign = -1.0 if (i % 2 == 0) else +1.0
    radicand = max(E * E - f0 * (1.0 + (L * L) / (r0 * r0)), 0.0)
    vr0 = sign * math.sqrt(radicand)

    y0 = np.array([0.0, r0, phi0, vt0, vr0, vphi0], dtype=float)
    return y0, E, L

# -----------------------------
# Worker (corrigido para starmap)
# -----------------------------
def integrate_particle(i: int, y0: np.ndarray) -> ParticleResult:
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

    if sol.status == 1:
        hit_horizon = len(sol.t_events[0]) > 0
        hit_escape = len(sol.t_events[1]) > 0
        if hit_horizon and not hit_escape:
            reason = "horizon_hit"
        elif hit_escape and not hit_horizon:
            reason = "escaped"
        else:
            reason = "mixed_event"
    elif sol.status == 0:
        reason = "tau_max_reached"
    else:
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
# Main
# -----------------------------
def main():
    # Pastas de saída com escopo do modo
    root_dir, figures_dir, logs_dir, results_dir = get_root_dirs(RUN_MODE)

    # Timestamp e nomes de arquivo com prefixo do modo
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig_path = figures_dir / f"{RUN_MODE}_{ts}_orbit.png"
    csv_path = results_dir / f"{RUN_MODE}_{ts}_trajectory.csv"
    json_path = logs_dir / f"{RUN_MODE}_{ts}_benchmark.json"

    # Métricas iniciais
    proc = psutil.Process(os.getpid())
    mem_start_mb = proc.memory_info().rss / (1024 ** 2)
    t0 = time.perf_counter()

    # RNG e condições iniciais
    rng = np.random.default_rng(RANDOM_SEED)
    initial_conditions: List[np.ndarray] = []
    for i in range(N_PARTICLES):
        y0, E, L = sample_initial_conditions(i, rng, M)
        initial_conditions.append(y0)

    # Paralelização
    num_workers = 6  # 6 núcleos físicos; depois você pode testar 12 (SMT)
    print(f"Integrating {N_PARTICLES} particles in parallel on {num_workers} cores...")

    with Pool(processes=num_workers) as pool:
        args_list = [(i, initial_conditions[i]) for i in range(N_PARTICLES)]
        results: List[ParticleResult] = pool.starmap(integrate_particle, args_list)

    # Exporta CSV
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["particle_id", "tau", "t", "r", "phi", "x", "y"])
        for res in results:
            x_vals = res.r * np.cos(res.phi)
            y_vals = res.r * np.sin(res.phi)
            for k in range(len(res.tau)):
                writer.writerow([
                    res.particle_id,
                    float(res.tau[k]),
                    float(res.t[k]),
                    float(res.r[k]),
                    float(res.phi[k]),
                    float(x_vals[k]),
                    float(y_vals[k])
                ])

    # Exporta figura
    plt.figure(figsize=(6, 6))
    theta = np.linspace(0, 2.0 * math.pi, 512)
    plt.plot(2.0 * M * np.cos(theta), 2.0 * M * np.sin(theta), 'k-', linewidth=2, label="Horizonte r=2M")
    for res in results:
        x_vals = res.r * np.cos(res.phi)
        y_vals = res.r * np.sin(res.phi)
        plt.plot(x_vals, y_vals, linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="datalim")
    plt.title(f"Órbitas de {N_PARTICLES} partículas de teste (Schwarzschild, plano equatorial)")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axhline(0.0, color='gray', linewidth=0.5)
    plt.axvline(0.0, color='gray', linewidth=0.5)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(fig_path, dpi=200)
    plt.close()

    # Métricas finais
    runtime_seconds = time.perf_counter() - t0
    mem_end_mb = proc.memory_info().rss / (1024 ** 2)
    total_steps = int(sum(res.steps_taken for res in results))
    reason_counts: Dict[str, int] = {}
    for res in results:
        reason_counts[res.termination_reason] = reason_counts.get(res.termination_reason, 0) + 1
    termination_summary = list(reason_counts.keys())[0] if len(reason_counts) == 1 else "mixed"

    benchmark = {
        "run_mode": RUN_MODE,
        "timestamp": ts,
        "runtime_seconds": runtime_seconds,
        "memory_start_mb": mem_start_mb,
        "memory_end_mb": mem_end_mb,
        "memory_delta_mb": mem_end_mb - mem_start_mb,
        "particles_simulated": N_PARTICLES,
        "integrator": "RK45",
        "steps_taken": total_steps,
        "termination_reason": termination_summary,
        "termination_reason_counts": reason_counts,
        "mass_M": M,
        "tau_max": TAU_MAX,
        "r_escape": R_ESCAPE,
        "rtol": RTOL,
        "atol": ATOL,
        "max_step": MAX_STEP,
        "seed": RANDOM_SEED,
        "processes_used": num_workers,
        "hardware": {
            "cpu_count_logical": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False)
        },
        "paths": {
            "figure": str(fig_path.relative_to(root_dir)),
            "csv": str(csv_path.relative_to(root_dir)),
            "json": str(json_path.relative_to(root_dir))
        }
    }
    with open(json_path, "w") as f:
        json.dump(benchmark, f, indent=4)

    # Sumário no console
    print(f"[OK] CSV:    {csv_path}")
    print(f"[OK] Figura: {fig_path}")
    print(f"[OK] JSON:   {json_path}")
    print(f"[INFO] Mode: {RUN_MODE} | Runtime: {runtime_seconds:.3f}s | ΔMem: {benchmark['memory_delta_mb']:.2f} MB | Steps: {total_steps}")
    print(f"[INFO] Termination counts: {reason_counts}")

if __name__ == "__main__":
    main()
