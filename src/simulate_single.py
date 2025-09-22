"""
simulate_single.py

Simulates a single test particle geodesic in Schwarzschild spacetime (equatorial plane),
using first-integrals formulation. Outputs:
- Trajectory plot (x-y)
- CSV of trajectory samples
- Benchmark log (runtime, memory, particle count)
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import psutil
import time
import os
import json
import csv
from datetime import datetime
from pathlib import Path

# ┌────────────────────────────────────┐
# │             CONFIGURATION          │
# └────────────────────────────────────┘
CONFIG = {
    "mass_M": 1.0,
    "trajectory_type": "timelike",
    "E": 0.95,
    "L": 3.5,
    "r0": 10.0,
    "phi0": 0.0,
    "t0": 0.0,
    "rdot0": None,
    "integrator": "RK45",
    "rtol": 1e-10,
    "atol": 1e-12,
    "max_step": 0.5,
    "horizon_eps": 1e-3,
    "r_max": 500.0,
    "tau_max": 2e4,
    "max_steps": 100000,
    "plot_options": {
        "figsize": (8, 8),
        "dpi": 150,
        "draw_horizon": True
    },
    "output_dirs": {
        "results": "../results/",
        "figures": "../figures/",
        "logs": "../logs/"
    }
}



# ┌────────────────────────────────────┐
# │           DERIVED CONSTANTS        │
# └────────────────────────────────────┘

M = CONFIG["mass_M"]
kappa = 1.0 if CONFIG["trajectory_type"] == "timelike" else 0.0
r_h = 2 * M
horizon_eps = CONFIG["horizon_eps"]
r_max = CONFIG["r_max"]
tau_max = CONFIG["tau_max"]
max_steps = CONFIG["max_steps"]

# ┌────────────────────────────────────┐
# │         INITIAL CONDITIONS         │
# └────────────────────────────────────┘

r0 = CONFIG["r0"]
phi0 = CONFIG["phi0"]
t0 = CONFIG["t0"]
rdot0 = CONFIG["rdot0"]

# Compute rdot0 if not provided
if rdot0 is None:
    f = 1 - 2 * M / r0
    Veff_sq = f * (kappa + CONFIG["L"]**2 / r0**2)
    rdot_sq = CONFIG["E"]**2 - Veff_sq
    if rdot_sq < 0:
        raise ValueError("Initial condition invalid: E^2 < Veff^2. No real ṙ0.")
    rdot0 = -np.sqrt(rdot_sq)  # Inbound

# ┌────────────────────────────────────┐
# │         RHS OF ODE SYSTEM          │
# └────────────────────────────────────┘

def geodesic_rhs(tau, y):
    """
    State vector: y = [t, r, phi, rdot]

    Equations:
    ṫ = E / (1 - 2M/r)
    φ̇ = L / r^2
    ṙ = y[3]
    r̈ = (1/2) d/dr [E^2 - (1 - 2M/r)(κ + L^2/r^2)] * ṙ
    """
    t, r, phi, rdot = y

    f = 1 - 2 * M / r
    f_inv = 1 / f
    L = CONFIG["L"]
    E = CONFIG["E"]

    t_dot = E / f
    phi_dot = L / r**2

    # Derivative of effective potential squared
    # Veff^2 = (1 - 2M/r)(κ + L^2/r^2)
    # d/dr Veff^2 = d/dr [ (1 - 2M/r)(κ + L^2/r^2) ]
    # = (2M/r^2)(κ + L^2/r^2) + (1 - 2M/r)(-2L^2/r^3)
    dVeff2_dr = (2 * M / r**2) * (kappa + L**2 / r**2) + (1 - 2 * M / r) * (-2 * L**2 / r**3)
    r_ddot = 0.5 * dVeff2_dr * rdot

    return np.array([t_dot, rdot, phi_dot, r_ddot])

# ┌────────────────────────────────────┐
# │            STOPPING EVENTS         │
# └────────────────────────────────────┘

def event_horizon(tau, y):
    return y[1] - (r_h + horizon_eps)

event_horizon.terminal = True
event_horizon.direction = -1  # Trigger when r decreases through r_h + eps

def event_far(tau, y):
    return y[1] - r_max

event_far.terminal = True
event_far.direction = 1  # Trigger when r increases through r_max

# ┌────────────────────────────────────┐
# │            MAIN SIMULATION         │
# └────────────────────────────────────┘

def main():
    start_time = time.time()
    start_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2  # MB

    y0 = np.array([t0, r0, phi0, rdot0])
    tau_span = (0, tau_max)

    sol = solve_ivp(
        fun=geodesic_rhs,
        t_span=tau_span,
        y0=y0,
        method=CONFIG["integrator"],
        rtol=CONFIG["rtol"],
        atol=CONFIG["atol"],
        max_step=CONFIG["max_step"],
        events=[event_horizon, event_far],
        dense_output=False
    )

    end_time = time.time()
    end_memory = psutil.Process(os.getpid()).memory_info().rss / 1024**2  # MB

    # ┌────────────────────────────────────┐
    # │          POST-PROCESSING           │
    # └────────────────────────────────────┘

    # Cartesian coordinates
    r = sol.y[1]
    phi = sol.y[2]
    x = r * np.cos(phi)
    y = r * np.sin(phi)

    # ┌────────────────────────────────────┐
    # │            OUTPUT PATHS            │
    # └────────────────────────────────────┘

    output_dirs = CONFIG["output_dirs"]
    for path in output_dirs.values():
        Path(path).mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"

    csv_path = os.path.join(output_dirs["results"], f"{run_id}_trajectory.csv")
    fig_path = os.path.join(output_dirs["figures"], f"{run_id}_orbit.png")
    log_path = os.path.join(output_dirs["logs"], f"{run_id}_benchmark.json")

    # ┌────────────────────────────────────┐
    # │            SAVE CSV                │
    # └────────────────────────────────────┘

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["tau", "t", "r", "phi", "x", "y"])
        for i in range(len(sol.t)):
            writer.writerow([
                sol.t[i],
                sol.y[0][i],
                sol.y[1][i],
                sol.y[2][i],
                x[i],
                y[i]
            ])

    # ┌────────────────────────────────────┐
    # │            PLOT TRAJECTORY         │
    # └────────────────────────────────────┘

    fig, ax = plt.subplots(figsize=CONFIG["plot_options"]["figsize"], dpi=CONFIG["plot_options"]["dpi"])
    ax.plot(x, y, label="Trajectory", color='blue', linewidth=1)
    ax.set_aspect('equal')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.grid(True)

    if CONFIG["plot_options"]["draw_horizon"]:
        horizon_radius = r_h + horizon_eps
        circle = plt.Circle((0, 0), horizon_radius, color='black', fill=False, linestyle='--', linewidth=1, label="Horizon")
        ax.add_patch(circle)

    ax.legend()
    fig.savefig(fig_path, dpi=CONFIG["plot_options"]["dpi"])
    plt.close(fig)

    # ┌────────────────────────────────────┐
    # │            BENCHMARK LOG           │
    # └────────────────────────────────────┘

    benchmark = {
        "timestamp": timestamp,
        "runtime_seconds": end_time - start_time,
        "memory_start_mb": start_memory,
        "memory_end_mb": end_memory,
        "memory_delta_mb": end_memory - start_memory,
        "particles_simulated": 1,
        "integrator": CONFIG["integrator"],
        "steps_taken": len(sol.t),
        "termination_reason": "unknown"
    }

    if sol.status == 1:
        benchmark["termination_reason"] = "horizon_hit"
    elif sol.status == 0:
        benchmark["termination_reason"] = "max_time_or_steps"
    elif len(sol.t_events[1]) > 0:
        benchmark["termination_reason"] = "escaped_to_infinity"

    with open(log_path, "w") as f:
        json.dump(benchmark, f, indent=4)

    print(f"[INFO] Simulation complete.")
    print(f"[INFO] CSV saved to: {csv_path}")
    print(f"[INFO] Plot saved to: {fig_path}")
    print(f"[INFO] Benchmark saved to: {log_path}")

# ┌────────────────────────────────────┐
# │               ENTRY                │
# └────────────────────────────────────┘

if __name__ == "__main__":
    main()
