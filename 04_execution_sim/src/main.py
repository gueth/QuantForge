"""
QuantForge — Module 4: Execution Simulator Demo
================================================
End-to-end demonstration of the execution simulation pipeline:
  1. Define market impact parameters
  2. Build TWAP, VWAP and Almgren-Chriss schedules
  3. Monte Carlo simulation of implementation shortfall
  4. Efficient frontier: E[IS] vs σ[IS] as λ varies
  5. Strategy comparison
  6. Visualizations

Usage
-----
    cd 04_execution_sim/src
    python main.py
"""
from __future__ import annotations

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure UTF-8 output on Windows cp1252 terminals
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from schedules import twap, vwap, almgren_chriss
from simulator import ExecutionSimulator, compare_schedules
from visualizer import (
    plot_schedule_dashboard, plot_is_dashboard, plot_efficient_frontier,
)


def sep(title: str = "") -> None:
    w = 58
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'-'*pad} {title} {'-'*(w - pad - len(title) - 2)}")
    else:
        print("-" * w)


def run():
    t0 = time.perf_counter()

    print("\n" + "=" * 58)
    print("  QuantForge - Module 4: Execution Simulator")
    print("=" * 58)

    # ── 1. Market parameters ──────────────────────────────────
    sep("PARAMETERS")

    X             = 100_000   # shares to liquidate
    N             = 20        # 20 trading intervals (e.g. 20 minutes)
    sigma         = 0.015     # 1.5% daily vol
    eta           = 2.5e-7    # temporary impact
    gamma         = 2.5e-8    # permanent impact
    lam           = 1e-6      # risk aversion
    tau           = 1.0 / N   # each interval = 1/20th of a day
    arrival_price = 100.0

    print(f"\n  Parent order : {X:,} shares at ${arrival_price:.2f}")
    print(f"  Market params: σ={sigma*100:.1f}%  η={eta:.2e}  γ={gamma:.2e}")
    print(f"  Intervals    : N={N}  τ={tau:.4f} days  λ={lam:.2e}")

    # ── 2. Build Schedules ────────────────────────────────────
    sep("EXECUTION SCHEDULES")

    sched_twap = twap(X, N)
    sched_vwap = vwap(X, N)
    sched_ac   = almgren_chriss(X, sigma=sigma, eta=eta, gamma=gamma,
                                 lam=lam, N=N, tau=tau)

    print(f"\n  {'':<26} {'Front-load':>11} {'Back-load':>11}")
    print(f"  {'-'*50}")
    for sched in [sched_twap, sched_vwap, sched_ac]:
        fl = sched.trades[:N//2].sum() / X * 100
        bl = sched.trades[N//2:].sum() / X * 100
        print(f"  {sched.name:<26} {fl:>10.1f}% {bl:>10.1f}%")
    print(f"\n  AC expected cost:  ${sched_ac.expected_cost:.2f}")
    print(f"  AC expected var:   {sched_ac.expected_variance:.4f}")

    # ── 3. Monte Carlo Simulation ─────────────────────────────
    sep("MONTE CARLO SIMULATION  (10,000 paths)")

    sim = ExecutionSimulator(
        sigma=sigma, eta=eta, gamma=gamma, tau=tau,
        arrival_price=arrival_price, n_paths=10_000, seed=42,
    )

    rep_twap = sim.simulate(sched_twap)
    rep_vwap = sim.simulate(sched_vwap)
    rep_ac   = sim.simulate(sched_ac, lam=lam)

    for rep in [rep_twap, rep_vwap, rep_ac]:
        print(rep)

    # Summary comparison
    sep("SCHEDULE COMPARISON")
    comp = compare_schedules([rep_twap, rep_vwap, rep_ac])
    print(f"\n{comp.to_string()}")

    # ── 4. Efficient Frontier ─────────────────────────────────
    sep("EFFICIENT FRONTIER  (λ sweep)")

    lambdas = np.logspace(-8, -4, 30)
    frontier = []
    print(f"  Computing {len(lambdas)} AC schedules...")

    for lam_i in lambdas:
        s   = almgren_chriss(X, sigma=sigma, eta=eta, gamma=gamma,
                              lam=lam_i, N=N, tau=tau)
        r   = sim.simulate(s, lam=lam_i)
        frontier.append({
            "lambda":  lam_i,
            "mean_is": r.mean_is_bps,
            "std_is":  r.std_is_bps,
        })

    min_idx = np.argmin([d["mean_is"] for d in frontier])
    mid_idx = len(frontier) // 2
    print(f"\n  λ = {lambdas[min_idx]:.2e}  → min E[IS]  = "
          f"{frontier[min_idx]['mean_is']:.1f} bps  "
          f"(σ = {frontier[min_idx]['std_is']:.1f} bps)")
    print(f"  λ = {lambdas[mid_idx]:.2e}  → E[IS]      = "
          f"{frontier[mid_idx]['mean_is']:.1f} bps  "
          f"(σ = {frontier[mid_idx]['std_is']:.1f} bps)")

    # ── 5. Impact Parameter Sensitivity ──────────────────────
    sep("SENSITIVITY ANALYSIS")

    eta_values = [1e-7, 2.5e-7, 5e-7, 1e-6]
    print(f"\n  Effect of temporary impact η on IS  (TWAP, N={N}):")
    print(f"\n  {'η':>10} {'E[IS] (bps)':>14} {'σ[IS] (bps)':>14}")
    print(f"  {'-'*40}")
    for eta_i in eta_values:
        sim_i  = ExecutionSimulator(sigma=sigma, eta=eta_i, gamma=gamma,
                                    tau=tau, arrival_price=arrival_price,
                                    n_paths=5000, seed=42)
        r_i    = sim_i.simulate(sched_twap)
        print(f"  {eta_i:>10.2e} {r_i.mean_is_bps:>14.2f} {r_i.std_is_bps:>14.2f}")

    # ── 6. Visualizations ─────────────────────────────────────
    sep("VISUALIZATIONS")

    print("  Schedule dashboard...")
    plot_schedule_dashboard([sched_twap, sched_vwap, sched_ac])

    print("  Implementation Shortfall dashboard...")
    plot_is_dashboard([rep_twap, rep_vwap, rep_ac])

    print("  Efficient frontier...")
    plot_efficient_frontier(frontier, rep_twap, rep_vwap)

    # ── Done ──────────────────────────────────────────────────
    sep()
    elapsed = time.perf_counter() - t0
    print(f"\n  Module 4 complete -- {elapsed:.2f}s")
    print(f"  Charts saved to: {Path(__file__).parent.parent / 'notebooks'}/\n")


if __name__ == "__main__":
    run()
