"""
QuantForge — Module 4: Execution Simulator
============================================

Simulates order execution along a given schedule and estimates the
Implementation Shortfall (IS) distribution via Monte Carlo.

Implementation Shortfall (IS)
    IS = (average execution price / arrival mid-price) - 1
       = permanent impact + temporary impact + timing risk

Components:
  Permanent impact: γ · Σ n_k   — lasting price move per share traded
  Temporary impact: η · (n_k/τ) — price move during interval k, reverts fully
  Timing risk:      σ√τ · dW    — Brownian price fluctuation between trades

Reference: Almgren & Chriss (2001).
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
try:
    import exec_cpp as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False

from schedules import ExecutionSchedule


# ── ExecutionReport ──────────────────────────────────────────

@dataclass
class ExecutionReport:
    """
    Summary of a Monte Carlo execution simulation.

    All costs are expressed in $/share (i.e., as a fraction of arrival price
    when arrival_price=1.0, or as cents if expressed per $100).
    """
    schedule_name    : str
    total_shares     : float
    n_intervals      : int
    n_paths          : int
    arrival_price    : float

    mean_is          : float    # mean implementation shortfall ($/share)
    std_is           : float    # std deviation of IS
    var_95           : float    # 95th percentile IS (execution cost VaR)
    cvar_95          : float    # conditional VaR at 95%
    mean_perm_impact : float    # mean permanent impact component
    mean_temp_impact : float    # mean temporary impact component
    mean_timing_risk : float    # mean timing risk component

    is_samples       : np.ndarray = field(repr=False, default_factory=lambda: np.array([]))

    @property
    def mean_is_bps(self) -> float:
        """Mean IS in basis points."""
        return self.mean_is / self.arrival_price * 1e4

    @property
    def std_is_bps(self) -> float:
        return self.std_is / self.arrival_price * 1e4

    def __str__(self) -> str:
        bar = "─" * 52
        return (
            f"\n{bar}\n"
            f"  {self.schedule_name}  |  {self.total_shares:.0f} shares  "
            f"|  {self.n_intervals} intervals\n"
            f"{bar}\n"
            f"  Mean IS              {self.mean_is:.4f}  ({self.mean_is_bps:.1f} bps)\n"
            f"  Std  IS              {self.std_is:.4f}  ({self.std_is_bps:.1f} bps)\n"
            f"  95% VaR of IS        {self.var_95:.4f}\n"
            f"  95% CVaR of IS       {self.cvar_95:.4f}\n"
            f"{bar}\n"
            f"  Perm. Impact         {self.mean_perm_impact:.4f}\n"
            f"  Temp. Impact         {self.mean_temp_impact:.4f}\n"
            f"  Timing Risk          {self.mean_timing_risk:.4f}\n"
            f"{bar}"
        )


# ── Pure-Python MC fallback ──────────────────────────────────

def _py_simulate(
    trades: np.ndarray,
    sigma: float, eta: float, gamma: float, tau: float,
    arrival_price: float,
    n_paths: int, seed: int,
) -> dict:
    N = len(trades)
    X = trades.sum()
    rng = np.random.default_rng(seed)

    is_s    = np.empty(n_paths)
    perm_s  = np.empty(n_paths)
    temp_s  = np.empty(n_paths)

    for p in range(n_paths):
        price      = arrival_price
        total_exec = 0.0
        perm_sum   = 0.0
        temp_sum   = 0.0

        for k in range(N):
            n_k = trades[k]
            dp_perm  = gamma * n_k
            dp_noise = sigma * np.sqrt(tau) * rng.standard_normal()
            price   += dp_perm + dp_noise

            exec_premium = eta * (n_k / tau)
            exec_price   = price + exec_premium

            total_exec += n_k * exec_price
            perm_sum   += n_k * dp_perm
            temp_sum   += n_k * exec_premium

        is_s[p]   = total_exec / X - arrival_price
        perm_s[p] = perm_sum  / X
        temp_s[p] = temp_sum  / X

    mean_is    = is_s.mean()
    std_is     = is_s.std(ddof=1)
    sorted_is  = np.sort(is_s)
    idx95      = int(0.95 * n_paths)
    var_95     = sorted_is[idx95]
    cvar_95    = sorted_is[idx95:].mean()

    return {
        "mean_is":          mean_is,
        "std_is":           std_is,
        "var_95":           var_95,
        "cvar_95":          cvar_95,
        "mean_perm_impact": perm_s.mean(),
        "mean_temp_impact": temp_s.mean(),
        "mean_timing_risk": std_is - abs(perm_s.mean() + temp_s.mean()),
        "n_paths":          n_paths,
        "samples":          is_s,
    }


# ── Simulator ────────────────────────────────────────────────

class ExecutionSimulator:
    """
    Monte Carlo execution simulator.

    Parameters
    ----------
    sigma         : float  Daily price volatility (fraction)
    eta           : float  Temporary market impact coefficient
    gamma         : float  Permanent market impact coefficient
    tau           : float  Interval duration in days
    arrival_price : float  Mid-price at order arrival
    n_paths       : int    Number of Monte Carlo paths
    seed          : int    Random seed for reproducibility
    """

    def __init__(
        self,
        sigma:         float = 0.015,
        eta:           float = 2.5e-7,
        gamma:         float = 2.5e-8,
        tau:           float = 1.0,
        arrival_price: float = 100.0,
        n_paths:       int   = 10_000,
        seed:          int   = 42,
    ):
        self.sigma         = sigma
        self.eta           = eta
        self.gamma         = gamma
        self.tau           = tau
        self.arrival_price = arrival_price
        self.n_paths       = n_paths
        self.seed          = seed

    def simulate(
        self,
        schedule: ExecutionSchedule,
        lam: float = 0.0,    # unused in Python fallback but needed for C++ call
    ) -> ExecutionReport:
        """
        Simulate execution of the given schedule and return an ExecutionReport.
        """
        trades = schedule.trades.astype(np.float64)

        if _HAS_CPP:
            d = _cpp.simulate_execution(
                trades, self.sigma, self.eta, self.gamma,
                lam, schedule.N, self.tau,
                self.arrival_price, self.n_paths, self.seed,
            )
        else:
            d = _py_simulate(
                trades, self.sigma, self.eta, self.gamma, self.tau,
                self.arrival_price, self.n_paths, self.seed,
            )

        return ExecutionReport(
            schedule_name    = schedule.name,
            total_shares     = schedule.total_shares,
            n_intervals      = schedule.N,
            n_paths          = self.n_paths,
            arrival_price    = self.arrival_price,
            mean_is          = d["mean_is"],
            std_is           = d["std_is"],
            var_95           = d["var_95"],
            cvar_95          = d["cvar_95"],
            mean_perm_impact = d["mean_perm_impact"],
            mean_temp_impact = d["mean_temp_impact"],
            mean_timing_risk = d["mean_timing_risk"],
            is_samples       = np.array(d["samples"]),
        )

    def simulate_multiple(
        self, schedules: list[ExecutionSchedule], lam: float = 0.0,
    ) -> list[ExecutionReport]:
        """Simulate a list of schedules with the same market parameters."""
        return [self.simulate(s, lam) for s in schedules]


# ── Synthetic market data ────────────────────────────────────

def synthetic_intraday_volume(
    n_intervals: int = 78,   # 5-min intervals in a 6.5h trading day
    seed: int = 0,
) -> np.ndarray:
    """
    Generate a realistic intraday volume profile.
    Returns an array summing to 1.0, U-shaped with noise.
    """
    rng = np.random.default_rng(seed)
    t   = np.linspace(0, 1, n_intervals)
    vol = 0.4 + 1.5 * (t - 0.5)**2 + 0.05 * rng.standard_normal(n_intervals)
    vol = np.maximum(vol, 0.01)
    return vol / vol.sum()


def compare_schedules(
    reports: list[ExecutionReport],
) -> pd.DataFrame:
    """Return a comparison DataFrame for multiple execution schedules."""
    rows = []
    for r in reports:
        rows.append({
            "Mean IS (bps)":   round(r.mean_is_bps,      2),
            "Std IS (bps)":    round(r.std_is_bps,       2),
            "95% VaR (bps)":   round(r.var_95 / r.arrival_price * 1e4, 2),
            "Perm. (bps)":     round(r.mean_perm_impact / r.arrival_price * 1e4, 2),
            "Temp. (bps)":     round(r.mean_temp_impact / r.arrival_price * 1e4, 2),
            "Timing (bps)":    round(r.mean_timing_risk / r.arrival_price * 1e4, 2),
        })
    return pd.DataFrame(rows, index=[r.schedule_name for r in reports])
