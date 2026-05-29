"""
QuantForge — Module 4: Execution Schedules
===========================================

Computes child-order sequences (execution trajectories) for a parent order
of X shares over N intervals.

Three algorithms:

  TWAP (Time-Weighted Average Price)
      Uniform schedule: trades X/N shares each interval.

  VWAP (Volume-Weighted Average Price)
      Participates in proportion to a historical (or modelled) intraday
      volume profile, typically U-shaped (high at open and close).

  AlmgrenChrissOptimal
      Solves the Almgren-Chriss (2001) mean-variance optimisation.
      Minimises E[cost] + λ · Var[cost] subject to full liquidation.

C++ backend (exec_cpp) is used when compiled; falls back to NumPy.

Reference
---------
Almgren, R. & Chriss, N. (2001). "Optimal Execution of Portfolio Transactions".
Journal of Risk, 3(2), 5–39.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
try:
    import exec_cpp as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


# ── Pure-Python fallbacks ────────────────────────────────────

def _py_ac_trajectory(X: float, sigma: float, eta: float, gamma: float,
                       lam: float, N: int, tau: float) -> dict:
    eta_tilde = eta - gamma * tau / 2.0
    if eta_tilde <= 0:
        raise ValueError("eta - gamma*tau/2 must be > 0")
    kappa = np.sqrt(lam * sigma**2 / eta_tilde)

    sinh_kNt = np.sinh(kappa * N * tau)
    if sinh_kNt < 1e-12:
        holdings = X * np.arange(N, -1, -1) / N
    else:
        holdings = X * np.sinh(kappa * np.arange(N, -1, -1) * tau) / sinh_kNt

    trades = np.diff(-holdings)   # n_k = x_k - x_{k+1}, positive for sales

    perm_cost = 0.5 * gamma * X**2
    temp_cost = eta / tau * np.sum(trades**2)
    timing_var = sigma**2 * tau * np.sum(holdings[:-1]**2)

    return {
        "name":               "Almgren-Chriss Optimal",
        "trades":             trades,
        "total_shares":       X,
        "expected_cost":      perm_cost + temp_cost,
        "expected_variance":  timing_var,
    }


def _py_vwap_volume_profile(N: int) -> np.ndarray:
    """U-shaped intraday volume profile."""
    t = (np.arange(N) + 0.5) / N
    vol = 0.5 + 2.0 * (t - 0.5)**2
    return vol / vol.sum()


# ── Schedule dataclass ───────────────────────────────────────

@dataclass
class ExecutionSchedule:
    """
    Represents a sequence of child orders for a parent order.

    Attributes
    ----------
    name             : str       Algorithm name
    trades           : np.ndarray (N,)  Shares traded per interval
    total_shares     : float     Sum of trades (should equal parent X)
    expected_cost    : float     Analytical expected IS (AC model only)
    expected_variance: float     Analytical IS variance (AC model only)
    """
    name              : str
    trades            : np.ndarray
    total_shares      : float
    expected_cost     : float = 0.0
    expected_variance : float = 0.0

    @property
    def N(self) -> int:
        return len(self.trades)

    @property
    def holdings(self) -> np.ndarray:
        """Remaining position at start of each interval, including initial."""
        return self.total_shares - np.concatenate([[0], np.cumsum(self.trades)])

    def participation_rate(self) -> np.ndarray:
        """Fraction of total shares traded per interval."""
        return self.trades / self.total_shares

    def __repr__(self) -> str:
        return (f"ExecutionSchedule('{self.name}', N={self.N}, "
                f"X={self.total_shares:.0f}, "
                f"E[cost]={self.expected_cost:.4f})")


# ── Schedule factories ───────────────────────────────────────

def twap(X: float, N: int) -> ExecutionSchedule:
    """
    Time-Weighted Average Price schedule.
    Trades equal-sized child orders: n_k = X / N for all k.

    Parameters
    ----------
    X : float  Total shares to execute
    N : int    Number of intervals
    """
    if _HAS_CPP:
        d = _cpp.twap_trajectory(X, N)
        return ExecutionSchedule(d["name"], np.array(d["trades"]),
                                 d["total_shares"])
    return ExecutionSchedule("TWAP", np.full(N, X / N), X)


def vwap(X: float, N: int,
          volume_profile: Optional[np.ndarray] = None) -> ExecutionSchedule:
    """
    Volume-Weighted Average Price schedule.
    Participates proportionally to expected intraday volume.

    Parameters
    ----------
    X              : float             Total shares to execute
    N              : int               Number of intervals
    volume_profile : array (N,) or None  Fraction of daily volume at each
                                          interval. Defaults to U-shaped profile.
    """
    if _HAS_CPP and volume_profile is None:
        d = _cpp.vwap_trajectory(X, N)
        return ExecutionSchedule(d["name"], np.array(d["trades"]),
                                 d["total_shares"])

    if volume_profile is None:
        volume_profile = _py_vwap_volume_profile(N)
    vol = np.asarray(volume_profile, dtype=float)
    vol = vol / vol.sum()
    return ExecutionSchedule("VWAP", X * vol, X)


def almgren_chriss(
    X:      float,
    sigma:  float,
    eta:    float,
    gamma:  float,
    lam:    float,
    N:      int,
    tau:    float = 1.0,
) -> ExecutionSchedule:
    """
    Almgren-Chriss optimal execution schedule.

    Parameters
    ----------
    X     : float  Total shares to liquidate (positive)
    sigma : float  Daily price volatility (fraction, e.g. 0.015 for 1.5%)
    eta   : float  Temporary impact coefficient ($/share²/day)
    gamma : float  Permanent impact coefficient ($/share)
    lam   : float  Risk aversion (λ); higher = more front-loading
    N     : int    Number of trading intervals
    tau   : float  Duration of each interval in days (default 1 day/N)

    Returns
    -------
    ExecutionSchedule
    """
    if _HAS_CPP:
        d = _cpp.ac_optimal_trajectory(X, sigma, eta, gamma, lam, N, tau)
        return ExecutionSchedule(
            d["name"], np.array(d["trades"]),
            d["total_shares"], d["expected_cost"], d["expected_variance"],
        )

    d = _py_ac_trajectory(X, sigma, eta, gamma, lam, N, tau)
    return ExecutionSchedule(
        d["name"], d["trades"], d["total_shares"],
        d["expected_cost"], d["expected_variance"],
    )
