"""
Type stub for the compiled C++ extension exec_cpp.

Mirrors the bindings defined in cpp/exec_bindings.cpp.
"""
from typing import Any
import numpy as np

def ac_optimal_trajectory(
    X:       float,
    sigma:   float,
    eta:     float,
    gamma:   float,
    lambda_: float,
    N:       int,
    tau:     float,
) -> dict[str, Any]:
    """
    Almgren-Chriss optimal execution trajectory.

    Returns
    -------
    dict with keys: name, trades (np.ndarray), total_shares,
        expected_cost, expected_variance
    """
    ...

def twap_trajectory(
    X: float,
    N: int,
) -> dict[str, Any]:
    """Uniform TWAP schedule: n_k = X/N for all k."""
    ...

def vwap_trajectory(
    X: float,
    N: int,
) -> dict[str, Any]:
    """VWAP schedule using a U-shaped intraday volume profile."""
    ...

def simulate_execution(
    trades:        np.ndarray,
    sigma:         float,
    eta:           float,
    gamma:         float,
    lambda_:       float,
    N:             int,
    tau:           float,
    arrival_price: float = ...,
    n_paths:       int   = ...,
    seed:          int   = ...,
) -> dict[str, Any]:
    """
    Monte Carlo simulation of implementation shortfall.

    Returns
    -------
    dict with keys: mean_is, std_is, var_95, cvar_95,
        mean_perm_impact, mean_temp_impact, mean_timing_risk,
        n_paths, samples (np.ndarray)
    """
    ...
