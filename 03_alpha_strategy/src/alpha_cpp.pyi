"""
Type stub for the compiled C++ extension alpha_cpp.

Mirrors the bindings defined in cpp/kalman_bindings.cpp.
"""
import numpy as np

def kalman_filter(
    y:     np.ndarray,
    x:     np.ndarray,
    delta: float = ...,
    Ve:    float = ...,
    beta0: float = ...,
    P0:    float = ...,
) -> dict[str, np.ndarray]:
    """
    Scalar Kalman filter for dynamic hedge ratio estimation.

    Returns
    -------
    dict with keys: betas, spreads, innovations, innovations_var
    """
    ...

def rolling_zscore(
    series: np.ndarray,
    window: int,
) -> np.ndarray:
    """Rolling z-score of a time series. First (window-1) values are NaN."""
    ...

def compute_metrics(
    daily_pnl:       np.ndarray,
    risk_free_rate:  float = ...,
    trading_days:    int   = ...,
) -> dict[str, float]:
    """
    Compute performance metrics from a daily P&L series.

    Returns
    -------
    dict with keys: total_return, annual_return, annual_vol, sharpe_ratio,
        sortino_ratio, calmar_ratio, max_drawdown, hit_rate, profit_factor,
        avg_win, avg_loss, n_trades, n_obs
    """
    ...
