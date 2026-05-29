"""
QuantForge — Module 3: Alpha Signal Generators
================================================

Three signal families:

  KalmanPairsSignal    — Dynamic pairs trading via scalar Kalman filter.
                          Tracks a time-varying hedge ratio β_t and generates
                          z-score signals on the residual spread.

  CrossSectionalMomentum — Long/short cross-sectional momentum factor.
                          Ranks assets on trailing return; goes long the top
                          quartile and short the bottom quartile.

  ZScoreSignal          — Simple rolling z-score mean-reversion signal for
                          a single price or spread series.

C++ backend (alpha_cpp) is used when compiled; falls back to NumPy otherwise.
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Optional C++ backend
sys.path.insert(0, str(Path(__file__).parent))
try:
    import alpha_cpp as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


# ── Pure-Python Kalman fallback ──────────────────────────────

def _py_kalman_filter(y: np.ndarray, x: np.ndarray,
                      delta: float, Ve: float,
                      beta0: float = 0.0, P0: float = 1.0) -> dict:
    T = len(y)
    Q = delta / (1.0 - delta)
    betas            = np.empty(T)
    spreads          = np.empty(T)
    innovations      = np.empty(T)
    innovations_var  = np.empty(T)

    beta, P = beta0, P0
    for t in range(T):
        P_pred = P + Q
        innov  = y[t] - beta * x[t]
        S      = x[t] * x[t] * P_pred + Ve
        K      = P_pred * x[t] / S
        beta   = beta + K * innov
        P      = (1.0 - K * x[t]) * P_pred

        betas[t]           = beta
        spreads[t]         = y[t] - beta * x[t]
        innovations[t]     = innov
        innovations_var[t] = S

    return {"betas": betas, "spreads": spreads,
            "innovations": innovations, "innovations_var": innovations_var}


def _py_rolling_zscore(series: np.ndarray, window: int) -> np.ndarray:
    T = len(series)
    out = np.full(T, np.nan)
    for t in range(window - 1, T):
        chunk = series[t - window + 1: t + 1]
        std   = chunk.std()
        if std > 1e-14:
            out[t] = (series[t] - chunk.mean()) / std
    return out


# ── KalmanPairsSignal ────────────────────────────────────────

@dataclass
class KalmanPairsSignal:
    """
    Statistical arbitrage signal via a scalar Kalman filter.

    The filter tracks a time-varying hedge ratio β_t so that:
        spread_t = log(y_t) - β_t · log(x_t)

    A z-score of the spread drives the long/short signal:
        signal = +1  when z < -enter_z  (spread too low  → buy spread)
        signal = -1  when z >  enter_z  (spread too high → sell spread)
        signal = 0   when |z| < exit_z  (reversion occurred)

    Parameters
    ----------
    delta       : float  Process noise factor; smaller = slower β drift
    Ve          : float  Observation noise variance
    zscore_window : int  Rolling window for z-score normalisation
    enter_z     : float  Z-score threshold to enter a position
    exit_z      : float  Z-score threshold to close a position
    """
    delta         : float = 1e-5
    Ve            : float = 1e-3
    zscore_window : int   = 30
    enter_z       : float = 2.0
    exit_z        : float = 0.5

    # Fitted attributes
    betas_        : Optional[pd.Series] = field(default=None, repr=False)
    spreads_      : Optional[pd.Series] = field(default=None, repr=False)
    zscore_       : Optional[pd.Series] = field(default=None, repr=False)
    signal_       : Optional[pd.Series] = field(default=None, repr=False)

    def fit(self, y: pd.Series, x: pd.Series) -> "KalmanPairsSignal":
        """
        Run the Kalman filter on log-price series y and x.

        Parameters
        ----------
        y : pd.Series  Log prices of asset A (dependent)
        x : pd.Series  Log prices of asset B (independent)

        Returns self for chaining.
        """
        idx  = y.index.intersection(x.index)
        yv   = np.log(y.loc[idx].values.astype(np.float64))
        xv   = np.log(x.loc[idx].values.astype(np.float64))

        if _HAS_CPP:
            res = _cpp.kalman_filter(yv, xv, self.delta, self.Ve)
        else:
            res = _py_kalman_filter(yv, xv, self.delta, self.Ve)

        self.betas_   = pd.Series(res["betas"],   index=idx, name="hedge_ratio")
        self.spreads_ = pd.Series(res["spreads"],  index=idx, name="spread")

        # Z-score of the spread
        sp = self.spreads_.values
        if _HAS_CPP:
            zs = _cpp.rolling_zscore(sp, self.zscore_window)
        else:
            zs = _py_rolling_zscore(sp, self.zscore_window)

        self.zscore_ = pd.Series(zs, index=idx, name="zscore")
        self.signal_ = self._build_signal(self.zscore_)
        return self

    def _build_signal(self, zscore: pd.Series) -> pd.Series:
        """Convert z-score series into {-1, 0, +1} positions."""
        z = zscore.values
        pos = np.zeros(len(z))
        current = 0

        for t in range(len(z)):
            if np.isnan(z[t]):
                pos[t] = 0
                continue
            if current == 0:
                if z[t] >  self.enter_z:
                    current = -1
                elif z[t] < -self.enter_z:
                    current = +1
            else:
                if abs(z[t]) < self.exit_z:
                    current = 0
            pos[t] = current

        return pd.Series(pos, index=zscore.index, name="signal")

    def pnl_series(self, y: pd.Series, x: pd.Series) -> pd.Series:
        """
        Compute daily P&L of the pairs strategy.

        Long spread:  +1 unit of A, -β units of B
        Short spread: -1 unit of A, +β units of B
        """
        if self.signal_ is None:
            raise RuntimeError("Call .fit() first.")

        common = y.index.intersection(x.index).intersection(self.signal_.index)
        sig    = self.signal_.loc[common].shift(1).fillna(0)   # trade next day
        betas  = self.betas_.loc[common].shift(1).fillna(0)

        ret_y  = y.loc[common].pct_change().fillna(0)
        ret_x  = x.loc[common].pct_change().fillna(0)

        pnl = sig * (ret_y - betas * ret_x)
        pnl.name = "pairs_pnl"
        return pnl


# ── CrossSectionalMomentum ───────────────────────────────────

@dataclass
class CrossSectionalMomentum:
    """
    Cross-sectional momentum factor.

    Ranks N assets on their trailing `lookback`-day return.
    Goes equally long the top `quantile` fraction and equally
    short the bottom `quantile` fraction, dollar-neutral.

    Parameters
    ----------
    lookback : int    Momentum lookback period (days)
    quantile : float  Fraction of assets in each leg (e.g. 0.25 = top/bottom quartile)
    """
    lookback : int   = 21
    quantile : float = 0.25

    signal_  : Optional[pd.DataFrame] = field(default=None, repr=False)

    def fit(self, returns: pd.DataFrame) -> "CrossSectionalMomentum":
        """
        Compute signals for all assets.

        Parameters
        ----------
        returns : pd.DataFrame  Daily returns (T × N)
        """
        momentum = (1 + returns).rolling(self.lookback).apply(
            np.prod, raw=True
        ) - 1

        n      = returns.shape[1]
        n_long = max(1, int(np.floor(n * self.quantile)))

        def rank_to_signal(row: pd.Series) -> pd.Series:
            if row.isna().all():
                return pd.Series(0.0, index=row.index)
            ranked = row.rank(ascending=True, na_option="keep")
            sig    = pd.Series(0.0, index=row.index)
            sig[ranked <= n_long]              = -1.0 / n_long   # short bottom
            sig[ranked >= (n - n_long + 1)]    =  1.0 / n_long   # long top
            return sig

        self.signal_ = momentum.apply(rank_to_signal, axis=1)
        self.signal_.name = None
        return self

    def pnl_series(self, returns: pd.DataFrame) -> pd.Series:
        """
        Daily P&L of the long/short momentum portfolio.
        Signals are shifted by 1 day to avoid look-ahead.
        """
        if self.signal_ is None:
            raise RuntimeError("Call .fit() first.")
        common = returns.index.intersection(self.signal_.index)
        sig    = self.signal_.loc[common].shift(1).fillna(0)
        pnl    = (sig * returns.loc[common]).sum(axis=1)
        pnl.name = "momentum_pnl"
        return pnl


# ── ZScoreSignal ─────────────────────────────────────────────

@dataclass
class ZScoreSignal:
    """
    Rolling z-score mean-reversion signal for a single series.

    Parameters
    ----------
    window  : int    Rolling window for mean and std
    enter_z : float  Enter threshold (|z| > enter_z)
    exit_z  : float  Exit threshold (|z| < exit_z)
    """
    window  : int   = 30
    enter_z : float = 2.0
    exit_z  : float = 0.5

    zscore_ : Optional[pd.Series] = field(default=None, repr=False)
    signal_ : Optional[pd.Series] = field(default=None, repr=False)

    def fit(self, prices: pd.Series) -> "ZScoreSignal":
        vals = prices.values.astype(np.float64)
        if _HAS_CPP:
            zs = _cpp.rolling_zscore(vals, self.window)
        else:
            zs = _py_rolling_zscore(vals, self.window)
        self.zscore_ = pd.Series(zs, index=prices.index, name="zscore")

        # Mean-reversion: fade extremes
        z    = self.zscore_.values
        pos  = np.zeros(len(z))
        curr = 0
        for t in range(len(z)):
            if np.isnan(z[t]):
                pos[t] = 0
                continue
            if curr == 0:
                if z[t] >  self.enter_z:
                    curr = -1
                elif z[t] < -self.enter_z:
                    curr = +1
            else:
                if abs(z[t]) < self.exit_z:
                    curr = 0
            pos[t] = curr

        self.signal_ = pd.Series(pos, index=prices.index, name="signal")
        return self

    def pnl_series(self, returns: pd.Series) -> pd.Series:
        if self.signal_ is None:
            raise RuntimeError("Call .fit() first.")
        common = returns.index.intersection(self.signal_.index)
        pnl    = self.signal_.loc[common].shift(1).fillna(0) * returns.loc[common]
        pnl.name = "zscore_pnl"
        return pnl


# ── Synthetic data generators ────────────────────────────────

def synthetic_pair(
    n_days: int = 1000,
    start:  str  = "2020-01-01",
    beta:   float = 0.8,
    spread_vol: float = 0.005,
    seed:   int  = 42,
) -> tuple[pd.Series, pd.Series]:
    """
    Generate a cointegrated pair (y, x) for testing.

    y_t = β · x_t + spread_t
    x follows a geometric Brownian motion; spread is mean-reverting (OU process).
    """
    rng   = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n_days)

    # Asset x: GBM
    ret_x  = rng.normal(2e-4, 0.01, n_days)
    log_x  = np.cumsum(ret_x) + np.log(100.0)

    # Mean-reverting spread (Ornstein-Uhlenbeck)
    theta_ou, kappa_ou = 0.0, 0.05
    spread = np.zeros(n_days)
    spread[0] = 0.0
    for t in range(1, n_days):
        spread[t] = (spread[t - 1]
                     + kappa_ou * (theta_ou - spread[t - 1])
                     + spread_vol * rng.standard_normal())

    log_y = beta * log_x + spread
    return (
        pd.Series(np.exp(log_y), index=dates, name="ASSET_A"),
        pd.Series(np.exp(log_x), index=dates, name="ASSET_B"),
    )


def synthetic_universe(
    n_assets: int = 10,
    n_days:   int = 1000,
    start:    str  = "2020-01-01",
    seed:     int  = 42,
) -> pd.DataFrame:
    """
    Generate a universe of correlated daily returns for momentum testing.
    """
    rng      = np.random.default_rng(seed)
    tickers  = [f"STOCK_{i+1:02d}" for i in range(n_assets)]
    dates    = pd.bdate_range(start=start, periods=n_days)

    A   = rng.standard_normal((n_assets, n_assets))
    cov = A @ A.T / n_assets + np.eye(n_assets) * 0.02
    L   = np.linalg.cholesky(cov)
    mu  = rng.uniform(0, 4e-4, n_assets)
    vol = rng.uniform(0.01, 0.025, n_assets)
    z   = rng.standard_normal((n_days, n_assets))

    returns = mu + (z @ L.T) * vol
    return pd.DataFrame(returns, index=dates, columns=tickers)
