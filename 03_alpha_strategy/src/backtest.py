"""
QuantForge — Module 3: Vectorized Backtesting Engine
=====================================================

Runs a signal-driven vectorized backtest with:
- Proportional transaction costs on signal changes
- Mark-to-market daily P&L
- Comprehensive performance metrics

References
----------
- Sharpe (1966) "Mutual Fund Performance"
- Sortino & van der Meer (1991) "Downside Risk"
- Calmar ratio (Young 1991)
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
    import alpha_cpp as _cpp
    _HAS_CPP = True
except ImportError:
    _HAS_CPP = False


# ── Performance metrics dataclass ────────────────────────────

@dataclass
class PerformanceMetrics:
    """Comprehensive risk-adjusted performance summary."""
    strategy_name  : str
    total_return   : float    # cumulative return over the period
    annual_return  : float    # annualised arithmetic return
    annual_vol     : float    # annualised volatility
    sharpe_ratio   : float    # annualised Sharpe (excess return / vol)
    sortino_ratio  : float    # Sortino (excess return / downside deviation)
    calmar_ratio   : float    # annual return / max drawdown
    max_drawdown   : float    # maximum peak-to-trough drawdown
    hit_rate       : float    # fraction of trading days with positive P&L
    profit_factor  : float    # gross profit / gross loss
    avg_win        : float    # average daily gain on winning days
    avg_loss       : float    # average daily loss on losing days
    n_trades       : int      # number of non-zero signal days
    n_obs          : int      # total trading days
    equity_curve   : pd.Series = field(repr=False, default_factory=pd.Series)
    drawdown_series: pd.Series = field(repr=False, default_factory=pd.Series)
    daily_pnl      : pd.Series = field(repr=False, default_factory=pd.Series)

    def __str__(self) -> str:
        bar = "-" * 52
        return (
            f"\n{bar}\n"
            f"  {self.strategy_name}\n"
            f"{bar}\n"
            f"  Observations       {self.n_obs:>10d}\n"
            f"  Trades (non-zero)  {self.n_trades:>10d}\n"
            f"{bar}\n"
            f"  Total Return       {self.total_return*100:>10.2f} %\n"
            f"  Annual Return      {self.annual_return*100:>10.2f} %\n"
            f"  Annual Vol         {self.annual_vol*100:>10.2f} %\n"
            f"{bar}\n"
            f"  Sharpe Ratio       {self.sharpe_ratio:>10.3f}\n"
            f"  Sortino Ratio      {self.sortino_ratio:>10.3f}\n"
            f"  Calmar Ratio       {self.calmar_ratio:>10.3f}\n"
            f"  Max Drawdown       {self.max_drawdown*100:>10.2f} %\n"
            f"{bar}\n"
            f"  Hit Rate           {self.hit_rate*100:>10.2f} %\n"
            f"  Profit Factor      {self.profit_factor:>10.3f}\n"
            f"  Avg Win            {self.avg_win*100:>10.4f} %\n"
            f"  Avg Loss           {self.avg_loss*100:>10.4f} %\n"
            f"{bar}"
        )


def _compute_metrics_py(
    pnl: np.ndarray,
    name: str,
    rfr: float = 0.0,
    trading_days: int = 252,
) -> dict:
    T = len(pnl)
    if T == 0:
        raise ValueError("Empty P&L series")

    mean_ret  = pnl.mean()
    std_ret   = pnl.std(ddof=1) if T > 1 else 1e-14

    # Equity curve & drawdown
    equity = np.cumprod(1 + pnl) - 1
    cum    = np.cumprod(1 + pnl)
    peak   = np.maximum.accumulate(cum)
    dd_ser = (cum - peak) / peak
    max_dd = -dd_ser.min()

    # Downside deviation
    rf_daily  = rfr / trading_days
    excess    = pnl - rf_daily
    downside  = excess[excess < 0]
    down_std  = np.sqrt((downside**2).mean()) if len(downside) > 0 else 1e-14

    # Trade stats
    nonzero   = pnl[pnl != 0.0]
    wins      = pnl[pnl > 0]
    losses    = pnl[pnl < 0]

    annual_ret = mean_ret * trading_days
    annual_vol = std_ret  * np.sqrt(trading_days)
    sharpe     = (annual_ret - rfr) / annual_vol if annual_vol > 1e-14 else 0.0
    sortino    = (annual_ret - rfr) / (down_std * np.sqrt(trading_days)) \
                 if down_std > 1e-14 else 0.0
    calmar     = annual_ret / max_dd if max_dd > 1e-14 else 0.0
    hit_rate   = len(wins) / len(nonzero) if len(nonzero) > 0 else 0.0
    pf         = wins.sum() / abs(losses.sum()) if len(losses) > 0 and abs(losses.sum()) > 1e-14 else 0.0

    return {
        "total_return": equity[-1],
        "annual_return": annual_ret,
        "annual_vol": annual_vol,
        "sharpe_ratio": sharpe,
        "sortino_ratio": sortino,
        "calmar_ratio": calmar,
        "max_drawdown": max_dd,
        "hit_rate": hit_rate,
        "profit_factor": pf,
        "avg_win":  wins.mean()   if len(wins)   > 0 else 0.0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0.0,
        "n_trades": len(nonzero),
        "n_obs": T,
        "equity": equity,
        "drawdown": dd_ser,
    }


# ── Backtest engine ──────────────────────────────────────────

class Backtest:
    """
    Vectorized backtesting engine.

    Parameters
    ----------
    pnl_series    : pd.Series  Daily P&L series (pre-cost)
    transaction_cost : float  One-way cost as a fraction of trade value
                              Applied whenever the signal changes
    signal_series : pd.Series  Signal driving the strategy (for cost calc)
    name          : str        Strategy display name
    rfr           : float      Annualised risk-free rate (for Sharpe)
    """

    def __init__(
        self,
        pnl_series:      pd.Series,
        signal_series:   Optional[pd.Series] = None,
        transaction_cost: float = 2e-4,
        name:  str   = "Strategy",
        rfr:   float = 0.0,
    ):
        self.pnl_raw          = pnl_series.astype(float)
        self.signal_series    = signal_series
        self.transaction_cost = transaction_cost
        self.name             = name
        self.rfr              = rfr
        self._metrics: Optional[PerformanceMetrics] = None

    def run(self) -> PerformanceMetrics:
        """Compute all performance metrics and return a PerformanceMetrics object."""
        pnl = self.pnl_raw.copy()

        # Subtract transaction costs on signal changes
        if self.signal_series is not None:
            common  = pnl.index.intersection(self.signal_series.index)
            sig     = self.signal_series.loc[common]
            changes = sig.diff().abs().fillna(0)
            costs   = changes * self.transaction_cost
            pnl     = (pnl.loc[common] - costs).fillna(0)

        pnl = pnl.fillna(0)
        arr = pnl.values.astype(np.float64)

        # Use C++ if available
        if _HAS_CPP:
            d = _cpp.compute_metrics(arr, self.rfr, 252)
            equity  = pd.Series(np.cumprod(1 + arr) - 1, index=pnl.index)
            cum     = np.cumprod(1 + arr)
            peak    = np.maximum.accumulate(cum)
            dd_ser  = pd.Series((cum - peak) / peak, index=pnl.index)
        else:
            d      = _compute_metrics_py(arr, self.name, self.rfr)
            equity = pd.Series(d.pop("equity"),   index=pnl.index)
            dd_ser = pd.Series(d.pop("drawdown"), index=pnl.index)

        self._metrics = PerformanceMetrics(
            strategy_name   = self.name,
            total_return    = d["total_return"],
            annual_return   = d["annual_return"],
            annual_vol      = d["annual_vol"],
            sharpe_ratio    = d["sharpe_ratio"],
            sortino_ratio   = d["sortino_ratio"],
            calmar_ratio    = d["calmar_ratio"],
            max_drawdown    = d["max_drawdown"],
            hit_rate        = d["hit_rate"],
            profit_factor   = d["profit_factor"],
            avg_win         = d["avg_win"],
            avg_loss        = d["avg_loss"],
            n_trades        = d["n_trades"],
            n_obs           = d["n_obs"],
            equity_curve    = equity,
            drawdown_series = dd_ser,
            daily_pnl       = pnl,
        )
        return self._metrics

    @property
    def metrics(self) -> PerformanceMetrics:
        if self._metrics is None:
            return self.run()
        return self._metrics


# ── Benchmark comparator ─────────────────────────────────────

def compare_strategies(
    strategies: list[PerformanceMetrics],
) -> pd.DataFrame:
    """Return a comparison DataFrame with key metrics for each strategy."""
    rows = []
    for m in strategies:
        rows.append({
            "Total Return (%)":  round(m.total_return   * 100, 2),
            "Annual Return (%)": round(m.annual_return  * 100, 2),
            "Annual Vol (%)":    round(m.annual_vol     * 100, 2),
            "Sharpe":            round(m.sharpe_ratio,          3),
            "Sortino":           round(m.sortino_ratio,         3),
            "Calmar":            round(m.calmar_ratio,          3),
            "Max DD (%)":        round(m.max_drawdown   * 100, 2),
            "Hit Rate (%)":      round(m.hit_rate       * 100, 2),
            "Profit Factor":     round(m.profit_factor,         3),
        })
    return pd.DataFrame(rows, index=[m.strategy_name for m in strategies])
