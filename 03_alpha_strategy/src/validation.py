"""
QuantForge — Module 3: Model Validation
=========================================

Tools for out-of-sample evaluation of trading strategies,
designed to avoid common pitfalls in financial machine learning.

PurgedKFold
    Implements purged k-fold cross-validation from
    Lopez de Prado (2018), "Advances in Financial Machine Learning".
    Removes training observations within an embargo window of the test
    set to prevent leakage due to serial correlation.

WalkForwardAnalysis
    Expanding or rolling window walk-forward analysis.
    The model is re-fitted at each step on the training window and
    evaluated on the following out-of-sample period.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generator, Optional

import numpy as np
import pandas as pd


# ── PurgedKFold ──────────────────────────────────────────────

@dataclass
class PurgedKFold:
    """
    Purged k-fold cross-validation for financial time series.

    Prevents information leakage by removing observations near the
    boundary between train and test folds.

    Parameters
    ----------
    n_splits    : int    Number of folds (≥ 2)
    purge_pct   : float  Fraction of training observations to purge
                         immediately before the test fold
    embargo_pct : float  Additional fraction to embargo after the test fold
    """
    n_splits    : int   = 5
    purge_pct   : float = 0.01
    embargo_pct : float = 0.005

    def split(
        self, X: pd.DataFrame
    ) -> Generator[tuple[np.ndarray, np.ndarray], None, None]:
        """
        Yield (train_indices, test_indices) for each fold.

        Parameters
        ----------
        X : pd.DataFrame  Feature matrix with a DatetimeIndex

        Yields
        ------
        (train_idx, test_idx) as integer arrays
        """
        n = len(X)
        indices = np.arange(n)
        fold_size = n // self.n_splits

        purge_size  = max(1, int(n * self.purge_pct))
        embargo_size = max(0, int(n * self.embargo_pct))

        for k in range(self.n_splits):
            test_start = k * fold_size
            test_end   = (k + 1) * fold_size if k < self.n_splits - 1 else n

            test_idx  = indices[test_start:test_end]

            # Training set: all observations not in test or embargo windows
            train_end    = max(0, test_start - purge_size)
            train_start2 = min(n, test_end + embargo_size)

            train_idx = np.concatenate([
                indices[:train_end],
                indices[train_start2:],
            ])

            if len(train_idx) == 0:
                continue

            yield train_idx, test_idx

    def n_splits_actual(self, X: pd.DataFrame) -> int:
        """Return the actual number of folds (may differ from n_splits for tiny datasets)."""
        return sum(1 for _ in self.split(X))


# ── WalkForwardAnalysis ──────────────────────────────────────

@dataclass
class WalkForwardResult:
    """Results of a walk-forward analysis."""
    oos_pnl        : pd.Series                   # Out-of-sample P&L concatenated
    fold_metrics   : list[dict]                  # Per-fold metrics
    fold_dates     : list[tuple[str, str]]       # (train_end, test_end) per fold

    @property
    def oos_sharpe(self) -> float:
        r = self.oos_pnl.values
        if r.std() < 1e-14:
            return 0.0
        return r.mean() / r.std() * np.sqrt(252)

    @property
    def oos_max_drawdown(self) -> float:
        cum  = np.cumprod(1 + self.oos_pnl.values)
        peak = np.maximum.accumulate(cum)
        return float(-(((cum - peak) / peak)).min())

    def summary(self) -> pd.DataFrame:
        return pd.DataFrame(self.fold_metrics,
                            index=[f"Fold {i+1}" for i in range(len(self.fold_metrics))])


class WalkForwardAnalysis:
    """
    Walk-forward analysis for strategy validation.

    At each step:
      1. Fit the strategy on the training window
      2. Evaluate P&L on the subsequent out-of-sample window
      3. Advance the window

    Parameters
    ----------
    train_size    : int   Number of observations in the training window
    test_size     : int   Number of observations in each test window
    step_size     : int   Step between successive windows (None = test_size)
    expanding     : bool  True = expanding window; False = rolling window
    """

    def __init__(
        self,
        train_size : int,
        test_size  : int,
        step_size  : Optional[int] = None,
        expanding  : bool = False,
    ):
        self.train_size = train_size
        self.test_size  = test_size
        self.step_size  = step_size or test_size
        self.expanding  = expanding

    def splits(
        self,
        index: pd.DatetimeIndex,
    ) -> Generator[tuple[pd.DatetimeIndex, pd.DatetimeIndex], None, None]:
        """
        Yield (train_index, test_index) date ranges.
        """
        n = len(index)
        start = 0 if self.expanding else 0

        t = self.train_size
        while t + self.test_size <= n:
            train_start = 0 if self.expanding else (t - self.train_size)
            train_idx   = index[train_start:t]
            test_idx    = index[t: t + self.test_size]
            yield train_idx, test_idx
            t += self.step_size

    def run(
        self,
        prices_y: pd.Series,
        prices_x: pd.Series,
        signal_factory,           # callable(y_train, x_train) → fitted signal object
        cost: float = 2e-4,
    ) -> WalkForwardResult:
        """
        Run walk-forward analysis on a pairs trading signal.

        Parameters
        ----------
        prices_y        : log prices of asset A
        prices_x        : log prices of asset B
        signal_factory  : callable that returns a fitted KalmanPairsSignal
                          given (y_train, x_train)
        cost            : one-way transaction cost

        Returns
        -------
        WalkForwardResult
        """
        from backtest import Backtest

        oos_pnl_parts: list[pd.Series] = []
        fold_metrics:  list[dict]      = []
        fold_dates:    list[tuple]     = []

        common = prices_y.index.intersection(prices_x.index)
        prices_y = prices_y.loc[common]
        prices_x = prices_x.loc[common]

        for train_idx, test_idx in self.splits(common):
            if len(train_idx) < 2 or len(test_idx) < 2:
                continue

            # Fit on training data
            sig = signal_factory(
                prices_y.loc[train_idx],
                prices_x.loc[train_idx],
            )

            # Evaluate on test data (re-fit Kalman with trained params,
            # continuing from the end-of-training state)
            full_idx  = train_idx.append(test_idx)
            sig_full  = type(sig)(delta=sig.delta, Ve=sig.Ve,
                                  zscore_window=sig.zscore_window,
                                  enter_z=sig.enter_z, exit_z=sig.exit_z)
            sig_full.fit(prices_y.loc[full_idx], prices_x.loc[full_idx])

            oos_signal = sig_full.signal_.loc[test_idx]
            oos_betas  = sig_full.betas_.loc[test_idx]
            ret_y      = prices_y.loc[test_idx].pct_change().fillna(0)
            ret_x      = prices_x.loc[test_idx].pct_change().fillna(0)
            oos_pnl    = oos_signal.shift(1).fillna(0) * (ret_y - oos_betas.shift(1).fillna(0) * ret_x)

            bt = Backtest(oos_pnl, signal_series=oos_signal,
                          transaction_cost=cost, name=f"OOS")
            m  = bt.run()
            oos_pnl_parts.append(oos_pnl)

            fold_metrics.append({
                "Sharpe":        round(m.sharpe_ratio,         3),
                "Annual Ret (%)":round(m.annual_return * 100,  2),
                "Max DD (%)":    round(m.max_drawdown  * 100,  2),
                "Hit Rate (%)":  round(m.hit_rate      * 100,  2),
            })
            fold_dates.append((
                str(train_idx[0].date()),
                str(test_idx[-1].date()),
            ))

        oos_pnl_cat = pd.concat(oos_pnl_parts).sort_index() if oos_pnl_parts else pd.Series(dtype=float)
        return WalkForwardResult(oos_pnl_cat, fold_metrics, fold_dates)
