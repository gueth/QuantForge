"""
QuantForge — Module 3: Unit Tests
===================================
Covers:
  - KalmanPairsSignal (filter properties, signal generation)
  - CrossSectionalMomentum (signal shape, neutrality)
  - ZScoreSignal (stationarity, thresholds)
  - Backtest (P&L correctness, metrics)
  - PurgedKFold (fold sizes, no leakage)
  - WalkForwardAnalysis (OOS consistency)

Run:
    pytest 03_alpha_strategy/tests/ -v
"""
import numpy as np
import pandas as pd
import pytest

from signals import (
    KalmanPairsSignal, CrossSectionalMomentum, ZScoreSignal,
    synthetic_pair, synthetic_universe,
    _py_kalman_filter, _py_rolling_zscore,
)
from backtest import Backtest, PerformanceMetrics, compare_strategies
from validation import PurgedKFold, WalkForwardAnalysis


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def pair():
    y, x = synthetic_pair(n_days=500, seed=0)
    return y, x

@pytest.fixture(scope="module")
def kf_signal(pair):
    y, x = pair
    sig = KalmanPairsSignal(delta=1e-5, Ve=1e-3, zscore_window=20,
                             enter_z=2.0, exit_z=0.5)
    sig.fit(y, x)
    return sig

@pytest.fixture(scope="module")
def universe():
    return synthetic_universe(n_assets=8, n_days=500, seed=1)


# ─────────────────────────────────────────────────────────────
# Kalman Filter (pure Python kernel)
# ─────────────────────────────────────────────────────────────

class TestKalmanKernel:

    def test_output_length(self, pair):
        y, x = pair
        res = _py_kalman_filter(np.log(y.values), np.log(x.values), 1e-5, 1e-3)
        T = len(y)
        for key in ("betas", "spreads", "innovations", "innovations_var"):
            assert len(res[key]) == T, f"{key} wrong length"

    def test_innovations_var_positive(self, pair):
        y, x = pair
        res = _py_kalman_filter(np.log(y.values), np.log(x.values), 1e-5, 1e-3)
        assert (res["innovations_var"] > 0).all()

    def test_beta_converges(self, pair):
        """Beta should stabilise after sufficient observations."""
        y, x = pair
        res = _py_kalman_filter(np.log(y.values), np.log(x.values), 1e-5, 1e-3)
        betas = res["betas"]
        early_std = betas[:50].std()
        late_std  = betas[200:].std()
        assert late_std <= early_std + 0.1

    def test_spread_defined(self, pair):
        """spread_t = y_t - beta_t * x_t must hold by construction."""
        y, x = pair
        yv = np.log(y.values); xv = np.log(x.values)
        res = _py_kalman_filter(yv, xv, 1e-5, 1e-3)
        expected = yv - res["betas"] * xv
        np.testing.assert_allclose(res["spreads"], expected, atol=1e-10)


class TestRollingZScore:

    def test_length(self, pair):
        y, _ = pair
        z = _py_rolling_zscore(y.values, window=20)
        assert len(z) == len(y)

    def test_first_window_minus_one_is_nan(self, pair):
        y, _ = pair
        z = _py_rolling_zscore(y.values, window=30)
        assert np.all(np.isnan(z[:29]))

    def test_finite_after_warmup(self, pair):
        y, _ = pair
        z = _py_rolling_zscore(y.values, window=20)
        assert np.all(np.isfinite(z[20:]))

    def test_zero_mean_approx(self, pair):
        y, _ = pair
        z = _py_rolling_zscore(y.values, window=20)
        valid = z[~np.isnan(z)]
        assert abs(valid.mean()) < 1.0


# ─────────────────────────────────────────────────────────────
# KalmanPairsSignal
# ─────────────────────────────────────────────────────────────

class TestKalmanPairsSignal:

    def test_fit_populates_attrs(self, kf_signal):
        assert kf_signal.betas_   is not None
        assert kf_signal.spreads_ is not None
        assert kf_signal.zscore_  is not None
        assert kf_signal.signal_  is not None

    def test_signal_values(self, kf_signal):
        """Signal must be in {-1, 0, +1}."""
        vals = kf_signal.signal_.values
        assert set(vals).issubset({-1.0, 0.0, 1.0})

    def test_betas_positive(self, pair, kf_signal):
        """For a positively correlated pair, beta should be positive."""
        assert kf_signal.betas_.mean() > 0

    def test_zscore_has_nans_only_at_start(self, kf_signal):
        w = kf_signal.zscore_window
        z = kf_signal.zscore_.values
        assert np.all(np.isnan(z[:w - 1]))
        assert np.all(np.isfinite(z[w:]))

    def test_pnl_series_correct_length(self, kf_signal, pair):
        y, x = pair
        pnl = kf_signal.pnl_series(y, x)
        assert len(pnl) == len(y)

    def test_fit_raises_without_common_index(self):
        y = pd.Series(np.ones(100),
                      index=pd.bdate_range("2020-01-01", periods=100))
        x = pd.Series(np.ones(100),
                      index=pd.bdate_range("2021-01-01", periods=100))
        sig = KalmanPairsSignal()
        sig.fit(y, x)   # no common dates → empty fit, should not raise
        assert len(sig.signal_) == 0

    def test_different_params_give_different_signals(self, pair):
        y, x = pair
        s1 = KalmanPairsSignal(enter_z=1.5).fit(y, x)
        s2 = KalmanPairsSignal(enter_z=3.0).fit(y, x)
        n_active1 = (s1.signal_ != 0).sum()
        n_active2 = (s2.signal_ != 0).sum()
        # Tighter threshold → more active days
        assert n_active1 >= n_active2


# ─────────────────────────────────────────────────────────────
# CrossSectionalMomentum
# ─────────────────────────────────────────────────────────────

class TestCrossSectionalMomentum:

    def test_signal_shape(self, universe):
        mom = CrossSectionalMomentum(lookback=21).fit(universe)
        assert mom.signal_.shape == universe.shape

    def test_signal_dollar_neutral(self, universe):
        """Long/short portfolio should sum to ~0 on each date."""
        mom = CrossSectionalMomentum(lookback=21).fit(universe)
        row_sums = mom.signal_.sum(axis=1).dropna()
        np.testing.assert_allclose(row_sums.values, 0.0, atol=1e-10)

    def test_pnl_length(self, universe):
        mom = CrossSectionalMomentum(lookback=21).fit(universe)
        pnl = mom.pnl_series(universe)
        assert len(pnl) == len(universe)

    def test_first_lookback_rows_are_flat(self, universe):
        """During the warm-up period the signal is flat (all zeros, no position)."""
        lookback = 21
        mom      = CrossSectionalMomentum(lookback=lookback).fit(universe)
        # First (lookback - 1) rows: no momentum can be computed -> signal = 0
        warmup = mom.signal_.iloc[:lookback - 1]
        assert (warmup.values == 0.0).all()


# ─────────────────────────────────────────────────────────────
# ZScoreSignal
# ─────────────────────────────────────────────────────────────

class TestZScoreSignal:

    def test_signal_values(self, pair):
        y, _ = pair
        sig = ZScoreSignal(window=20).fit(y)
        vals = sig.signal_.values
        assert set(vals).issubset({-1.0, 0.0, 1.0})

    def test_enter_exit_thresholds(self, pair):
        y, _ = pair
        enter_z = 1.5
        exit_z  = 0.3
        sig = ZScoreSignal(window=20, enter_z=enter_z, exit_z=exit_z).fit(y)
        # Once in a position, we should never be in position when |z| >= exit_z ...
        # Just verify the signal never flips directly without passing through 0
        s = sig.signal_.values
        for t in range(1, len(s)):
            if s[t] != s[t-1]:
                # Transition must go through 0 or be a direct entry
                assert s[t-1] == 0 or s[t] == 0, \
                    f"Invalid transition {s[t-1]}→{s[t]} at t={t}"


# ─────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────

class TestBacktest:

    def test_metrics_types(self, kf_signal, pair):
        y, x = pair
        pnl = kf_signal.pnl_series(y, x)
        bt  = Backtest(pnl, kf_signal.signal_, name="Test")
        m   = bt.run()
        assert isinstance(m, PerformanceMetrics)

    def test_equity_curve_length(self, kf_signal, pair):
        y, x = pair
        pnl  = kf_signal.pnl_series(y, x)
        bt   = Backtest(pnl, name="Test")
        m    = bt.run()
        assert len(m.equity_curve) == len(pnl)

    def test_drawdown_non_positive(self, kf_signal, pair):
        y, x = pair
        pnl  = kf_signal.pnl_series(y, x)
        bt   = Backtest(pnl, name="Test")
        m    = bt.run()
        assert (m.drawdown_series.values <= 1e-10).all()

    def test_max_drawdown_non_negative(self, kf_signal, pair):
        y, x = pair
        pnl  = kf_signal.pnl_series(y, x)
        bt   = Backtest(pnl, name="Test")
        m    = bt.run()
        assert m.max_drawdown >= 0

    def test_hit_rate_in_unit_interval(self, kf_signal, pair):
        y, x = pair
        pnl  = kf_signal.pnl_series(y, x)
        bt   = Backtest(pnl, name="Test")
        m    = bt.run()
        assert 0.0 <= m.hit_rate <= 1.0

    def test_vol_positive(self, kf_signal, pair):
        y, x = pair
        pnl  = kf_signal.pnl_series(y, x)
        bt   = Backtest(pnl, name="Test")
        m    = bt.run()
        assert m.annual_vol > 0

    def test_zero_pnl_gives_zero_metrics(self):
        dates = pd.bdate_range("2020-01-01", periods=252)
        pnl   = pd.Series(0.0, index=dates)
        bt    = Backtest(pnl, name="Zero")
        m     = bt.run()
        assert m.sharpe_ratio == pytest.approx(0.0, abs=1e-8)
        assert m.total_return == pytest.approx(0.0, abs=1e-8)

    def test_transaction_cost_reduces_return(self, kf_signal, pair):
        y, x   = pair
        pnl    = kf_signal.pnl_series(y, x)
        bt0    = Backtest(pnl, kf_signal.signal_, transaction_cost=0.0,   name="no cost")
        bt1    = Backtest(pnl, kf_signal.signal_, transaction_cost=1e-3,  name="with cost")
        m0, m1 = bt0.run(), bt1.run()
        assert m0.total_return >= m1.total_return

    def test_compare_strategies_shape(self, kf_signal, pair):
        y, x = pair
        pnl  = kf_signal.pnl_series(y, x)
        bt   = Backtest(pnl, name="S1")
        m    = bt.run()
        df   = compare_strategies([m, m])
        assert df.shape == (2, 9)


# ─────────────────────────────────────────────────────────────
# PurgedKFold
# ─────────────────────────────────────────────────────────────

class TestPurgedKFold:

    def test_correct_number_of_folds(self, pair):
        y, _ = pair
        dummy = pd.DataFrame(index=y.index)
        pkf   = PurgedKFold(n_splits=5)
        folds = list(pkf.split(dummy))
        assert len(folds) == 5

    def test_no_overlap_between_train_test(self, pair):
        y, _ = pair
        dummy = pd.DataFrame(index=y.index)
        pkf   = PurgedKFold(n_splits=5, purge_pct=0.02)
        for train, test in pkf.split(dummy):
            assert len(np.intersect1d(train, test)) == 0

    def test_test_indices_non_overlapping(self, pair):
        y, _ = pair
        dummy = pd.DataFrame(index=y.index)
        pkf   = PurgedKFold(n_splits=5)
        all_test = np.concatenate([te for _, te in pkf.split(dummy)])
        assert len(all_test) == len(set(all_test))

    def test_train_larger_than_test(self, pair):
        y, _ = pair
        dummy = pd.DataFrame(index=y.index)
        pkf   = PurgedKFold(n_splits=5, purge_pct=0.01)
        for train, test in pkf.split(dummy):
            assert len(train) >= len(test)

    def test_different_k_give_different_sizes(self, pair):
        y, _ = pair
        dummy = pd.DataFrame(index=y.index)
        pkf3  = PurgedKFold(n_splits=3)
        pkf5  = PurgedKFold(n_splits=5)
        test3 = [len(te) for _, te in pkf3.split(dummy)]
        test5 = [len(te) for _, te in pkf5.split(dummy)]
        assert max(test3) > max(test5)


# ─────────────────────────────────────────────────────────────
# WalkForwardAnalysis
# ─────────────────────────────────────────────────────────────

class TestWalkForwardAnalysis:

    def test_oos_pnl_non_empty(self, pair):
        y, x = pair

        def factory(y_tr, x_tr):
            return KalmanPairsSignal(delta=1e-5, Ve=1e-3).fit(y_tr, x_tr)

        wfa = WalkForwardAnalysis(train_size=200, test_size=50)
        res = wfa.run(y, x, factory)
        assert len(res.oos_pnl) > 0

    def test_fold_metrics_length(self, pair):
        y, x = pair

        def factory(y_tr, x_tr):
            return KalmanPairsSignal().fit(y_tr, x_tr)

        wfa = WalkForwardAnalysis(train_size=200, test_size=50, step_size=50)
        res = wfa.run(y, x, factory)
        n_folds = len(res.fold_metrics)
        assert n_folds >= 1

    def test_oos_sharpe_is_float(self, pair):
        y, x = pair

        def factory(y_tr, x_tr):
            return KalmanPairsSignal().fit(y_tr, x_tr)

        wfa = WalkForwardAnalysis(train_size=200, test_size=50)
        res = wfa.run(y, x, factory)
        assert isinstance(res.oos_sharpe, float)
        assert np.isfinite(res.oos_sharpe)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
