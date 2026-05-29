"""
QuantForge — Module 3: Alpha Strategy Demo
==========================================
End-to-end demonstration of the alpha strategy pipeline:
  1. Generate synthetic cointegrated pairs and equity universe
  2. Kalman filter pairs trading signal
  3. Cross-sectional momentum signal
  4. Backtest with transaction costs — performance metrics
  5. Purged k-fold cross-validation
  6. Walk-forward analysis
  7. Strategy comparison
  8. Visualizations

Usage
-----
    cd 03_alpha_strategy/src
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

from signals import (
    KalmanPairsSignal, CrossSectionalMomentum, ZScoreSignal,
    synthetic_pair, synthetic_universe,
)
from backtest import Backtest, compare_strategies
from validation import PurgedKFold, WalkForwardAnalysis
from visualizer import (
    plot_performance_dashboard, plot_signal_dashboard,
    plot_comparison_dashboard, plot_walkforward_dashboard,
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
    print("  QuantForge - Module 3: Alpha Strategy")
    print("=" * 58)

    # ── 1. Data ───────────────────────────────────────────────
    sep("DATA")

    print("  Generating synthetic cointegrated pair (1000 days)...")
    prices_y, prices_x = synthetic_pair(n_days=1000, seed=42)
    returns_y = prices_y.pct_change().dropna()
    returns_x = prices_x.pct_change().dropna()
    print(f"  {prices_y.name}: {len(prices_y)} obs  "
          f"|  σ_ann={returns_y.std()*np.sqrt(252)*100:.1f}%")
    print(f"  {prices_x.name}: {len(prices_x)} obs  "
          f"|  σ_ann={returns_x.std()*np.sqrt(252)*100:.1f}%")

    print("\n  Generating synthetic equity universe (10 assets, 1000 days)...")
    universe_rets = synthetic_universe(n_assets=10, n_days=1000, seed=7)
    print(f"  {universe_rets.shape[1]} assets  |  {len(universe_rets)} obs")

    # ── 2. Kalman Pairs Signal ────────────────────────────────
    sep("KALMAN PAIRS SIGNAL")

    kf_signal = KalmanPairsSignal(delta=1e-5, Ve=1e-3,
                                   zscore_window=30, enter_z=2.0, exit_z=0.5)
    kf_signal.fit(prices_y, prices_x)

    n_long  = (kf_signal.signal_ ==  1).sum()
    n_short = (kf_signal.signal_ == -1).sum()
    n_flat  = (kf_signal.signal_ ==  0).sum()
    total   = len(kf_signal.signal_)
    print(f"\n  Hedge ratio β: mean={kf_signal.betas_.mean():.4f}  "
          f"std={kf_signal.betas_.std():.4f}")
    print(f"  Positions: long={n_long} ({n_long/total*100:.1f}%)  "
          f"short={n_short} ({n_short/total*100:.1f}%)  "
          f"flat={n_flat} ({n_flat/total*100:.1f}%)")

    # ── 3. Cross-Sectional Momentum ───────────────────────────
    sep("CROSS-SECTIONAL MOMENTUM")

    mom = CrossSectionalMomentum(lookback=21, quantile=0.30)
    mom.fit(universe_rets)
    mom_pnl = mom.pnl_series(universe_rets)
    print(f"\n  {len(universe_rets.columns)} assets  |  lookback=21d  |  top/bottom 30%")
    print(f"  Annual signal turnover: "
          f"{(mom.signal_.diff().abs().sum(axis=1)).mean()*252:.1f}  positions/year")

    # ── 4. Backtest ───────────────────────────────────────────
    sep("BACKTEST")

    # Kalman pairs strategy
    kf_pnl  = kf_signal.pnl_series(prices_y, prices_x)
    bt_kf   = Backtest(kf_pnl, kf_signal.signal_,
                       transaction_cost=2e-4, name="Kalman Pairs")
    m_kf    = bt_kf.run()
    print(m_kf)

    # Z-score baseline (same data, simpler signal)
    spread = (np.log(prices_y) - kf_signal.betas_.mean() * np.log(prices_x))
    zscore_sig = ZScoreSignal(window=30, enter_z=2.0, exit_z=0.5)
    zscore_sig.fit(prices_y)
    zscore_pnl = zscore_sig.pnl_series(returns_y)
    bt_zs      = Backtest(zscore_pnl, zscore_sig.signal_,
                          transaction_cost=2e-4, name="Z-Score MR")
    m_zs = bt_zs.run()

    # Momentum strategy
    bt_mom = Backtest(mom_pnl, name="CS Momentum", transaction_cost=2e-4)
    m_mom  = bt_mom.run()
    print(m_mom)

    # ── 5. Purged K-Fold Cross-Validation ─────────────────────
    sep("PURGED K-FOLD CV")

    pkf    = PurgedKFold(n_splits=5, purge_pct=0.02, embargo_pct=0.01)
    dummy_X = pd.DataFrame(index=kf_pnl.index)
    print(f"\n  n_splits={pkf.n_splits}  purge={pkf.purge_pct*100:.1f}%  "
          f"embargo={pkf.embargo_pct*100:.1f}%")
    print(f"\n  {'Fold':<6} {'Train obs':>10} {'Test obs':>10}")
    print(f"  {'-'*30}")
    for i, (tr, te) in enumerate(pkf.split(dummy_X)):
        print(f"  {i+1:<6} {len(tr):>10} {len(te):>10}")

    # Per-fold Sharpe on Kalman signal
    fold_sharpes = []
    for tr, te in pkf.split(dummy_X):
        fold_pnl    = kf_pnl.iloc[te]
        fold_signal = kf_signal.signal_.iloc[te]
        bt_fold     = Backtest(fold_pnl, fold_signal,
                               transaction_cost=2e-4, name="fold")
        fm = bt_fold.run()
        fold_sharpes.append(fm.sharpe_ratio)

    print(f"\n  OOS Sharpe per fold: {[round(float(s), 2) for s in fold_sharpes]}")
    print(f"  Mean OOS Sharpe:      {np.mean(fold_sharpes):.3f}")
    print(f"  Std  OOS Sharpe:      {np.std(fold_sharpes):.3f}")

    # ── 6. Walk-Forward Analysis ──────────────────────────────
    sep("WALK-FORWARD ANALYSIS")

    wfa = WalkForwardAnalysis(train_size=252, test_size=63, expanding=False)

    def signal_factory(y_train, x_train):
        sig = KalmanPairsSignal(delta=1e-5, Ve=1e-3,
                                zscore_window=30, enter_z=2.0, exit_z=0.5)
        sig.fit(y_train, x_train)
        return sig

    wf_result = wfa.run(prices_y, prices_x, signal_factory, cost=2e-4)
    print(f"\n{wf_result.summary().to_string()}")
    print(f"\n  OOS Sharpe:        {wf_result.oos_sharpe:.3f}")
    print(f"  OOS Max Drawdown:  {wf_result.oos_max_drawdown*100:.2f}%")

    # ── 7. Strategy Comparison ────────────────────────────────
    sep("STRATEGY COMPARISON")

    comp = compare_strategies([m_kf, m_mom, m_zs])
    print(f"\n{comp.to_string()}")

    # ── 8. Visualizations ─────────────────────────────────────
    sep("VISUALIZATIONS")

    print("  Performance dashboard (Kalman Pairs)...")
    plot_performance_dashboard(m_kf)

    print("  Signal dashboard...")
    plot_signal_dashboard(kf_signal, prices_y, prices_x)

    print("  Strategy comparison dashboard...")
    plot_comparison_dashboard([m_kf, m_mom, m_zs])

    print("  Walk-forward dashboard...")
    plot_walkforward_dashboard(wf_result)

    # ── Done ──────────────────────────────────────────────────
    sep()
    elapsed = time.perf_counter() - t0
    print(f"\n  Module 3 complete -- {elapsed:.2f}s")
    print(f"  Charts saved to: {Path(__file__).parent.parent / 'notebooks'}/\n")


if __name__ == "__main__":
    run()
