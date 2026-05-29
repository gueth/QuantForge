"""
QuantForge — Module 3: Visualization Dashboards
=================================================
Publication-ready dashboards for alpha strategy analysis.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy import stats

from backtest import PerformanceMetrics
from signals import KalmanPairsSignal

OUT = Path(__file__).parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

C = {
    "bg":      "#0d1117",
    "surface": "#161b22",
    "accent1": "#58a6ff",
    "accent2": "#f85149",
    "accent3": "#3fb950",
    "accent4": "#e3b341",
    "text":    "#c9d1d9",
    "muted":   "#484f58",
    "grid":    "#21262d",
}

def _style():
    plt.rcParams.update({
        "figure.facecolor": C["bg"],     "axes.facecolor":   C["surface"],
        "axes.edgecolor":   C["muted"],  "axes.labelcolor":  C["text"],
        "axes.grid":        True,         "grid.color":       C["grid"],
        "grid.linewidth":   0.5,          "text.color":       C["text"],
        "xtick.color":      C["muted"],   "ytick.color":      C["muted"],
        "xtick.labelsize":  8,            "ytick.labelsize":  8,
        "legend.framealpha":0,            "legend.labelcolor":C["text"],
        "font.family":      "monospace",  "figure.dpi":       120,
    })
_style()

def _fig(w=16, h=10):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(C["bg"])
    return fig


# ── 1. Strategy Performance Dashboard ────────────────────────

def plot_performance_dashboard(
    metrics: PerformanceMetrics,
    save: bool = True,
    fname: str = "performance_dashboard.png",
) -> plt.Figure:
    """
    4 panels: equity curve, drawdown, daily P&L distribution, rolling Sharpe.
    """
    fig = _fig()
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    eq  = metrics.equity_curve * 100
    dd  = metrics.drawdown_series * 100
    pnl = metrics.daily_pnl * 100

    # ── A : Equity Curve ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(eq.index, eq.values, color=C["accent1"], lw=1.5)
    ax1.fill_between(eq.index, eq.values,
                     where=eq.values >= 0, alpha=0.12, color=C["accent3"])
    ax1.fill_between(eq.index, eq.values,
                     where=eq.values <  0, alpha=0.20, color=C["accent2"])
    ax1.axhline(0, color=C["muted"], lw=0.8, ls="--")
    tot = metrics.total_return * 100
    ax1.set_title(f"Equity Curve  |  Total: {tot:+.2f}%", color=C["text"])
    ax1.set_xlabel("Date"); ax1.set_ylabel("Cumulative Return (%)")

    # ── B : Drawdown ──────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.fill_between(dd.index, dd.values, alpha=0.65, color=C["accent2"])
    ax2.axhline(0, color=C["muted"], lw=0.6)
    ax2.set_title(f"Drawdown  |  Max: {metrics.max_drawdown*100:.2f}%",
                  color=C["text"])
    ax2.set_xlabel("Date"); ax2.set_ylabel("Drawdown (%)")

    # ── C : Daily P&L Distribution ────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    n_bins = min(60, max(20, len(pnl) // 12))
    ax3.hist(pnl, bins=n_bins, density=True, color=C["accent1"],
             alpha=0.65, edgecolor="none")
    mu, sig_d = pnl.mean(), pnl.std()
    xs = np.linspace(pnl.min(), pnl.max(), 300)
    ax3.plot(xs, stats.norm.pdf(xs, mu, sig_d),
             color=C["accent3"], lw=1.8, label="Gaussian fit")
    ax3.axvline(0, color=C["muted"], lw=0.8, ls="--")
    sk, ku = stats.skew(pnl.dropna()), stats.kurtosis(pnl.dropna())
    ax3.text(0.97, 0.97, f"skew={sk:.2f}\nkurt={ku:.2f}",
             transform=ax3.transAxes, va="top", ha="right",
             fontsize=8, color=C["accent4"])
    ax3.set_title("Daily P&L Distribution", color=C["text"])
    ax3.set_xlabel("Daily P&L (%)"); ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)

    # ── D : Rolling Sharpe ────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    window = min(63, len(pnl) // 4)
    roll   = pnl.rolling(window)
    r_sharpe = roll.mean() / roll.std(ddof=1) * np.sqrt(252)
    ax4.plot(r_sharpe.index, r_sharpe.values, color=C["accent4"], lw=1.4)
    ax4.axhline(0,  color=C["muted"], lw=0.6, ls="--")
    ax4.axhline(metrics.sharpe_ratio, color=C["accent3"], lw=1.2, ls=":",
                label=f"Full-period: {metrics.sharpe_ratio:.2f}")
    ax4.set_title(f"Rolling Sharpe ({window}d window)", color=C["text"])
    ax4.set_xlabel("Date"); ax4.set_ylabel("Sharpe Ratio")
    ax4.legend(fontsize=8)

    fig.suptitle(f"QuantForge — {metrics.strategy_name}  |  "
                 f"Sharpe: {metrics.sharpe_ratio:.2f}  |  "
                 f"Calmar: {metrics.calmar_ratio:.2f}",
                 color=C["text"], fontsize=12, y=0.99)

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig


# ── 2. Kalman Pairs Signal Dashboard ─────────────────────────

def plot_signal_dashboard(
    signal: KalmanPairsSignal,
    prices_y: pd.Series,
    prices_x: pd.Series,
    save: bool = True,
    fname: str = "signal_dashboard.png",
) -> plt.Figure:
    """
    4 panels: price series, hedge ratio, spread + z-score, signal positions.
    """
    common = prices_y.index.intersection(prices_x.index)
    y = prices_y.loc[common]
    x = prices_x.loc[common]
    betas  = signal.betas_
    zscore = signal.zscore_
    sig    = signal.signal_

    fig = _fig()
    gs  = gridspec.GridSpec(4, 1, figure=fig, hspace=0.45)

    # ── A : Price series ──────────────────────────────────────
    ax1 = fig.add_subplot(gs[0])
    ax1b = ax1.twinx()
    ax1.plot(y.index, y.values, color=C["accent1"], lw=1.2, label=y.name or "Asset A")
    ax1b.plot(x.index, x.values, color=C["accent4"], lw=1.2, label=x.name or "Asset B")
    ax1.set_ylabel("Asset A", color=C["accent1"])
    ax1b.set_ylabel("Asset B", color=C["accent4"])
    ax1.set_title("Price Series", color=C["text"])
    ax1.legend(loc="upper left", fontsize=8)
    ax1b.legend(loc="upper right", fontsize=8)

    # ── B : Hedge ratio β ─────────────────────────────────────
    ax2 = fig.add_subplot(gs[1])
    if betas is not None:
        ax2.plot(betas.index, betas.values, color=C["accent3"], lw=1.2)
    ax2.axhline(0, color=C["muted"], lw=0.6, ls="--")
    ax2.set_title("Dynamic Hedge Ratio β (Kalman)", color=C["text"])
    ax2.set_ylabel("β")

    # ── C : Z-score ───────────────────────────────────────────
    ax3 = fig.add_subplot(gs[2])
    if zscore is not None:
        ax3.plot(zscore.index, zscore.values, color=C["accent1"], lw=1.0)
        ax3.axhline( signal.enter_z, color=C["accent2"], lw=1.2, ls="--",
                     label=f"+{signal.enter_z}σ")
        ax3.axhline(-signal.enter_z, color=C["accent2"], lw=1.2, ls="--",
                     label=f"-{signal.enter_z}σ")
        ax3.axhline(0, color=C["muted"], lw=0.6, ls=":")
    ax3.set_title("Spread Z-Score", color=C["text"])
    ax3.set_ylabel("Z-Score"); ax3.legend(fontsize=8)

    # ── D : Signal positions ──────────────────────────────────
    ax4 = fig.add_subplot(gs[3])
    if sig is not None:
        ax4.fill_between(sig.index, sig.values,
                         where=sig.values > 0, color=C["accent3"], alpha=0.65,
                         step="post", label="Long spread")
        ax4.fill_between(sig.index, sig.values,
                         where=sig.values < 0, color=C["accent2"], alpha=0.65,
                         step="post", label="Short spread")
        ax4.axhline(0, color=C["muted"], lw=0.8)
    ax4.set_title("Signal (Position)", color=C["text"])
    ax4.set_ylabel("Position"); ax4.legend(fontsize=8)
    ax4.set_yticks([-1, 0, 1]); ax4.set_ylim(-1.5, 1.5)

    fig.suptitle(f"QuantForge — Kalman Pairs Signal  |  "
                 f"δ={signal.delta}  Ve={signal.Ve}  "
                 f"z_enter={signal.enter_z}",
                 color=C["text"], fontsize=12, y=1.01)

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig


# ── 3. Strategy Comparison Dashboard ─────────────────────────

def plot_comparison_dashboard(
    strategies: list[PerformanceMetrics],
    save: bool = True,
    fname: str = "comparison_dashboard.png",
) -> plt.Figure:
    """Bar charts comparing multiple strategies across key metrics."""
    from backtest import compare_strategies
    table = compare_strategies(strategies)

    metrics_to_plot = [
        ("Sharpe",           "Sharpe Ratio"),
        ("Annual Return (%)", "Annual Return (%)"),
        ("Max DD (%)",       "Max Drawdown (%)"),
        ("Hit Rate (%)",     "Hit Rate (%)"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.patch.set_facecolor(C["bg"])
    palette = [C["accent1"], C["accent3"], C["accent4"], C["accent2"],
               "#a371f7", "#79c0ff"]
    colors = palette[:len(strategies)]

    for ax, (col, title) in zip(axes.flat, metrics_to_plot):
        ax.set_facecolor(C["surface"])
        vals = table[col].values
        bars = ax.bar(table.index, vals, color=colors, alpha=0.85, width=0.5)
        ax.axhline(0, color=C["muted"], lw=0.8, ls="--")
        ax.set_title(title, color=C["text"])
        ax.set_ylabel(col)
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", color=C["grid"], lw=0.5)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.02 * max(abs(vals)),
                    f"{v:.2f}", ha="center", fontsize=8, color=C["text"])

    fig.suptitle("QuantForge — Strategy Comparison",
                 color=C["text"], fontsize=12, y=0.99)
    plt.tight_layout()

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig


# ── 4. Walk-Forward Dashboard ─────────────────────────────────

def plot_walkforward_dashboard(
    wf_result,
    save: bool = True,
    fname: str = "walkforward_dashboard.png",
) -> plt.Figure:
    """
    3 panels: OOS equity curve, per-fold Sharpe, cumulative OOS returns.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.patch.set_facecolor(C["bg"])

    oos   = wf_result.oos_pnl
    eq    = (np.cumprod(1 + oos.values) - 1) * 100
    dates = oos.index

    # A : OOS equity curve
    ax1 = axes[0]; ax1.set_facecolor(C["surface"])
    ax1.plot(dates, eq, color=C["accent1"], lw=1.5)
    ax1.axhline(0, color=C["muted"], lw=0.8, ls="--")
    ax1.fill_between(dates, eq, where=np.array(eq) >= 0,
                     alpha=0.12, color=C["accent3"])
    ax1.fill_between(dates, eq, where=np.array(eq) < 0,
                     alpha=0.20, color=C["accent2"])
    ax1.set_title(f"OOS Equity  (Sharpe={wf_result.oos_sharpe:.2f})",
                  color=C["text"])
    ax1.set_xlabel("Date"); ax1.set_ylabel("Cumulative Return (%)")

    # B : Per-fold Sharpe
    ax2 = axes[1]; ax2.set_facecolor(C["surface"])
    fold_sharpes = [m.get("Sharpe", 0) for m in wf_result.fold_metrics]
    x_pos = np.arange(len(fold_sharpes))
    colors = [C["accent3"] if s >= 0 else C["accent2"] for s in fold_sharpes]
    ax2.bar(x_pos, fold_sharpes, color=colors, alpha=0.85)
    ax2.axhline(0, color=C["muted"], lw=0.8, ls="--")
    ax2.set_title("Per-Fold OOS Sharpe Ratio", color=C["text"])
    ax2.set_xlabel("Fold"); ax2.set_ylabel("Sharpe")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels([f"F{i+1}" for i in x_pos], fontsize=9)

    # C : Rolling OOS Sharpe
    ax3 = axes[2]; ax3.set_facecolor(C["surface"])
    window = min(63, len(oos) // 4)
    if window > 1:
        roll = oos.rolling(window)
        rs   = roll.mean() / roll.std(ddof=1) * np.sqrt(252)
        ax3.plot(rs.index, rs.values, color=C["accent4"], lw=1.4)
        ax3.axhline(0, color=C["muted"], lw=0.6, ls="--")
    ax3.set_title(f"Rolling OOS Sharpe ({window}d)", color=C["text"])
    ax3.set_xlabel("Date"); ax3.set_ylabel("Sharpe")

    fig.suptitle("QuantForge — Walk-Forward Analysis",
                 color=C["text"], fontsize=12, y=1.00)
    plt.tight_layout()

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig
