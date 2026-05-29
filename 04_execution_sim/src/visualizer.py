"""
QuantForge — Module 4: Visualization Dashboards
=================================================
Publication-ready execution analysis charts.
"""
from __future__ import annotations

from typing import Optional
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.figure import Figure
from pathlib import Path

from schedules import ExecutionSchedule
from simulator import ExecutionReport

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

PALETTE = [C["accent1"], C["accent3"], C["accent4"], C["accent2"], "#a371f7"]


# ── 1. Schedule Comparison Dashboard ─────────────────────────

def plot_schedule_dashboard(
    schedules: list[ExecutionSchedule],
    save: bool = True,
    fname: str = "schedule_dashboard.png",
) -> Figure:
    """
    4 panels: trade sizes, cumulative trades, participation rate, holdings profile.
    """
    fig = _fig()
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── A : Trade sizes ───────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for s, col in zip(schedules, PALETTE):
        xs = np.arange(1, s.N + 1)
        ax1.bar(xs + schedules.index(s) * 0.2, s.trades,
                width=0.2, color=col, alpha=0.85, label=s.name)
    ax1.set_title("Child Order Sizes (shares)", color=C["text"])
    ax1.set_xlabel("Interval"); ax1.set_ylabel("Shares")
    ax1.legend(fontsize=8)

    # ── B : Cumulative execution ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    for s, col in zip(schedules, PALETTE):
        cumtrades = np.concatenate([[0], np.cumsum(s.trades) / s.total_shares * 100])
        ax2.plot(np.arange(s.N + 1), cumtrades, color=col, lw=1.8, label=s.name)
    ax2.set_title("Cumulative Execution (%)", color=C["text"])
    ax2.set_xlabel("Interval"); ax2.set_ylabel("% of Order Completed")
    ax2.legend(fontsize=8)

    # ── C : Participation rate ─────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    for s, col in zip(schedules, PALETTE):
        ax3.plot(np.arange(1, s.N + 1), s.participation_rate() * 100,
                 color=col, lw=1.6, marker="o", ms=4, label=s.name)
    ax3.set_title("Participation Rate per Interval (%)", color=C["text"])
    ax3.set_xlabel("Interval"); ax3.set_ylabel("% of Parent Order")
    ax3.legend(fontsize=8)

    # ── D : Holdings profile ──────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    for s, col in zip(schedules, PALETTE):
        h = s.holdings / s.total_shares * 100
        ax4.plot(np.arange(s.N + 1), h, color=col, lw=1.8, label=s.name)
    ax4.axhline(0, color=C["muted"], lw=0.6, ls="--")
    ax4.set_title("Remaining Holdings (%)", color=C["text"])
    ax4.set_xlabel("Interval"); ax4.set_ylabel("% Remaining")
    ax4.legend(fontsize=8)

    names = " | ".join(s.name for s in schedules)
    fig.suptitle(f"QuantForge — Execution Schedules  |  {names}",
                 color=C["text"], fontsize=12, y=0.99)

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig


# ── 2. Implementation Shortfall Distribution ──────────────────

def plot_is_dashboard(
    reports: list[ExecutionReport],
    save: bool = True,
    fname: str = "is_dashboard.png",
) -> Figure:
    """
    4 panels: IS distributions, VaR comparison, cost decomposition, IS vs λ.
    """
    fig = _fig()
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── A : IS distributions ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    for r, col in zip(reports, PALETTE):
        arr = r.is_samples / r.arrival_price * 1e4  # bps
        n_bins = min(50, max(15, len(arr) // 30))
        ax1.hist(arr, bins=n_bins, density=True, color=col, alpha=0.55,
                 edgecolor="none", label=f"{r.schedule_name} ({r.mean_is_bps:.1f} bps)")
    ax1.axvline(0, color=C["muted"], lw=0.8, ls="--")
    ax1.set_title("IS Distribution (bps)", color=C["text"])
    ax1.set_xlabel("Implementation Shortfall (bps)")
    ax1.set_ylabel("Density"); ax1.legend(fontsize=7.5)

    # ── B : Mean & VaR bar chart ──────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    names    = [r.schedule_name for r in reports]
    means    = [r.mean_is_bps for r in reports]
    var95s   = [r.var_95 / r.arrival_price * 1e4 for r in reports]
    x_pos    = np.arange(len(names)); w = 0.35
    ax2.bar(x_pos - w/2, means,  w, color=C["accent1"], alpha=0.85, label="Mean IS")
    ax2.bar(x_pos + w/2, var95s, w, color=C["accent2"], alpha=0.85, label="95% VaR")
    ax2.axhline(0, color=C["muted"], lw=0.6, ls="--")
    ax2.set_title("Mean IS & 95% VaR (bps)", color=C["text"])
    ax2.set_xticks(x_pos); ax2.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax2.legend(fontsize=8)

    # ── C : Cost decomposition ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    scale = 1e4  # to bps
    perms  = [r.mean_perm_impact / r.arrival_price * scale for r in reports]
    temps  = [r.mean_temp_impact / r.arrival_price * scale for r in reports]
    x_pos2 = np.arange(len(names))
    ax3.bar(x_pos2, perms, color=C["accent4"], alpha=0.85, label="Permanent")
    ax3.bar(x_pos2, temps, bottom=perms, color=C["accent3"], alpha=0.85, label="Temporary")
    ax3.set_title("IS Decomposition: Perm + Temp (bps)", color=C["text"])
    ax3.set_xticks(x_pos2)
    ax3.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
    ax3.legend(fontsize=8)

    # ── D : Risk-return trade-off (if multiple AC reports) ─────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.scatter([r.mean_is_bps for r in reports],
                [r.std_is_bps  for r in reports],
                c=PALETTE[:len(reports)], s=80, zorder=5)
    for r, col in zip(reports, PALETTE):
        ax4.annotate(r.schedule_name,
                     (r.mean_is_bps, r.std_is_bps),
                     textcoords="offset points", xytext=(6, 4),
                     fontsize=8, color=col)
    ax4.set_title("Risk-Return of Execution (E[IS] vs σ[IS])", color=C["text"])
    ax4.set_xlabel("Mean IS (bps)"); ax4.set_ylabel("Std IS (bps)")

    fig.suptitle("QuantForge — Implementation Shortfall Analysis",
                 color=C["text"], fontsize=12, y=0.99)

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig


# ── 3. Efficient Frontier Dashboard ──────────────────────────

def plot_efficient_frontier(
    frontier_data: list[dict],   # [{"lambda": λ, "mean_is": ..., "std_is": ...}]
    twap_report: Optional[ExecutionReport] = None,
    vwap_report: Optional[ExecutionReport] = None,
    save: bool = True,
    fname: str = "efficient_frontier.png",
) -> Figure:
    """
    Efficient frontier: E[IS] vs Std[IS] as λ varies.
    Benchmarks TWAP and VWAP shown as reference points.
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["surface"])

    means  = [d["mean_is"] for d in frontier_data]
    stds   = [d["std_is"]  for d in frontier_data]
    lambdas = [d["lambda"] for d in frontier_data]

    sc = ax.scatter(stds, means, c=np.log10(lambdas), cmap="plasma",
                    s=40, zorder=5)
    ax.plot(stds, means, color=C["accent1"], lw=1.4, alpha=0.7, zorder=4)
    plt.colorbar(sc, ax=ax, label="log₁₀(λ)")

    if twap_report:
        ax.scatter(twap_report.std_is_bps, twap_report.mean_is_bps,
                   color=C["accent4"], s=150, marker="^", zorder=6, label="TWAP")
    if vwap_report:
        ax.scatter(vwap_report.std_is_bps, vwap_report.mean_is_bps,
                   color=C["accent3"], s=150, marker="s", zorder=6, label="VWAP")

    ax.set_xlabel("Std of IS (bps)"); ax.set_ylabel("Mean IS (bps)")
    ax.set_title("Almgren-Chriss Efficient Frontier\n"
                 "(E[IS] vs σ[IS] as risk-aversion λ varies)",
                 color=C["text"])
    ax.legend(fontsize=9)
    ax.grid(True, color=C["grid"], lw=0.5)

    fig.suptitle("QuantForge — Execution Efficient Frontier",
                 color=C["text"], fontsize=12, y=0.99)

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig
