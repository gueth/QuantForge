"""
QuantForge - Module 2: Visualization Dashboards
=================================================
Publication-ready charts for risk analytics: VaR, PCA, Fama-French,
rolling VaR, and C++ benchmark.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from pathlib import Path

from portfolio import Portfolio, PCAResult, FFResult, RiskReport

OUT = Path(__file__).parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Color palette (dark theme)
# ---------------------------------------------------------------------------
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


def _apply_style() -> None:
    plt.rcParams.update({
        "figure.facecolor":  C["bg"],
        "axes.facecolor":    C["surface"],
        "axes.edgecolor":    C["muted"],
        "axes.labelcolor":   C["text"],
        "axes.grid":         True,
        "grid.color":        C["grid"],
        "grid.linewidth":    0.5,
        "text.color":        C["text"],
        "xtick.color":       C["muted"],
        "ytick.color":       C["muted"],
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "legend.framealpha": 0,
        "legend.labelcolor": C["text"],
        "font.family":       "monospace",
        "figure.dpi":        120,
    })


_apply_style()


def _fig(w: int = 16, h: int = 10) -> plt.Figure:
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(C["bg"])
    return fig


# ---------------------------------------------------------------------------
# 1. VaR / CVaR Dashboard
# ---------------------------------------------------------------------------

def plot_var_dashboard(
    portfolio: Portfolio,
    report:    RiskReport,
    save:      bool = True,
    fname:     str  = "var_dashboard.png",
) -> plt.Figure:
    """
    4-panel VaR/CVaR dashboard:
      A - Return distribution with VaR/CVaR markers
      B - Q-Q plot (normality test)
      C - Cumulative P&L
      D - VaR sensitivity to confidence level
    """
    r    = portfolio.portfolio_returns.values * 100  # convert to %
    conf = report.confidence

    fig = _fig()
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # -- A: Return distribution ----------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    n_bins = min(80, max(20, len(r) // 12))
    ax.hist(r, bins=n_bins, density=True, color=C["accent1"],
            alpha=0.65, edgecolor="none")

    mu, sig = r.mean(), r.std()
    xs = np.linspace(r.min(), r.max(), 400)
    ax.plot(xs, stats.norm.pdf(xs, mu, sig),
            color=C["accent3"], lw=1.8, label="Gaussian fit")

    var_h  = -report.var_historical  * 100  # negative threshold on x-axis
    cvar_h = -report.cvar_historical * 100
    ax.axvline(var_h,  color=C["accent2"], lw=1.8, ls="--",
               label=f"VaR hist {conf*100:.0f}%: {-var_h:.2f}%")
    ax.axvline(cvar_h, color=C["accent4"], lw=1.8, ls="-.",
               label=f"CVaR hist: {-cvar_h:.2f}%")
    ax.fill_between(xs[xs <= var_h],
                    stats.norm.pdf(xs[xs <= var_h], mu, sig),
                    alpha=0.22, color=C["accent2"])

    sk, ku = stats.skew(r), stats.kurtosis(r)
    ax.text(0.97, 0.97, f"skew={sk:.2f}\nkurt={ku:.2f}",
            transform=ax.transAxes, va="top", ha="right",
            fontsize=8, color=C["accent4"])
    ax.set_title("Daily Return Distribution", color=C["text"])
    ax.set_xlabel("Daily Return (%)");  ax.set_ylabel("Density")
    ax.legend(fontsize=7.5)

    # -- B: Q-Q plot ---------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    ax2.scatter(osm, osr, color=C["accent1"], s=4, alpha=0.45)
    lx = np.array([osm[0], osm[-1]])
    ax2.plot(lx, slope * lx + intercept, color=C["accent3"], lw=1.8)
    _, jb_p = stats.jarque_bera(r)
    color = C["accent2"] if jb_p < 0.05 else C["accent3"]
    ax2.text(0.04, 0.95, f"Jarque-Bera p = {jb_p:.4f}",
             transform=ax2.transAxes, fontsize=8.5, color=color, va="top")
    ax2.set_title("Q-Q Plot (normality check)", color=C["text"])
    ax2.set_xlabel("Theoretical Quantiles")
    ax2.set_ylabel("Empirical Quantiles")

    # -- C: Cumulative P&L ---------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    cum_pct = ((1 + portfolio.portfolio_returns).cumprod() - 1) * 100
    ax3.plot(cum_pct.index, cum_pct.values, color=C["accent1"], lw=1.4)
    ax3.fill_between(cum_pct.index, cum_pct.values,
                     where=cum_pct.values >= 0, alpha=0.12, color=C["accent3"])
    ax3.fill_between(cum_pct.index, cum_pct.values,
                     where=cum_pct.values <  0, alpha=0.20, color=C["accent2"])
    ax3.axhline(0, color=C["muted"], lw=0.8, ls="--")
    ax3.set_title("Cumulative Portfolio P&L", color=C["text"])
    ax3.set_xlabel("Date");  ax3.set_ylabel("Cumulative Return (%)")

    # -- D: VaR sensitivity --------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    confs = np.linspace(0.90, 0.999, 70)
    vh = [portfolio.var_historical(c) * 100 for c in confs]
    vp = [portfolio.var_parametric(c)  * 100 for c in confs]
    ch = [portfolio.cvar_historical(c) * 100 for c in confs]

    ax4.plot(confs * 100, vh, color=C["accent1"], lw=1.8, label="VaR hist.")
    ax4.plot(confs * 100, vp, color=C["accent3"], lw=1.8, ls="--",
             label="VaR param.")
    ax4.plot(confs * 100, ch, color=C["accent2"], lw=1.8, ls="-.",
             label="CVaR hist.")
    ax4.axvline(conf * 100, color=C["muted"], lw=1, ls=":", alpha=0.7)
    ax4.set_title("VaR Sensitivity to Confidence Level", color=C["text"])
    ax4.set_xlabel("Confidence Level (%)");  ax4.set_ylabel("VaR (%)")
    ax4.legend(fontsize=8)

    date_range = (
        f"{portfolio.returns.index[0].strftime('%Y-%m-%d')} to "
        f"{portfolio.returns.index[-1].strftime('%Y-%m-%d')}"
    )
    fig.suptitle(
        f"QuantForge - Risk Engine | {portfolio.name} | {date_range}",
        color=C["text"], fontsize=12, y=0.99,
    )

    if save:
        path = OUT / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {path}")
    return fig


# ---------------------------------------------------------------------------
# 2. PCA Dashboard
# ---------------------------------------------------------------------------

def plot_pca_dashboard(
    pca:       PCAResult,
    portfolio: Portfolio,
    n_show:    int  = 5,
    save:      bool = True,
    fname:     str  = "pca_dashboard.png",
) -> plt.Figure:
    """
    4-panel PCA dashboard:
      A - Scree plot (explained + cumulative variance)
      B - Factor loading heatmap
      C - PC1 vs PC2 biplot
      D - PC1 / PC2 factor return time series
    """
    k     = min(n_show, len(pca.eigenvalues))
    names = [f"PC{i + 1}" for i in range(k)]
    fig   = _fig()
    gs    = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # -- A: Scree plot -------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    xs  = np.arange(1, k + 1)
    ax1.bar(xs, pca.explained_variance_ratio[:k] * 100,
            color=C["accent1"], alpha=0.75, width=0.55)
    ax1b = ax1.twinx()
    ax1b.plot(xs, pca.cumulative_variance[:k] * 100,
              color=C["accent3"], marker="o", ms=5, lw=1.8, label="Cumulative")
    ax1b.axhline(90, color=C["accent4"], lw=1, ls="--", alpha=0.7)
    ax1b.set_ylabel("Cumulative (%)", color=C["accent3"])
    ax1b.tick_params(axis="y", labelcolor=C["accent3"])
    ax1.set_xticks(xs);  ax1.set_xticklabels(names, fontsize=9)
    ax1.set_title("Scree Plot", color=C["text"])
    ax1.set_xlabel("Component");  ax1.set_ylabel("Explained Variance (%)")

    # -- B: Loading heatmap --------------------------------------------------
    ax2   = fig.add_subplot(gs[0, 1])
    load  = pca.factor_loadings[names].T.values   # shape (k, n_assets)
    im    = ax2.imshow(load, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(len(pca.asset_names)))
    ax2.set_xticklabels(pca.asset_names, rotation=40, ha="right", fontsize=8)
    ax2.set_yticks(range(k));  ax2.set_yticklabels(names, fontsize=9)
    ax2.set_title("Factor Loadings", color=C["text"])
    plt.colorbar(im, ax=ax2, fraction=0.03, pad=0.04)
    for i in range(k):
        for j in range(len(pca.asset_names)):
            v = load[i, j]
            ax2.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6.5,
                     color="white" if abs(v) > 0.5 else C["text"])

    # -- C: Biplot PC1 vs PC2 ------------------------------------------------
    ax3 = fig.add_subplot(gs[1, 0])
    if pca.factor_returns.shape[1] >= 2:
        pc1   = pca.factor_returns["PC1"].values
        pc2   = pca.factor_returns["PC2"].values
        scale = pc1.std() * 3
        ax3.scatter(pc1, pc2, alpha=0.25, s=4, color=C["accent1"])
        for i, asset in enumerate(pca.asset_names):
            vx = pca.factor_loadings.iloc[i, 0] * scale
            vy = pca.factor_loadings.iloc[i, 1] * scale
            ax3.annotate("", xy=(vx, vy), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->",
                                         color=C["accent2"], lw=1.5))
            ax3.text(vx * 1.1, vy * 1.1, asset, color=C["accent2"],
                     fontsize=8, ha="center")
    ax3.axhline(0, color=C["muted"], lw=0.5, ls="--")
    ax3.axvline(0, color=C["muted"], lw=0.5, ls="--")
    ax3.set_title("Biplot PC1 vs PC2", color=C["text"])
    ax3.set_xlabel("PC1");  ax3.set_ylabel("PC2")

    # -- D: Factor return series --------------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(pca.factor_returns.index, pca.factor_returns["PC1"],
             color=C["accent1"], lw=1.2, label="PC1")
    if "PC2" in pca.factor_returns.columns:
        ax4.plot(pca.factor_returns.index, pca.factor_returns["PC2"],
                 color=C["accent3"], lw=1.2, label="PC2", alpha=0.85)
    ax4.axhline(0, color=C["muted"], lw=0.6, ls="--")
    ax4.set_title("Factor Return Series (PC1, PC2)", color=C["text"])
    ax4.set_xlabel("Date");  ax4.set_ylabel("Factor Return")
    ax4.legend(fontsize=9)

    fig.suptitle(f"QuantForge - PCA | {portfolio.name}",
                 color=C["text"], fontsize=12, y=0.99)
    if save:
        path = OUT / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {path}")
    return fig


# ---------------------------------------------------------------------------
# 3. Fama-French Dashboard
# ---------------------------------------------------------------------------

def plot_ff_dashboard(
    ff:    FFResult,
    save:  bool = True,
    fname: str  = "ff_dashboard.png",
) -> plt.Figure:
    """
    4-panel Fama-French dashboard:
      A - Factor beta heatmap
      B - Jensen's alpha (annualised)
      C - R-squared per asset
      D - Systematic vs idiosyncratic volatility decomposition
    """
    fig    = _fig()
    gs     = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    assets = ff.asset_names
    n      = len(assets)
    k      = len(ff.factor_names)

    betas_mat = np.array([[r["betas"][j] for j in range(k)] for r in ff.raw])
    alphas    = np.array([r["alpha"] * 252 * 100 for r in ff.raw])
    r2s       = np.array([r["r_squared"]           for r in ff.raw])
    sys_vols  = np.array([r["systematic_vol"] * 100 for r in ff.raw])
    idio_vols = np.array([r["idio_vol"]        * 100 for r in ff.raw])

    # -- A: Beta heatmap -----------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    im  = ax1.imshow(betas_mat, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax1.set_xticks(range(k));  ax1.set_xticklabels(ff.factor_names, fontsize=10)
    ax1.set_yticks(range(n));  ax1.set_yticklabels(assets, fontsize=9)
    ax1.set_title("Fama-French Factor Betas", color=C["text"])
    plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.04)
    for i in range(n):
        for j in range(k):
            v = betas_mat[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if abs(v) > 1 else C["text"])

    # -- B: Jensen's alpha ---------------------------------------------------
    ax2    = fig.add_subplot(gs[0, 1])
    colors = [C["accent3"] if a >= 0 else C["accent2"] for a in alphas]
    ax2.barh(assets, alphas, color=colors, alpha=0.85, height=0.55)
    ax2.axvline(0, color=C["muted"], lw=1, ls="--")
    ax2.set_title("Jensen's Alpha (ann. %)", color=C["text"])
    ax2.set_xlabel("Alpha (%)")
    if len(alphas) > 0:
        scale = max(abs(alphas)) if max(abs(alphas)) > 0 else 1
        for i, v in enumerate(alphas):
            ax2.text(v + 0.02 * np.sign(v) * scale,
                     i, f"{v:.2f}%", va="center", fontsize=8, color=C["text"])

    # -- C: R-squared --------------------------------------------------------
    ax3    = fig.add_subplot(gs[1, 0])
    colors = [C["accent1"] if r2 >= 0.5 else C["accent4"] for r2 in r2s]
    ax3.bar(assets, r2s, color=colors, alpha=0.85, width=0.55)
    ax3.axhline(0.5, color=C["muted"], lw=1, ls="--", alpha=0.6)
    ax3.set_ylim(0, 1.05)
    ax3.set_title("R-Squared (Explanatory Power)", color=C["text"])
    ax3.set_ylabel("R2")
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(assets, rotation=30, ha="right", fontsize=9)

    # -- D: Volatility decomposition -----------------------------------------
    ax4 = fig.add_subplot(gs[1, 1])
    xp  = np.arange(n);  w = 0.36
    ax4.bar(xp - w / 2, sys_vols,  w, color=C["accent1"], alpha=0.85,
            label="Systematic")
    ax4.bar(xp + w / 2, idio_vols, w, color=C["accent2"], alpha=0.85,
            label="Idiosyncratic")
    ax4.set_title("Volatility Decomposition (ann. %)", color=C["text"])
    ax4.set_ylabel("Vol (%)")
    ax4.set_xticks(xp)
    ax4.set_xticklabels(assets, rotation=30, ha="right", fontsize=9)
    ax4.legend(fontsize=9)

    fig.suptitle("QuantForge - Fama-French 3-Factor Model",
                 color=C["text"], fontsize=12, y=0.99)
    if save:
        path = OUT / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {path}")
    return fig


# ---------------------------------------------------------------------------
# 4. Rolling VaR Dashboard
# ---------------------------------------------------------------------------

def plot_rolling_var(
    portfolio:  Portfolio,
    window:     int   = 252,
    confidence: float = 0.95,
    save:       bool  = True,
    fname:      str   = "rolling_var.png",
) -> plt.Figure:
    """
    3-panel rolling VaR dashboard:
      A - Daily returns with rolling VaR overlay (full-width)
      B - Historical vs parametric rolling VaR
      C - Rolling VaR distribution
    """
    rv_hist  = portfolio.rolling_var(confidence=confidence, window=window,
                                     parametric=False)
    rv_param = portfolio.rolling_var(confidence=confidence, window=window,
                                     parametric=True)
    pr       = portfolio.portfolio_returns

    fig = _fig(w=16, h=10)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # -- A: Returns + rolling VaR overlay (spans full top row) ---------------
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(pr.index, pr.values * 100,
                     where=pr.values >= 0, alpha=0.35, color=C["accent3"])
    ax1.fill_between(pr.index, pr.values * 100,
                     where=pr.values <  0, alpha=0.45, color=C["accent2"])
    ax1.plot(pr.index, pr.values * 100, lw=0.6, color=C["muted"], alpha=0.7)
    ax1.plot(rv_hist.index, -rv_hist.values * 100,
             lw=1.6, color=C["accent2"], ls="--",
             label=f"-VaR hist {confidence*100:.0f}%")
    ax1.plot(rv_param.index, -rv_param.values * 100,
             lw=1.6, color=C["accent4"], ls="-.",
             label=f"-VaR param {confidence*100:.0f}%")
    ax1.axhline(0, color=C["muted"], lw=0.8, ls="--")
    ax1.set_title(f"Daily Returns & Rolling VaR (window={window}d)",
                  color=C["text"])
    ax1.set_xlabel("Date");  ax1.set_ylabel("Return (%)")
    ax1.legend(fontsize=9)

    # -- B: Hist vs param rolling VaR ----------------------------------------
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rv_hist.index,  rv_hist.values  * 100,
             color=C["accent2"], lw=1.4, label="Historical")
    ax2.plot(rv_param.index, rv_param.values * 100,
             color=C["accent4"], lw=1.4, ls="--", label="Parametric")
    ax2.set_title(f"Rolling VaR {confidence*100:.0f}% ({window}d window)",
                  color=C["text"])
    ax2.set_xlabel("Date");  ax2.set_ylabel("VaR (%)")
    ax2.legend(fontsize=9)

    # -- C: VaR distribution -------------------------------------------------
    ax3    = fig.add_subplot(gs[1, 1])
    v_hist = rv_hist.values * 100
    n_bins = min(50, max(15, len(v_hist) // 10))
    ax3.hist(v_hist, bins=n_bins, density=True, color=C["accent2"],
             alpha=0.65, edgecolor="none", label="Hist. VaR")
    mean_v  = float(np.mean(v_hist))
    pct95_v = float(np.percentile(v_hist, 95))
    ax3.axvline(mean_v,  color=C["accent3"], lw=1.8, ls="--",
                label=f"Mean {mean_v:.2f}%")
    ax3.axvline(pct95_v, color=C["accent4"], lw=1.5, ls="-.",
                label=f"95th pct. {pct95_v:.2f}%")
    ax3.set_title("Rolling VaR Distribution", color=C["text"])
    ax3.set_xlabel("VaR (%)");  ax3.set_ylabel("Density")
    ax3.legend(fontsize=8)

    fig.suptitle(
        f"QuantForge - Rolling VaR | {portfolio.name} | window={window}d",
        color=C["text"], fontsize=12, y=0.99,
    )

    if save:
        path = OUT / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {path}")
    return fig


# ---------------------------------------------------------------------------
# 5. C++ Benchmark Chart
# ---------------------------------------------------------------------------

def plot_benchmark(
    results: dict,
    save:    bool = True,
    fname:   str  = "benchmark.png",
) -> plt.Figure:
    """Grouped bar chart comparing Python vs C++ latencies."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["surface"])

    labels    = list(results.keys())
    py_times  = [results[k]["python_ms"] for k in labels]
    cpp_times = [results[k]["cpp_ms"]    for k in labels]
    speedups  = [results[k]["speedup"]   for k in labels]

    x, w  = np.arange(len(labels)), 0.35
    bars1 = ax.bar(x - w/2, py_times,  w, color=C["accent4"],
                   alpha=0.85, label="Python (NumPy)")
    bars2 = ax.bar(x + w/2, cpp_times, w, color=C["accent1"],
                   alpha=0.85, label="C++ (-O2)")

    for i, (b1, b2, sp) in enumerate(zip(bars1, bars2, speedups)):
        ax.text(b1.get_x() + b1.get_width() / 2, b1.get_height() + 0.3,
                f"{py_times[i]:.1f}ms", ha="center", va="bottom",
                fontsize=8, color=C["text"])
        ax.text(b2.get_x() + b2.get_width() / 2, b2.get_height() + 0.3,
                f"{cpp_times[i]:.1f}ms", ha="center", va="bottom",
                fontsize=8, color=C["text"])
        ax.text(i, max(py_times[i], cpp_times[i]) + 2,
                f"x{sp:.0f}", ha="center", fontsize=9,
                color=C["accent3"], fontweight="bold")

    ax.set_xticks(x);  ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Benchmark: Python (NumPy) vs C++ (-O2)",
                 color=C["text"], fontsize=12)
    ax.grid(axis="y", color=C["grid"], lw=0.5)
    ax.legend(fontsize=9)

    if save:
        path = OUT / fname
        fig.savefig(path, bbox_inches="tight", dpi=150)
        print(f"  Saved -> {path}")
    return fig
