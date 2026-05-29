"""
QuantForge — Module 2 : Visualisations
========================================
Dashboards publication-ready : VaR, PCA, Fama-French.
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

# ── Palette & style ──────────────────────────────────────────
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
        "figure.facecolor": C["bg"],
        "axes.facecolor":   C["surface"],
        "axes.edgecolor":   C["muted"],
        "axes.labelcolor":  C["text"],
        "axes.grid":        True,
        "grid.color":       C["grid"],
        "grid.linewidth":   0.5,
        "text.color":       C["text"],
        "xtick.color":      C["muted"],
        "ytick.color":      C["muted"],
        "xtick.labelsize":  8,
        "ytick.labelsize":  8,
        "legend.framealpha":0,
        "legend.labelcolor":C["text"],
        "font.family":      "monospace",
        "figure.dpi":       120,
    })
_style()


def _fig(w=16, h=10):
    fig = plt.figure(figsize=(w, h))
    fig.patch.set_facecolor(C["bg"])
    return fig


# ── 1. Dashboard VaR / CVaR ──────────────────────────────────

def plot_var_dashboard(
    portfolio: Portfolio,
    report: RiskReport,
    save: bool = True,
    fname: str = "var_dashboard.png",
) -> plt.Figure:
    """
    4 panneaux :
    A — Distribution des rendements + marqueurs VaR/CVaR
    B — Q-Q plot (test de normalité)
    C — P&L cumulé
    D — Sensibilité VaR / niveau de confiance
    """
    r = portfolio.portfolio_returns.values * 100   # en %
    conf = report.confidence

    fig = _fig()
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── A : Distribution ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    n_bins = min(80, max(20, len(r) // 12))
    ax.hist(r, bins=n_bins, density=True, color=C["accent1"],
            alpha=0.65, edgecolor="none")

    mu, sig = r.mean(), r.std()
    xs = np.linspace(r.min(), r.max(), 400)
    ax.plot(xs, stats.norm.pdf(xs, mu, sig),
            color=C["accent3"], lw=1.8, label="Gaussien ajusté")

    var_h  = -report.var_historical  * 100
    cvar_h = -report.cvar_historical * 100
    ax.axvline(var_h,  color=C["accent2"], lw=1.8, ls="--",
               label=f"VaR hist {conf*100:.0f}%: {-var_h:.2f}%")
    ax.axvline(cvar_h, color=C["accent4"], lw=1.8, ls="-.",
               label=f"CVaR hist: {-cvar_h:.2f}%")
    ax.fill_between(xs[xs <= var_h],
                    stats.norm.pdf(xs[xs <= var_h], mu, sig),
                    alpha=0.22, color=C["accent2"])

    sk = stats.skew(r)
    ku = stats.kurtosis(r)
    ax.text(0.97, 0.97, f"skew={sk:.2f}\nkurt={ku:.2f}",
            transform=ax.transAxes, va="top", ha="right", fontsize=8,
            color=C["accent4"])
    ax.set_title("Distribution des rendements", color=C["text"])
    ax.set_xlabel("Rendement journalier (%)"); ax.set_ylabel("Densité")
    ax.legend(fontsize=7.5)

    # ── B : Q-Q plot ─────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    (osm, osr), (slope, intercept, _) = stats.probplot(r, dist="norm")
    ax2.scatter(osm, osr, color=C["accent1"], s=4, alpha=0.45)
    lx = np.array([osm[0], osm[-1]])
    ax2.plot(lx, slope * lx + intercept, color=C["accent3"], lw=1.8)
    jb_stat, jb_p = stats.jarque_bera(r)
    col = C["accent2"] if jb_p < 0.05 else C["accent3"]
    ax2.text(0.04, 0.95, f"Jarque-Bera p = {jb_p:.4f}",
             transform=ax2.transAxes, fontsize=8.5, color=col, va="top")
    ax2.set_title("Q-Q Plot (normalité)", color=C["text"])
    ax2.set_xlabel("Quantiles théoriques"); ax2.set_ylabel("Quantiles empiriques")

    # ── C : P&L cumulé ───────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    cum = (1 + portfolio.portfolio_returns).cumprod() - 1
    cum_pct = cum * 100
    ax3.plot(cum_pct.index, cum_pct.values, color=C["accent1"], lw=1.4)
    ax3.fill_between(cum_pct.index, cum_pct.values,
                     where=cum_pct.values >= 0, alpha=0.12, color=C["accent3"])
    ax3.fill_between(cum_pct.index, cum_pct.values,
                     where=cum_pct.values < 0,  alpha=0.20, color=C["accent2"])
    ax3.axhline(0, color=C["muted"], lw=0.8, ls="--")
    ax3.set_title("P&L cumulé du portefeuille", color=C["text"])
    ax3.set_xlabel("Date"); ax3.set_ylabel("Rendement cumulé (%)")

    # ── D : Sensibilité VaR ───────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    confs = np.linspace(0.90, 0.999, 70)
    vh, vp, ch = [], [], []
    for c in confs:
        vh.append(portfolio.var_historical(c) * 100)
        vp.append(portfolio.var_parametric(c) * 100)
        ch.append(portfolio.cvar_historical(c) * 100)

    ax4.plot(confs * 100, vh, color=C["accent1"], lw=1.8, label="VaR hist.")
    ax4.plot(confs * 100, vp, color=C["accent3"], lw=1.8, ls="--", label="VaR param.")
    ax4.plot(confs * 100, ch, color=C["accent2"], lw=1.8, ls="-.", label="CVaR hist.")
    ax4.axvline(conf * 100, color=C["muted"], lw=1, ls=":", alpha=0.7)
    ax4.set_title("Sensibilité VaR / niveau de confiance", color=C["text"])
    ax4.set_xlabel("Niveau de confiance (%)"); ax4.set_ylabel("VaR (%)")
    ax4.legend(fontsize=8)

    date_range = (f"{portfolio.returns.index[0].strftime('%Y-%m-%d')} → "
                  f"{portfolio.returns.index[-1].strftime('%Y-%m-%d')}")
    fig.suptitle(f"QuantForge — Risk Engine | {portfolio.name} | {date_range}",
                 color=C["text"], fontsize=12, y=0.99)

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Sauvegardé → {p}")
    return fig


# ── 2. Dashboard PCA ─────────────────────────────────────────

def plot_pca_dashboard(
    pca: PCAResult,
    portfolio: Portfolio,
    n_show: int = 5,
    save: bool = True,
    fname: str = "pca_dashboard.png",
) -> plt.Figure:
    """
    4 panneaux :
    A — Scree plot (variance expliquée + cumulée)
    B — Heatmap des loadings
    C — Biplot PC1 vs PC2
    D — Séries temporelles PC1 / PC2
    """
    k = min(n_show, len(pca.eigenvalues))
    names = [f"PC{i+1}" for i in range(k)]
    fig = _fig()
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.35)

    # ── A : Scree plot ────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    xs = np.arange(1, k + 1)
    ax1.bar(xs, pca.explained_variance_ratio[:k] * 100,
            color=C["accent1"], alpha=0.75, width=0.55)
    ax2 = ax1.twinx()
    ax2.plot(xs, pca.cumulative_variance[:k] * 100,
             color=C["accent3"], marker="o", ms=5, lw=1.8, label="Cumulé")
    ax2.axhline(90, color=C["accent4"], lw=1, ls="--", alpha=0.7)
    ax2.set_ylabel("Cumulé (%)", color=C["accent3"])
    ax2.tick_params(axis="y", labelcolor=C["accent3"])
    ax1.set_xticks(xs); ax1.set_xticklabels(names, fontsize=9)
    ax1.set_title("Scree Plot", color=C["text"])
    ax1.set_xlabel("Composante"); ax1.set_ylabel("Variance expliquée (%)")

    # ── B : Heatmap loadings ──────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 1])
    load = pca.factor_loadings[names].T.values    # k × n_assets
    im = ax3.imshow(load, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax3.set_xticks(range(len(pca.asset_names)))
    ax3.set_xticklabels(pca.asset_names, rotation=40, ha="right", fontsize=8)
    ax3.set_yticks(range(k)); ax3.set_yticklabels(names, fontsize=9)
    ax3.set_title("Loadings factoriels", color=C["text"])
    plt.colorbar(im, ax=ax3, fraction=0.03, pad=0.04)
    for i in range(k):
        for j in range(len(pca.asset_names)):
            v = load[i, j]
            ax3.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=6.5,
                     color="white" if abs(v) > 0.5 else C["text"])

    # ── C : Biplot PC1 vs PC2 ─────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    if pca.factor_returns.shape[1] >= 2:
        pc1 = pca.factor_returns["PC1"].values
        pc2 = pca.factor_returns["PC2"].values
        ax4.scatter(pc1, pc2, alpha=0.25, s=4, color=C["accent1"])
        scale = pc1.std() * 3
        for i, asset in enumerate(pca.asset_names):
            vx = pca.factor_loadings.iloc[i, 0] * scale
            vy = pca.factor_loadings.iloc[i, 1] * scale
            ax4.annotate("", xy=(vx, vy), xytext=(0, 0),
                         arrowprops=dict(arrowstyle="->", color=C["accent2"], lw=1.5))
            ax4.text(vx * 1.1, vy * 1.1, asset, color=C["accent2"],
                     fontsize=8, ha="center")
    ax4.axhline(0, color=C["muted"], lw=0.5, ls="--")
    ax4.axvline(0, color=C["muted"], lw=0.5, ls="--")
    ax4.set_title("Biplot PC1 vs PC2", color=C["text"])
    ax4.set_xlabel("PC1"); ax4.set_ylabel("PC2")

    # ── D : Séries factorielles ───────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    ax5.plot(pca.factor_returns.index, pca.factor_returns["PC1"],
             color=C["accent1"], lw=1.2, label="PC1")
    if "PC2" in pca.factor_returns.columns:
        ax5.plot(pca.factor_returns.index, pca.factor_returns["PC2"],
                 color=C["accent3"], lw=1.2, label="PC2", alpha=0.85)
    ax5.axhline(0, color=C["muted"], lw=0.6, ls="--")
    ax5.set_title("Rendements factoriels (PC1, PC2)", color=C["text"])
    ax5.set_xlabel("Date"); ax5.set_ylabel("Rendement factoriel")
    ax5.legend(fontsize=9)

    fig.suptitle(f"QuantForge — PCA | {portfolio.name}",
                 color=C["text"], fontsize=12, y=0.99)
    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Sauvegardé → {p}")
    return fig


# ── 3. Dashboard Fama-French ──────────────────────────────────

def plot_ff_dashboard(
    ff: FFResult,
    save: bool = True,
    fname: str = "ff_dashboard.png",
) -> plt.Figure:
    """
    4 panneaux :
    A — Heatmap bêtas
    B — Alpha de Jensen (annualisé)
    C — R² par actif
    D — Décomposition vol systématique / idiosyncratique
    """
    fig = _fig()
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.35)
    assets = ff.asset_names
    n = len(assets)
    fnames = ff.factor_names
    K = len(fnames)

    betas_mat = np.array([[r["betas"][k] for k in range(K)] for r in ff.raw])
    alphas    = np.array([r["alpha"] * 252 * 100 for r in ff.raw])
    r2s       = np.array([r["r_squared"]           for r in ff.raw])
    sys_vols  = np.array([r["systematic_vol"] * 100 for r in ff.raw])
    idio_vols = np.array([r["idio_vol"] * 100        for r in ff.raw])

    # ── A : Bêtas ────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(betas_mat, cmap="RdBu_r", aspect="auto", vmin=-2, vmax=2)
    ax1.set_xticks(range(K)); ax1.set_xticklabels(fnames, fontsize=10)
    ax1.set_yticks(range(n)); ax1.set_yticklabels(assets, fontsize=9)
    ax1.set_title("Bêtas Fama-French (β)", color=C["text"])
    plt.colorbar(im, ax=ax1, fraction=0.04, pad=0.04)
    for i in range(n):
        for j in range(K):
            v = betas_mat[i, j]
            ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                     fontsize=8, color="white" if abs(v) > 1 else C["text"])

    # ── B : Alphas ───────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [C["accent3"] if a >= 0 else C["accent2"] for a in alphas]
    ax2.barh(assets, alphas, color=colors, alpha=0.85, height=0.55)
    ax2.axvline(0, color=C["muted"], lw=1, ls="--")
    ax2.set_title("Alpha de Jensen (ann. %)", color=C["text"])
    ax2.set_xlabel("Alpha (%)")
    for i, v in enumerate(alphas):
        ax2.text(v + 0.02 * np.sign(v) * max(abs(alphas)),
                 i, f"{v:.2f}%", va="center", fontsize=8, color=C["text"])

    # ── C : R² ───────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    bar_c = [C["accent1"] if r >= 0.5 else C["accent4"] for r in r2s]
    ax3.bar(assets, r2s, color=bar_c, alpha=0.85, width=0.55)
    ax3.axhline(0.5, color=C["muted"], lw=1, ls="--", alpha=0.6)
    ax3.set_ylim(0, 1.05); ax3.set_title("R² — Pouvoir explicatif", color=C["text"])
    ax3.set_ylabel("R²")
    ax3.set_xticks(range(n))
    ax3.set_xticklabels(assets, rotation=30, ha="right", fontsize=9)

    # ── D : Vol systématique / idiosyncratique ────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    xp = np.arange(n); w = 0.36
    ax4.bar(xp - w/2, sys_vols,  w, color=C["accent1"], alpha=0.85, label="Systématique")
    ax4.bar(xp + w/2, idio_vols, w, color=C["accent2"], alpha=0.85, label="Idiosyncratique")
    ax4.set_title("Décomposition volatilité (ann. %)", color=C["text"])
    ax4.set_ylabel("Vol (%)")
    ax4.set_xticks(xp); ax4.set_xticklabels(assets, rotation=30, ha="right", fontsize=9)
    ax4.legend(fontsize=9)

    fig.suptitle("QuantForge — Modèle Fama-French 3 facteurs",
                 color=C["text"], fontsize=12, y=0.99)
    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Sauvegardé → {p}")
    return fig


# ── 4. Rolling VaR ───────────────────────────────────────────

def plot_rolling_var(
    portfolio: Portfolio,
    window: int = 252,
    confidence: float = 0.95,
    save: bool = True,
    fname: str = "rolling_var.png",
) -> plt.Figure:
    """
    3 panneaux :
    A — Rendements journaliers du portefeuille + VaR glissante
    B — VaR historique vs paramétrique (fenêtre glissante)
    C — Distribution de la VaR glissante
    """
    rv_hist  = portfolio.rolling_var(confidence=confidence, window=window,
                                     parametric=False)
    rv_param = portfolio.rolling_var(confidence=confidence, window=window,
                                     parametric=True)
    pr       = portfolio.portfolio_returns

    fig = _fig(w=16, h=10)
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── A : Rendements + VaR glissante ────────────────────────
    ax1 = fig.add_subplot(gs[0, :])   # occupe toute la rangée du haut
    ax1.fill_between(pr.index, pr.values * 100,
                     where=pr.values >= 0, alpha=0.35, color=C["accent3"])
    ax1.fill_between(pr.index, pr.values * 100,
                     where=pr.values < 0,  alpha=0.45, color=C["accent2"])
    ax1.plot(pr.index, pr.values * 100, lw=0.6, color=C["muted"], alpha=0.7)
    ax1.plot(rv_hist.index,  -rv_hist.values  * 100,
             lw=1.6, color=C["accent2"], ls="--",
             label=f"−VaR hist. {confidence*100:.0f}%")
    ax1.plot(rv_param.index, -rv_param.values * 100,
             lw=1.6, color=C["accent4"], ls="-.",
             label=f"−VaR param. {confidence*100:.0f}%")
    ax1.axhline(0, color=C["muted"], lw=0.8, ls="--")
    ax1.set_title(f"Rendements journaliers & VaR glissante (fenêtre {window}j)",
                  color=C["text"])
    ax1.set_xlabel("Date"); ax1.set_ylabel("Rendement (%)")
    ax1.legend(fontsize=9)

    # ── B : VaR hist vs param ─────────────────────────────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(rv_hist.index,  rv_hist.values  * 100,
             color=C["accent2"], lw=1.4, label="VaR hist.")
    ax2.plot(rv_param.index, rv_param.values * 100,
             color=C["accent4"], lw=1.4, ls="--", label="VaR param.")
    ax2.set_title(f"VaR glissante {confidence*100:.0f}% ({window}j)", color=C["text"])
    ax2.set_xlabel("Date"); ax2.set_ylabel("VaR (%)")
    ax2.legend(fontsize=9)

    # ── C : Distribution de la VaR glissante ─────────────────
    ax3 = fig.add_subplot(gs[1, 1])
    v_hist = rv_hist.values * 100
    n_bins = min(50, max(15, len(v_hist) // 10))
    ax3.hist(v_hist, bins=n_bins, density=True, color=C["accent2"],
             alpha=0.65, edgecolor="none", label="VaR hist.")
    ax3.axvline(np.mean(v_hist), color=C["accent3"], lw=1.8, ls="--",
                label=f"Moyenne {np.mean(v_hist):.2f}%")
    ax3.axvline(np.percentile(v_hist, 95), color=C["accent4"], lw=1.5, ls="-.",
                label=f"95e pct. {np.percentile(v_hist, 95):.2f}%")
    ax3.set_title("Distribution de la VaR glissante", color=C["text"])
    ax3.set_xlabel("VaR (%)"); ax3.set_ylabel("Densité")
    ax3.legend(fontsize=8)

    fig.suptitle(f"QuantForge — Rolling VaR | {portfolio.name} | fenêtre {window}j",
                 color=C["text"], fontsize=12, y=0.99)

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Sauvegardé → {p}")
    return fig


# ── 5. Benchmark Python vs C++ ────────────────────────────────

def plot_benchmark(
    results: dict,
    save: bool = True,
    fname: str = "benchmark.png",
) -> plt.Figure:
    """Bar chart comparant les temps Python pur vs C++."""
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    fig.patch.set_facecolor(C["bg"])
    ax.set_facecolor(C["surface"])

    labels   = list(results.keys())
    py_times = [results[k]["python_ms"] for k in labels]
    cpp_times= [results[k]["cpp_ms"]    for k in labels]
    speedups = [results[k]["speedup"]   for k in labels]

    x, w = np.arange(len(labels)), 0.35
    bars1 = ax.bar(x - w/2, py_times,  w, color=C["accent4"], alpha=0.85, label="Python pur")
    bars2 = ax.bar(x + w/2, cpp_times, w, color=C["accent1"], alpha=0.85, label="C++ (-O2)")

    for i, (b1, b2, sp) in enumerate(zip(bars1, bars2, speedups)):
        ax.text(b1.get_x() + b1.get_width()/2, b1.get_height() + 0.3,
                f"{py_times[i]:.1f}ms", ha="center", va="bottom", fontsize=8, color=C["text"])
        ax.text(b2.get_x() + b2.get_width()/2, b2.get_height() + 0.3,
                f"{cpp_times[i]:.1f}ms", ha="center", va="bottom", fontsize=8, color=C["text"])
        ax.text(i, max(py_times[i], cpp_times[i]) + 2,
                f"×{sp:.0f}", ha="center", fontsize=9, color=C["accent3"], fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Temps (ms)")
    ax.set_title("Benchmark Python pur vs C++ (-O2)", color=C["text"], fontsize=12)
    ax.grid(axis="y", color=C["grid"], lw=0.5)
    ax.legend(fontsize=9)

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Sauvegardé → {p}")
    return fig
