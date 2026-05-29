"""
QuantForge — Module 1: Visualization Dashboards
=================================================
Publication-ready charts for option pricing analysis.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from scipy.stats import norm

from black_scholes import (
    bs_call_price, bs_put_price, bs_greeks, bs_implied_vol
)
from monte_carlo import mc_call_price

OUT = Path(__file__).parent.parent / "notebooks"
OUT.mkdir(parents=True, exist_ok=True)

# ── Palette ──────────────────────────────────────────────────
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


# ── 1. Pricing Dashboard ──────────────────────────────────────

def plot_pricing_dashboard(
    S0: float = 100.0,
    r: float  = 0.05,
    sigma: float = 0.20,
    T: float  = 1.0,
    save: bool = True,
    fname: str = "pricing_dashboard.png",
) -> plt.Figure:
    """
    4 panels:
    A — Call & Put prices vs Strike (BS closed-form + MC comparison)
    B — Implied volatility smile (flat for BS, perturbed for demonstration)
    C — Price surface vs (T, sigma)
    D — Put-call parity verification across strikes
    """
    strikes = np.linspace(60, 140, 60)

    fig = _fig()
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── A : Prices vs Strike ──────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    bs_calls = [bs_call_price(S0, K, r, sigma, T) for K in strikes]
    bs_puts  = [bs_put_price(S0, K, r, sigma, T)  for K in strikes]
    np.random.seed(42)
    mc_calls = [mc_call_price(S0, K, r, sigma, T, 50_000) for K in strikes[::5]]

    ax1.plot(strikes, bs_calls, color=C["accent1"], lw=1.8, label="Call (BS)")
    ax1.plot(strikes, bs_puts,  color=C["accent2"], lw=1.8, label="Put (BS)")
    ax1.scatter(strikes[::5], mc_calls, color=C["accent3"], s=20, zorder=5,
                label="Call (MC, 50k)")
    ax1.axvline(S0, color=C["muted"], lw=1, ls="--", label=f"ATM (S={S0})")
    ax1.set_title("Call & Put Prices vs Strike", color=C["text"])
    ax1.set_xlabel("Strike K"); ax1.set_ylabel("Option Price")
    ax1.legend(fontsize=8)

    # ── B : Implied Volatility Smile ──────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    # Perturb BS prices slightly to simulate a vol smile
    np.random.seed(7)
    noise = np.random.normal(0, 0.05, len(strikes)) * np.abs(strikes - S0) / S0
    market_calls = np.array(bs_calls) * (1 + noise)
    market_calls = np.maximum(market_calls, 0.01)

    ivs = []
    for K, mp in zip(strikes, market_calls):
        iv = bs_implied_vol(mp, S0, K, r, T, "call")
        ivs.append(iv * 100 if not np.isnan(iv) else np.nan)

    valid = [(k, v) for k, v in zip(strikes, ivs) if not np.isnan(v)]
    if valid:
        ks, vs = zip(*valid)
        ax2.plot(ks, vs, color=C["accent4"], lw=1.8)
        ax2.axhline(sigma * 100, color=C["muted"], lw=1, ls="--",
                    label=f"Flat vol {sigma*100:.0f}%")
    ax2.set_title("Implied Volatility Smile", color=C["text"])
    ax2.set_xlabel("Strike K"); ax2.set_ylabel("Implied Vol (%)")
    ax2.legend(fontsize=8)

    # ── C : Price Surface ─────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    sigmas = np.linspace(0.10, 0.50, 30)
    maturities = np.linspace(0.1, 2.0, 30)
    SG, TG = np.meshgrid(sigmas, maturities)
    prices_surf = np.vectorize(lambda s, t: bs_call_price(S0, S0, r, s, t))(SG, TG)
    im = ax3.contourf(sigmas * 100, maturities, prices_surf, levels=20, cmap="plasma")
    plt.colorbar(im, ax=ax3, label="Call Price")
    ax3.set_title("ATM Call Price Surface (T, σ)", color=C["text"])
    ax3.set_xlabel("Volatility σ (%)"); ax3.set_ylabel("Maturity T (years)")

    # ── D : Put-Call Parity Error ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    parity_errors = [
        abs(bs_call_price(S0, K, r, sigma, T) - bs_put_price(S0, K, r, sigma, T)
            - (S0 - K * np.exp(-r * T)))
        for K in strikes
    ]
    ax4.semilogy(strikes, parity_errors, color=C["accent1"], lw=1.5)
    ax4.axhline(1e-10, color=C["muted"], lw=1, ls="--", label="1e-10 reference")
    ax4.set_title("Put-Call Parity Numerical Error", color=C["text"])
    ax4.set_xlabel("Strike K"); ax4.set_ylabel("Absolute Error")
    ax4.legend(fontsize=8)

    fig.suptitle(
        f"QuantForge — Pricing Engine  |  S={S0}, r={r*100:.0f}%, σ={sigma*100:.0f}%, T={T}y",
        color=C["text"], fontsize=12, y=0.99
    )

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig


# ── 2. Greeks Dashboard ───────────────────────────────────────

def plot_greeks_dashboard(
    S0: float = 100.0,
    K: float  = 100.0,
    r: float  = 0.05,
    sigma: float = 0.20,
    T: float  = 1.0,
    save: bool = True,
    fname: str = "greeks_dashboard.png",
) -> plt.Figure:
    """
    4 panels: Delta, Gamma, Vega, Theta — call & put — vs spot price.
    """
    spots = np.linspace(60, 140, 200)
    greek_series: dict[str, list] = {
        g: [] for g in ("delta_call", "delta_put", "gamma",
                        "vega", "theta_call", "theta_put")
    }
    for S in spots:
        g = bs_greeks(S, K, r, sigma, T)
        for key in greek_series:
            greek_series[key].append(g[key])

    fig = _fig()
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)
    specs = [
        ("Delta", "delta_call", "delta_put", gs[0, 0]),
        ("Gamma", "gamma",      None,        gs[0, 1]),
        ("Vega (per 1% vol)",  "vega",  None, gs[1, 0]),
        ("Theta (per day)",    "theta_call", "theta_put", gs[1, 1]),
    ]

    for title, key_call, key_put, pos in specs:
        ax = fig.add_subplot(pos)
        ax.plot(spots, greek_series[key_call], color=C["accent1"], lw=1.8,
                label="Call")
        if key_put:
            ax.plot(spots, greek_series[key_put], color=C["accent2"], lw=1.8,
                    label="Put")
        ax.axvline(K, color=C["muted"], lw=1, ls="--", label=f"K={K}")
        ax.axhline(0, color=C["muted"], lw=0.6, ls=":")
        ax.set_title(title, color=C["text"])
        ax.set_xlabel("Spot Price S"); ax.legend(fontsize=8)

    fig.suptitle(
        f"QuantForge — Greeks  |  K={K}, r={r*100:.0f}%, σ={sigma*100:.0f}%, T={T}y",
        color=C["text"], fontsize=12, y=0.99
    )

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig


# ── 3. Monte Carlo Dashboard ──────────────────────────────────

def plot_mc_dashboard(
    S0: float = 100.0,
    K: float  = 100.0,
    r: float  = 0.05,
    sigma: float = 0.20,
    T: float  = 1.0,
    save: bool = True,
    fname: str = "mc_dashboard.png",
) -> plt.Figure:
    """
    4 panels:
    A — Terminal price distribution (log-normal verification)
    B — MC price convergence as n_paths grows
    C — Barrier option price vs barrier level
    D — Antithetic variance reduction demonstration
    """
    from monte_carlo import (
        mc_call_price, mc_call_price_antithetic, mc_barrier_call_price
    )

    fig = _fig()
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

    # ── A : Terminal price distribution ──────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    np.random.seed(42)
    Z   = np.random.standard_normal(200_000)
    ST  = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    ax1.hist(ST, bins=100, density=True, color=C["accent1"], alpha=0.6,
             edgecolor="none", label="Simulated S_T")
    xs  = np.linspace(ST.min(), ST.max(), 400)
    mu_ln  = np.log(S0) + (r - 0.5 * sigma**2) * T
    sig_ln = sigma * np.sqrt(T)
    pdf_ln = (1 / (xs * sig_ln * np.sqrt(2 * np.pi)) *
              np.exp(-0.5 * ((np.log(xs) - mu_ln) / sig_ln)**2))
    ax1.plot(xs, pdf_ln, color=C["accent3"], lw=1.8, label="Log-normal fit")
    ax1.axvline(ST.mean(), color=C["accent4"], lw=1.5, ls="--",
                label=f"E[S_T]={ST.mean():.2f}")
    ax1.set_title("Terminal Price Distribution", color=C["text"])
    ax1.set_xlabel("S_T"); ax1.set_ylabel("Density")
    ax1.legend(fontsize=8)

    # ── B : MC Convergence ───────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    path_counts = [500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
    bs_ref = bs_call_price(S0, K, r, sigma, T)
    mc_prices = []
    for n in path_counts:
        np.random.seed(42)
        mc_prices.append(mc_call_price(S0, K, r, sigma, T, n))
    ax2.semilogx(path_counts, mc_prices, color=C["accent1"], lw=1.6,
                 marker="o", ms=5, label="MC price")
    ax2.axhline(bs_ref, color=C["accent3"], lw=1.5, ls="--",
                label=f"BS exact = {bs_ref:.4f}")
    ax2.set_title("MC Convergence vs Path Count", color=C["text"])
    ax2.set_xlabel("Number of Paths (log scale)"); ax2.set_ylabel("Call Price")
    ax2.legend(fontsize=8)

    # ── C : Barrier option vs barrier level ──────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    barriers = np.linspace(105, 160, 25)
    np.random.seed(42)
    barrier_prices = []
    for B in barriers:
        np.random.seed(42)
        barrier_prices.append(mc_barrier_call_price(S0, K, B, r, sigma, T, 30_000))
    vanilla_price = bs_call_price(S0, K, r, sigma, T)
    ax3.plot(barriers, barrier_prices, color=C["accent2"], lw=1.8,
             marker="o", ms=4, label="Up-and-out call")
    ax3.axhline(vanilla_price, color=C["accent3"], lw=1.5, ls="--",
                label=f"Vanilla call = {vanilla_price:.2f}")
    ax3.set_title("Barrier Call Price vs Barrier Level", color=C["text"])
    ax3.set_xlabel("Barrier B"); ax3.set_ylabel("Option Price")
    ax3.legend(fontsize=8)

    # ── D : Antithetic variance reduction ────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    n_trials, n_paths_small = 300, 5_000
    naive_prices = []
    anti_prices  = []
    for seed in range(n_trials):
        np.random.seed(seed)
        naive_prices.append(mc_call_price(S0, K, r, sigma, T, n_paths_small))
        np.random.seed(seed)
        anti_prices.append(mc_call_price_antithetic(S0, K, r, sigma, T, n_paths_small))
    ax4.hist(naive_prices, bins=30, density=True, color=C["accent4"],
             alpha=0.6, label=f"Naive σ={np.std(naive_prices):.4f}")
    ax4.hist(anti_prices,  bins=30, density=True, color=C["accent3"],
             alpha=0.6, label=f"Antithetic σ={np.std(anti_prices):.4f}")
    ax4.axvline(bs_ref, color=C["accent2"], lw=1.5, ls="--",
                label=f"BS exact = {bs_ref:.4f}")
    ax4.set_title(f"Antithetic Variance Reduction ({n_paths_small} paths)",
                  color=C["text"])
    ax4.set_xlabel("Estimated Price"); ax4.set_ylabel("Density")
    ax4.legend(fontsize=8)

    fig.suptitle(
        f"QuantForge — Monte Carlo Pricing  |  S={S0}, K={K}, σ={sigma*100:.0f}%",
        color=C["text"], fontsize=12, y=0.99
    )

    if save:
        p = OUT / fname
        fig.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  ✓ Saved → {p}")
    return fig
