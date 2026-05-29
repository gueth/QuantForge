"""
QuantForge — Module 1: Pricing Engine Demo
===========================================
End-to-end demonstration of the pricing engine:
  1. Black-Scholes closed-form pricing
  2. Greeks computation
  3. Implied volatility round-trip
  4. Monte Carlo pricing with variance reduction
  5. Barrier option pricing
  6. C++ vs Python benchmark
  7. Visualizations

Usage
-----
    cd 01_pricing_engine/src
    python main.py
"""
from __future__ import annotations

import sys
import time
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from black_scholes import (
    bs_call_price, bs_put_price, bs_greeks, bs_implied_vol,
)
from monte_carlo import (
    mc_call_price, mc_put_price,
    mc_call_price_antithetic, mc_barrier_call_price,
)
from visualizer import (
    plot_pricing_dashboard, plot_greeks_dashboard, plot_mc_dashboard,
)


def sep(title: str = ""):
    w = 58
    if title:
        p = (w - len(title) - 2) // 2
        print(f"\n{'─'*p} {title} {'─'*(w - p - len(title) - 2)}")
    else:
        print("─" * w)


def run():
    t_global = time.perf_counter()

    print("\n" + "═" * 58)
    print("  QuantForge — Module 1: Pricing Engine")
    print("═" * 58)

    # Reference parameters
    S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0

    # ── 1. Black-Scholes Pricing ──────────────────────────────
    sep("BLACK-SCHOLES")

    cases = [
        ("ATM  (K=100)", 100.0),
        ("OTM  (K=110)", 110.0),
        ("ITM  (K=90) ", 90.0),
    ]
    print(f"\n  {'Contract':<20} {'Call':>10} {'Put':>10}  {'Parity err':>12}")
    print(f"  {'─'*55}")
    for label, k in cases:
        call = bs_call_price(S0, k, r, sigma, T)
        put  = bs_put_price(S0, k, r, sigma, T)
        err  = abs(call - put - (S0 - k * np.exp(-r * T)))
        print(f"  {label:<20} {call:>10.4f} {put:>10.4f}  {err:>12.2e}")

    # ── 2. Greeks ─────────────────────────────────────────────
    sep("GREEKS (ATM)")

    g = bs_greeks(S0, K, r, sigma, T)
    print(f"\n  {'Greek':<15} {'Call':>10} {'Put':>10}")
    print(f"  {'─'*38}")
    greek_pairs = [
        ("Delta",   "delta_call", "delta_put"),
        ("Gamma",   "gamma",      None),
        ("Vega",    "vega",       None),
        ("Theta",   "theta_call", "theta_put"),
        ("Rho",     "rho_call",   "rho_put"),
    ]
    for name, ck, pk in greek_pairs:
        cv = f"{g[ck]:>10.4f}"
        pv = f"{g[pk]:>10.4f}" if pk else f"{'(same)':>10}"
        print(f"  {name:<15} {cv} {pv}")

    # ── 3. Implied Volatility ──────────────────────────────────
    sep("IMPLIED VOLATILITY")

    test_vols = [0.15, 0.20, 0.25, 0.30, 0.35]
    print(f"\n  {'True σ':>10} {'IV (call)':>12} {'IV (put)':>12} {'Error':>10}")
    print(f"  {'─'*48}")
    for true_vol in test_vols:
        mp_call = bs_call_price(S0, K, r, true_vol, T)
        mp_put  = bs_put_price(S0, K, r, true_vol, T)
        iv_call = bs_implied_vol(mp_call, S0, K, r, T, "call")
        iv_put  = bs_implied_vol(mp_put,  S0, K, r, T, "put")
        err = max(abs(iv_call - true_vol), abs(iv_put - true_vol))
        print(f"  {true_vol*100:>9.1f}% {iv_call*100:>11.4f}% {iv_put*100:>11.4f}% "
              f"{err:.2e}")

    # ── 4. Monte Carlo Pricing ─────────────────────────────────
    sep("MONTE CARLO")

    bs_ref = bs_call_price(S0, K, r, sigma, T)
    print(f"\n  BS exact call = {bs_ref:.4f}")
    print(f"\n  {'Method':<30} {'Price':>8} {'Error':>8} {'Time (ms)':>10}")
    print(f"  {'─'*60}")

    methods = [
        ("MC naive (100k)", lambda: mc_call_price(S0, K, r, sigma, T, 100_000)),
        ("MC antithetic (50k)", lambda: mc_call_price_antithetic(S0, K, r, sigma, T, 50_000)),
        ("MC put (100k)", lambda: mc_put_price(S0, K, r, sigma, T, 100_000)),
    ]
    np.random.seed(42)
    for name, fn in methods:
        t0    = time.perf_counter()
        price = fn()
        ms    = (time.perf_counter() - t0) * 1000
        ref   = bs_ref if "put" not in name else bs_put_price(S0, K, r, sigma, T)
        print(f"  {name:<30} {price:>8.4f} {abs(price-ref):>8.4f} {ms:>10.1f}")

    # ── 5. Barrier Options ────────────────────────────────────
    sep("BARRIER OPTIONS")

    print(f"\n  Vanilla call (BS):          {bs_ref:.4f}")
    barriers = [110, 120, 130, 150]
    print(f"\n  {'Barrier':>10} {'Price':>10} {'Discount %':>12}")
    print(f"  {'─'*36}")
    np.random.seed(42)
    for B in barriers:
        np.random.seed(42)
        bp = mc_barrier_call_price(S0, K, B, r, sigma, T, n_paths=100_000)
        disc = (bs_ref - bp) / bs_ref * 100
        print(f"  B={B:>5}   {bp:>10.4f}   {disc:>10.1f}%")

    # ── 6. C++ vs Python Benchmark ────────────────────────────
    sep("BENCHMARK  (C++ vs Python)")

    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        import mc_pricer as cpp
        n_runs = 20
        n_paths = 100_000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            mc_call_price(S0, K, r, sigma, T, n_paths)
        py_ms = (time.perf_counter() - t0) / n_runs * 1000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            cpp.mc_call_price(S0, K, r, sigma, T, n_paths)
        cpp_ms = (time.perf_counter() - t0) / n_runs * 1000

        print(f"\n  MC ({n_paths:,} paths, {n_runs} runs avg)")
        print(f"  Python  : {py_ms:>8.2f} ms")
        print(f"  C++ -O2 : {cpp_ms:>8.2f} ms")
        print(f"  Speedup :  ×{py_ms/cpp_ms:.0f}")
    except ImportError:
        print("\n  C++ extension not compiled. Run: python setup.py build_ext --inplace")

    # ── 7. Visualizations ─────────────────────────────────────
    sep("VISUALIZATIONS")

    print("  Pricing dashboard...")
    plot_pricing_dashboard(S0, r, sigma, T)

    print("  Greeks dashboard...")
    plot_greeks_dashboard(S0, K, r, sigma, T)

    print("  Monte Carlo dashboard...")
    plot_mc_dashboard(S0, K, r, sigma, T)

    # ── Done ──────────────────────────────────────────────────
    sep()
    elapsed = time.perf_counter() - t_global
    print(f"\n  ✓ Module 1 complete — {elapsed:.2f}s")
    print(f"  Charts → {Path(__file__).parent.parent / 'notebooks'}/\n")


if __name__ == "__main__":
    run()
