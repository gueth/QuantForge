"""
QuantForge - Module 2: Risk Engine Demo
=========================================
End-to-end demonstration of the risk analytics pipeline:
  1. Data (synthetic or real market data via yfinance)
  2. VaR / CVaR (historical + parametric, multiple horizons)
  3. PCA decomposition
  4. Fama-French 3-factor regression
  5. Stress testing
  6. Rolling VaR
  7. C++ vs Python benchmark (requires compiled extension)
  8. Visualizations

Usage
-----
    cd 02_risk_engine/src
    python main.py                   # synthetic data
    python main.py --real            # real data via yfinance
    python main.py --real --ff       # real data + real Fama-French factors
"""
from __future__ import annotations

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure UTF-8 output on Windows (cp1252 terminals reject box-drawing chars)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from portfolio import (
    Portfolio, synthetic_portfolio, synthetic_ff_factors, load_fama_french,
    _HAS_CPP,
)
from visualizer import (
    plot_var_dashboard, plot_pca_dashboard, plot_ff_dashboard,
    plot_rolling_var, plot_benchmark,
)


def sep(title: str = "") -> None:
    w = 58
    if title:
        pad = (w - len(title) - 2) // 2
        print(f"\n{'-'*pad} {title} {'-'*(w - pad - len(title) - 2)}")
    else:
        print("-" * w)


# ---------------------------------------------------------------------------
# Pure-Python baseline functions (used for benchmarking vs C++)
# ---------------------------------------------------------------------------

def _py_var_historical(
    returns: np.ndarray, weights: np.ndarray, confidence: float = 0.95
) -> float:
    port_r = returns @ weights
    return float(-np.quantile(port_r, 1.0 - confidence))


def _py_cvar_historical(
    returns: np.ndarray, weights: np.ndarray, confidence: float = 0.95
) -> float:
    port_r = returns @ weights
    var    = np.quantile(port_r, 1.0 - confidence)
    tail   = port_r[port_r <= var]
    return float(-tail.mean()) if len(tail) > 0 else 0.0


def _py_pca(returns: np.ndarray) -> np.ndarray:
    cov          = np.cov((returns - returns.mean(axis=0)).T)
    eigenvalues, _ = np.linalg.eigh(cov)
    return eigenvalues[::-1]


def benchmark(
    returns: np.ndarray, weights: np.ndarray, n_runs: int = 50
) -> dict:
    """
    Compare pure-Python vs C++ engine on VaR, CVaR, and PCA.
    Requires the compiled risk_engine_cpp extension.
    """
    import risk_engine_cpp as cpp_engine
    engine  = cpp_engine.RiskEngine(returns, weights)
    results = {}

    for label, py_fn, cpp_fn in [
        ("VaR hist.",
         lambda: _py_var_historical(returns, weights),
         lambda: engine.var_historical(0.95, 1)),
        ("CVaR hist.",
         lambda: _py_cvar_historical(returns, weights),
         lambda: engine.cvar_historical(0.95, 1)),
        ("PCA",
         lambda: _py_pca(returns),
         lambda: engine.pca(-1)),
    ]:
        t0 = time.perf_counter()
        for _ in range(n_runs):
            py_fn()
        py_ms = (time.perf_counter() - t0) / n_runs * 1000

        t0 = time.perf_counter()
        for _ in range(n_runs):
            cpp_fn()
        cpp_ms = (time.perf_counter() - t0) / n_runs * 1000

        results[label] = {
            "python_ms": py_ms,
            "cpp_ms":    cpp_ms,
            "speedup":   py_ms / cpp_ms if cpp_ms > 0 else 0.0,
        }

    return results


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def run(use_real: bool = False, use_ff: bool = False) -> None:
    t_start = time.perf_counter()

    print("\n" + "=" * 58)
    print("  QuantForge - Module 2: Risk Engine")
    print("=" * 58)
    print(f"  Backend: {'C++ (compiled)' if _HAS_CPP else 'Python (fallback)'}")

    # -- 1. Data --------------------------------------------------------------
    sep("DATA")

    if use_real:
        try:
            import yfinance as yf
            tickers      = ["AAPL", "MSFT", "JPM", "XOM", "GLD"]
            start, end   = "2019-01-01", "2024-01-01"
            print(f"  Downloading {tickers} ({start} to {end})...")
            raw        = yf.download(tickers, start=start, end=end,
                                     auto_adjust=True, progress=False)
            prices     = raw["Close"][tickers].dropna(how="all")
            returns_df = np.log(prices / prices.shift(1)).dropna()
            print(f"  {len(returns_df)} days, {len(tickers)} assets loaded.")
        except Exception as exc:
            print(f"  Yahoo Finance failed ({exc}) -- using synthetic data.")
            use_real = False

    if not use_real:
        print("  Generating synthetic returns (5 assets, 1260 days)...")
        returns_df = synthetic_portfolio(n_assets=5, n_days=1260)
        print(f"  {len(returns_df)} days, {returns_df.shape[1]} assets.")

    n         = returns_df.shape[1]
    weights   = np.ones(n) / n
    portfolio = Portfolio(returns_df, weights, name="QuantForge EW")
    print(f"\n  {portfolio}")

    # -- 2. VaR / CVaR -------------------------------------------------------
    sep("VaR / CVaR")

    for conf in [0.95, 0.99]:
        report = portfolio.full_report(conf, 1)
        print(report)

    report_10d = portfolio.full_report(0.99, 10)
    print(f"\n  10-day 99% VaR  (Basel III): {report_10d.var_historical*100:.4f}%")
    print(f"  10-day 99% CVaR            : {report_10d.cvar_historical*100:.4f}%")

    # -- 3. PCA ---------------------------------------------------------------
    sep("PCA")

    pca_result = portfolio.pca()
    print("\n" + pca_result.variance_table().to_string(index=False))
    n90 = pca_result.n_factors_for(0.90)
    print(f"\n  {n90} component(s) explain >=90% of variance.")
    print(f"\n  Top-3 factor loadings:\n"
          f"{pca_result.factor_loadings.iloc[:, :3].round(4).to_string()}")

    # -- 4. Fama-French -------------------------------------------------------
    sep("FAMA-FRENCH 3-FACTOR MODEL")

    if use_ff and use_real:
        try:
            factors = load_fama_french(
                returns_df.index[0].strftime("%Y-%m-%d"),
                returns_df.index[-1].strftime("%Y-%m-%d"),
            )
        except Exception as exc:
            print(f"  FF download failed ({exc}) -- using synthetic factors.")
            factors = synthetic_ff_factors(returns_df.index)
    else:
        print("  Using synthetic Fama-French factors.")
        factors = synthetic_ff_factors(returns_df.index)

    ff_result = portfolio.fama_french(factors)
    print(f"\n{ff_result.summary().to_string()}")

    # -- 5. Stress test -------------------------------------------------------
    sep("STRESS TEST")

    assets    = portfolio.assets
    scenarios = {
        "GFC 2008 Crash"      : {a: -0.40 for a in assets},
        "COVID March 2020"    : {a: -0.30 for a in assets},
        "Tech Selloff 2022"   : {assets[0]: -0.50, assets[1]: -0.45,
                                  **{a: -0.10 for a in assets[2:]}},
        "Rate Shock +200 bps" : {a: -0.15 for a in assets},
        "Bull Market Rally"   : {a: +0.25 for a in assets},
    }
    stress = portfolio.stress_test(scenarios)
    print(f"\n{stress.to_string()}")

    # -- 6. Rolling VaR -------------------------------------------------------
    sep("ROLLING VaR")

    window_rv = min(252, portfolio.T // 4)
    rv        = portfolio.rolling_var(confidence=0.95, window=window_rv)
    peak_date = rv.idxmax().strftime("%Y-%m-%d")
    print(f"\n  Window: {window_rv} days | {len(rv)} observations")
    print(f"  Mean VaR   : {rv.mean()*100:.4f}%")
    print(f"  Max VaR    : {rv.max()*100:.4f}%  (peak: {peak_date})")
    print(f"  Min VaR    : {rv.min()*100:.4f}%")

    # -- 7. Benchmark (C++ only) ----------------------------------------------
    sep("BENCHMARK C++ vs Python")

    if _HAS_CPP:
        print(f"  {n_runs := 50} runs per method, T=1260, N=5...")
        bench = benchmark(returns_df.values, weights, n_runs=n_runs)
        print(f"\n  {'Method':<15} {'Python':>10} {'C++':>10} {'Speedup':>10}")
        print(f"  {'-'*47}")
        for method, v in bench.items():
            print(f"  {method:<15} {v['python_ms']:>8.3f} ms "
                  f"{v['cpp_ms']:>8.3f} ms  x{v['speedup']:>5.1f}")
    else:
        print("  C++ extension not compiled -- skipping benchmark.")
        print("  Run: cd 02_risk_engine && python setup.py build_ext --inplace")
        bench = {}

    # -- 8. Visualizations ----------------------------------------------------
    sep("VISUALIZATIONS")

    report_95 = portfolio.full_report(0.95)
    print("  VaR/CVaR dashboard...")
    plot_var_dashboard(portfolio, report_95)

    print("  PCA dashboard...")
    plot_pca_dashboard(pca_result, portfolio, n_show=min(5, n))

    print("  Fama-French dashboard...")
    plot_ff_dashboard(ff_result)

    print("  Rolling VaR dashboard...")
    plot_rolling_var(portfolio, window=window_rv, confidence=0.95)

    if bench:
        print("  Benchmark chart...")
        plot_benchmark(bench)

    # -- Done -----------------------------------------------------------------
    sep()
    elapsed = time.perf_counter() - t_start
    print(f"\n  Module 2 complete -- {elapsed:.2f}s")
    print(f"  Charts saved to: {Path(__file__).parent.parent / 'notebooks'}/\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantForge Risk Engine")
    parser.add_argument("--real", action="store_true",
                        help="Use real market data via yfinance")
    parser.add_argument("--ff",   action="store_true",
                        help="Use real Fama-French factors (requires --real)")
    args = parser.parse_args()
    run(use_real=args.real, use_ff=args.ff)
