"""
QuantForge — Module 2 : Pipeline principal
============================================
Lance l'analyse complète :
  1. Données (synthétiques ou réelles)
  2. VaR / CVaR (historique + paramétrique)
  3. PCA
  4. Fama-French 3 facteurs
  5. Stress test
  6. Benchmark C++ vs Python pur
  7. Visualisations

Usage
-----
cd 02_risk_engine/python
python main.py                   # données synthétiques
python main.py --real            # données Yahoo Finance
python main.py --real --ff       # + facteurs Fama-French réels
"""
from __future__ import annotations

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from portfolio import (
    Portfolio, synthetic_portfolio, synthetic_ff_factors, load_fama_french
)
from visualizer import (
    plot_var_dashboard, plot_pca_dashboard, plot_ff_dashboard,
    plot_rolling_var, plot_benchmark,
)


def sep(title: str = ""):
    w = 58
    if title:
        p = (w - len(title) - 2) // 2
        print(f"\n{'─'*p} {title} {'─'*(w - p - len(title) - 2)}")
    else:
        print("─" * w)


# ─────────────────────────────────────────────────────────────
# Implémentations Python pur (pour benchmark)
# ─────────────────────────────────────────────────────────────

def _py_var_historical(returns: np.ndarray, weights: np.ndarray,
                       confidence: float = 0.95) -> float:
    port_r = returns @ weights
    return -np.quantile(port_r, 1 - confidence)

def _py_cvar_historical(returns: np.ndarray, weights: np.ndarray,
                        confidence: float = 0.95) -> float:
    port_r = returns @ weights
    var = np.quantile(port_r, 1 - confidence)
    return -port_r[port_r <= var].mean()

def _py_pca(returns: np.ndarray) -> np.ndarray:
    R_c = returns - returns.mean(axis=0)
    cov = np.cov(R_c.T)
    eigenvalues, _ = np.linalg.eigh(cov)
    return eigenvalues[::-1]


def benchmark(returns: np.ndarray, weights: np.ndarray,
              n_runs: int = 50) -> dict:
    """Compare Python pur vs C++ sur VaR, CVaR, PCA."""
    import risk_engine_cpp as cpp

    engine = cpp.RiskEngine(returns, weights)
    results = {}

    # VaR Historique
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _py_var_historical(returns, weights)
    py_ms = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        engine.var_historical(0.95, 1)
    cpp_ms = (time.perf_counter() - t0) / n_runs * 1000

    results["VaR hist."] = {
        "python_ms": py_ms, "cpp_ms": cpp_ms,
        "speedup": py_ms / cpp_ms if cpp_ms > 0 else 0,
    }

    # CVaR Historique
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _py_cvar_historical(returns, weights)
    py_ms = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        engine.cvar_historical(0.95, 1)
    cpp_ms = (time.perf_counter() - t0) / n_runs * 1000

    results["CVaR hist."] = {
        "python_ms": py_ms, "cpp_ms": cpp_ms,
        "speedup": py_ms / cpp_ms if cpp_ms > 0 else 0,
    }

    # PCA
    t0 = time.perf_counter()
    for _ in range(n_runs):
        _py_pca(returns)
    py_ms = (time.perf_counter() - t0) / n_runs * 1000

    t0 = time.perf_counter()
    for _ in range(n_runs):
        engine.pca(-1)
    cpp_ms = (time.perf_counter() - t0) / n_runs * 1000

    results["PCA"] = {
        "python_ms": py_ms, "cpp_ms": cpp_ms,
        "speedup": py_ms / cpp_ms if cpp_ms > 0 else 0,
    }

    return results


# ─────────────────────────────────────────────────────────────
# Pipeline principal
# ─────────────────────────────────────────────────────────────

def run(use_real: bool = False, use_ff: bool = False):
    t_global = time.perf_counter()

    print("\n" + "═"*58)
    print("  QuantForge — Module 2 : Risk Engine")
    print("═"*58)

    # ── 1. Données ────────────────────────────────────────────
    sep("DONNÉES")

    if use_real:
        try:
            import yfinance as yf
            tickers = ["AAPL", "MSFT", "JPM", "XOM", "GLD"]
            start, end = "2019-01-01", "2024-01-01"
            print(f"  Téléchargement {tickers} ({start} → {end})...")
            raw = yf.download(tickers, start=start, end=end,
                              auto_adjust=True, progress=False)
            prices = raw["Close"][tickers].dropna(how="all")
            returns_df = np.log(prices / prices.shift(1)).dropna()
            print(f"  {len(returns_df)} jours, {len(tickers)} actifs.")
        except Exception as e:
            print(f"  Échec Yahoo Finance ({e}) → données synthétiques")
            use_real = False

    if not use_real:
        print("  Génération rendements synthétiques (5 actifs, 1260 jours)...")
        returns_df = synthetic_portfolio(n_assets=5, n_days=1260)
        print(f"  {len(returns_df)} jours, {returns_df.shape[1]} actifs")

    n = returns_df.shape[1]
    weights = np.ones(n) / n
    portfolio = Portfolio(returns_df, weights, name="QuantForge EW")
    print(f"\n  {portfolio}")

    # ── 2. VaR / CVaR ─────────────────────────────────────────
    sep("VaR / CVaR")

    for conf in [0.95, 0.99]:
        report = portfolio.full_report(conf, 1)
        print(report)

    report_10d = portfolio.full_report(0.99, 10)
    print(f"\n  VaR 10j 99% (Basel III) : {report_10d.var_historical*100:.4f}%")
    print(f"  CVaR 10j 99%            : {report_10d.cvar_historical*100:.4f}%")

    # ── 3. PCA ────────────────────────────────────────────────
    sep("PCA")

    pca = portfolio.pca()
    print("\n" + pca.variance_table().to_string(index=False))
    n90 = pca.n_factors_for(0.90)
    print(f"\n  → {n90} composante(s) pour ≥90% variance expliquée")
    print(f"\n  Loadings top-3 :\n{pca.factor_loadings.iloc[:, :3].round(4).to_string()}")

    # ── 4. Fama-French ────────────────────────────────────────
    sep("FAMA-FRENCH 3 FACTEURS")

    if use_ff and use_real:
        try:
            factors = load_fama_french(
                returns_df.index[0].strftime("%Y-%m-%d"),
                returns_df.index[-1].strftime("%Y-%m-%d"),
            )
        except Exception as e:
            print(f"  FF download échoué ({e}) → synthétique")
            factors = synthetic_ff_factors(returns_df.index)
    else:
        print("  Utilisation de facteurs synthétiques.")
        factors = synthetic_ff_factors(returns_df.index)

    ff = portfolio.fama_french(factors)
    print(f"\n{ff.summary().to_string()}")

    # ── 5. Stress test ────────────────────────────────────────
    sep("STRESS TEST")

    assets = portfolio.assets
    scenarios = {
        "Krach 2008"         : {a: -0.40 for a in assets},
        "COVID mars 2020"    : {a: -0.30 for a in assets},
        "Selloff tech 2022"  : {assets[0]: -0.50, assets[1]: -0.45,
                                **{a: -0.10 for a in assets[2:]}},
        "Choc taux +200 bp"  : {a: -0.15 for a in assets},
        "Rally haussier"     : {a: +0.25 for a in assets},
    }
    stress = portfolio.stress_test(scenarios)
    print(f"\n{stress.to_string()}")

    # ── 6. Rolling VaR ────────────────────────────────────────
    sep("ROLLING VaR")

    window_rv = min(252, portfolio.T // 4)
    rv = portfolio.rolling_var(confidence=0.95, window=window_rv)
    print(f"\n  Fenêtre : {window_rv} jours | {len(rv)} observations")
    print(f"  VaR moy.   : {rv.mean()*100:.4f}%")
    print(f"  VaR max    : {rv.max()*100:.4f}%")
    print(f"  VaR min    : {rv.min()*100:.4f}%")
    peak_date = rv.idxmax().strftime("%Y-%m-%d")
    print(f"  Pic le     : {peak_date} ({rv.max()*100:.4f}%)")

    # ── 7. Benchmark C++ vs Python ────────────────────────────
    sep("BENCHMARK C++ vs PYTHON")

    print("  50 runs par méthode, T=1260, N=5...")
    bench = benchmark(returns_df.values, weights)
    print(f"\n  {'Méthode':<15} {'Python':>10} {'C++':>10} {'Accél.':>10}")
    print(f"  {'─'*47}")
    for k, v in bench.items():
        print(f"  {k:<15} {v['python_ms']:>9.3f}ms {v['cpp_ms']:>9.3f}ms   ×{v['speedup']:>5.1f}")

    # ── 8. Visualisations ─────────────────────────────────────
    sep("VISUALISATIONS")

    report_95 = portfolio.full_report(0.95)
    print("  Dashboard VaR/CVaR...")
    plot_var_dashboard(portfolio, report_95)

    print("  Dashboard PCA...")
    plot_pca_dashboard(pca, portfolio, n_show=min(5, n))

    print("  Dashboard Fama-French...")
    plot_ff_dashboard(ff)

    print("  Dashboard Rolling VaR...")
    plot_rolling_var(portfolio, window=window_rv, confidence=0.95)

    print("  Benchmark chart...")
    plot_benchmark(bench)

    # ── Fin ───────────────────────────────────────────────────
    sep()
    elapsed = time.perf_counter() - t_global
    print(f"\n  ✓ Module 2 terminé en {elapsed:.2f}s")
    print(f"  Graphiques → {Path(__file__).parent.parent / 'notebooks'}/\n")


# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantForge Risk Engine")
    parser.add_argument("--real", action="store_true",
                        help="Données réelles via yfinance")
    parser.add_argument("--ff",   action="store_true",
                        help="Facteurs Fama-French réels (requiert --real)")
    args = parser.parse_args()
    run(use_real=args.real, use_ff=args.ff)
