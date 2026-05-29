# QuantForge

### A Full-Stack Quantitative Trading System — from Stochastic Pricing to Optimal Execution

[![Language](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)
[![Language](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![CI](https://github.com/gueth/QuantForge/actions/workflows/ci.yml/badge.svg)](https://github.com/gueth/QuantForge/actions)

QuantForge is an end-to-end quantitative trading system built from scratch, covering the full pipeline of a quantitative desk: **derivative pricing → risk analytics → alpha generation → optimal execution**.

Each module is self-contained and production-inspired: rigorous mathematics, high-performance C++17 backends compiled via pybind11, comprehensive Python APIs, publication-quality visualizations, and pytest test suites.

---

## Architecture

```
QuantForge/
├── 01_pricing_engine/   ─── Black-Scholes · Monte Carlo · Barrier options
├── 02_risk_engine/      ─── VaR · CVaR · PCA · Fama-French · Rolling VaR
├── 03_alpha_strategy/   ─── Kalman pairs trading · Momentum · Purged CV
└── 04_execution_sim/    ─── Almgren-Chriss · TWAP/VWAP · IS Monte Carlo
```

---

## Modules

### Module 1 — Pricing Engine

| Component | Details |
|-----------|---------|
| **BS pricing** | Closed-form call/put + full Greeks (Δ, Γ, ν, Θ, ρ) |
| **Implied vol** | Bisection solver, tolerance $10^{-6}$ |
| **Monte Carlo** | GBM simulation, antithetic variates, up-and-out barrier |
| **C++ backend** | 6× speedup vs NumPy for 100k-path simulation |

```
src/black_scholes.py   src/monte_carlo.py   src/visualizer.py   src/main.py
cpp/mc_bindings.cpp
tests/test_pricing_engine.py   (30+ tests)
```

### Module 2 — Risk Engine

| Component | Details |
|-----------|---------|
| **VaR / CVaR** | Historical and parametric, square-root-of-time scaling |
| **PCA** | Jacobi eigensolver (from scratch), factor loadings & returns |
| **Fama-French** | 3-factor OLS regression, $\alpha$, $\beta$, $R^2$, systematic/idio vol |
| **Rolling VaR** | Sliding-window risk, configurable window and confidence |
| **Stress testing** | Named scenario P&L attribution |
| **C++ backend** | 3× speedup on VaR/CVaR; from-scratch Gauss-Jordan OLS |

```
src/portfolio.py   src/visualizer.py   src/main.py
include/linalg.hpp   include/risk_engine.hpp   include/risk_types.hpp
cpp/bindings.cpp
tests/test_risk_engine.py   (44 tests)
```

### Module 3 — Alpha Strategy

| Component | Details |
|-----------|---------|
| **Kalman pairs** | Dynamic hedge ratio $\beta_t$ via scalar Kalman filter |
| **CS momentum** | Long top-$q$, short bottom-$q$ quartile, dollar-neutral |
| **Z-score MR** | Rolling z-score mean-reversion signal |
| **Backtest** | Vectorized engine, transaction costs, 9 performance metrics |
| **Purged K-Fold** | Anti-leakage CV (Lopez de Prado 2018) |
| **Walk-forward** | Expanding/rolling window OOS evaluation |
| **C++ backend** | Fast Kalman filter and performance metrics computation |

```
src/signals.py   src/backtest.py   src/validation.py
src/visualizer.py   src/main.py
include/kalman.hpp   include/alpha_types.hpp
cpp/kalman_bindings.cpp
tests/test_alpha_strategy.py   (40+ tests)
```

### Module 4 — Execution Simulator

| Component | Details |
|-----------|---------|
| **Almgren-Chriss** | Optimal trajectory: $x_k = X\sinh(\kappa(N-k)\tau)/\sinh(\kappa N\tau)$ |
| **TWAP** | Uniform $n_k = X/N$ schedule |
| **VWAP** | U-shaped intraday volume-profile participation |
| **MC simulation** | Permanent + temporary impact + Brownian timing risk |
| **IS decomposition** | Perm. impact, temp. impact, timing risk |
| **Efficient frontier** | $(\mathbb{E}[\text{IS}], \sigma[\text{IS}])$ curve as $\lambda$ varies |
| **C++ backend** | Fast Monte Carlo IS simulation |

```
src/schedules.py   src/simulator.py   src/visualizer.py   src/main.py
include/almgren_chriss.hpp   include/exec_types.hpp
cpp/exec_bindings.cpp
tests/test_execution_sim.py   (30+ tests)
```

---

## Uniform Module Structure

Every module follows the same layout:

```
0X_module_name/
├── README.md          # Module documentation with mathematical background
├── setup.py           # pybind11 C++ extension build script
├── include/           # C++ headers (header-only, no external dependencies)
├── src/               # Python source (module.py, visualizer.py, main.py)
├── cpp/               # C++ source and pybind11 bindings
└── tests/             # pytest unit tests

QuantForge/
├── pyproject.toml     # Project metadata, dependencies, pytest config
├── conftest.py        # Adds each src/ to sys.path — no path hacks in tests
└── .github/workflows/ci.yml  # GitHub Actions: tests on Python 3.10/3.11/3.12
```

---

## Quick Start

```bash
# 1. Install the project and dev dependencies
pip install -e ".[dev]"

# 2. (Optional) Compile C++ extensions for each module
cd 01_pricing_engine && python setup.py build_ext --inplace && cd ..
cd 02_risk_engine     && python setup.py build_ext --inplace && cd ..
cd 03_alpha_strategy  && python setup.py build_ext --inplace && cd ..
cd 04_execution_sim   && python setup.py build_ext --inplace && cd ..

# 3. Run any module demo
python 01_pricing_engine/src/main.py
python 02_risk_engine/src/main.py
python 03_alpha_strategy/src/main.py
python 04_execution_sim/src/main.py
```

All modules fall back to pure NumPy when the C++ extension is not compiled.

---

## Run All Tests

```bash
# From the project root — runs all 150+ tests across every module
pytest

# Or target a specific module
pytest 02_risk_engine/tests/ -v
```

Total: **150+ unit tests** covering mathematical invariants, edge cases, and performance contracts.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Numerical core | C++17 (header-only, zero external C++ deps) |
| Python binding | pybind11 |
| Data/analysis | NumPy · SciPy · pandas |
| Visualisation | Matplotlib (dark-theme, publication-ready) |
| Testing | pytest (config in `pyproject.toml`) |
| Build | setuptools · `pyproject.toml` · pybind11 |
| CI | GitHub Actions (Python 3.10 / 3.11 / 3.12) |

---

## Mathematical Foundations

| Topic | Module | Key references |
|-------|--------|----------------|
| Black-Scholes PDE | 1 | Black & Scholes (1973), Merton (1973) |
| Monte Carlo methods | 1 | Glasserman (2003) *Monte Carlo Methods in Financial Engineering* |
| Value at Risk | 2 | Basel Committee on Banking Supervision (2019) |
| PCA / Factor models | 2 | Jolliffe (2002) *Principal Component Analysis* |
| Fama-French factors | 2 | Fama & French (1993) |
| Kalman filter | 3 | Kalman (1960), Avellaneda & Lee (2010) |
| Purged CV | 3 | Lopez de Prado (2018) *Advances in Financial ML* |
| Almgren-Chriss | 4 | Almgren & Chriss (2001) *Optimal Execution of Portfolio Transactions* |
