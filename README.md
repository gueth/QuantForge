# QuantForge
### A Full-Stack Quantitative Trading System — from Stochastic Pricing to Execution

[![Language](https://img.shields.io/badge/C%2B%2B-17-blue)](https://isocpp.org/)
[![Language](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

QuantForge is an end-to-end quantitative trading system built from scratch, covering the full production pipeline of a quant desk: **derivative pricing → risk management → alpha generation → optimal execution**.

Each module is self-contained and production-inspired, combining rigorous mathematics with high-performance C++ and Python.

---

## Architecture

```
QuantForge/
├── 01_pricing_engine/     # C++17 Monte Carlo pricer + Python bindings
├── 02_risk_engine/        # Portfolio risk: VaR, CVaR, PCA factor model
├── 03_alpha_strategy/     # Stat-arb: Kalman filter, purged CV backtesting
├── 04_execution_sim/      # Almgren-Chriss optimal execution simulator
├── notebooks/             # Analysis, visualizations, results
└── README.md
```

---

## Modules

### Module 1 — Derivative Pricing Engine ✅
> *Stochastic calculus · Monte Carlo · Black-Scholes · Greeks · C++17 · pybind11*

A high-performance options pricing engine implementing both analytical (Black-Scholes) and numerical (Monte Carlo) methods, with a C++ backend exposed to Python via pybind11.

**Key results:**

| Method | Price (S₀=100, K=100, σ=0.2, r=5%, T=1y) | Latency |
|---|---|---|
| Black-Scholes analytical | 10.4506 | < 1ms |
| Monte Carlo (Python) | ~10.45 | 1710ms |
| Monte Carlo (C++, -O2) | ~10.45 | **4.9ms** |

The C++ engine is **348x faster** than the pure Python implementation.

→ [See Module 1 documentation](01_pricing_engine/README.md)

---

### Module 2 — Risk Engine 🔄
> *Portfolio VaR · CVaR · PCA · Fama-French factor model*

*In progress*

---

### Module 3 — Alpha Strategy 🔄
> *Statistical arbitrage · Kalman filter · Purged cross-validation*

*In progress*

---

### Module 4 — Execution Simulator 🔄
> *Almgren-Chriss model · Market impact · Optimal execution · C++*

*In progress*

---

## Mathematical Foundation

The system is built on the following mathematical pillars.

**Stochastic Calculus** — Geometric Brownian Motion under the risk-neutral measure $`\mathbb{Q}`$:

```math
S_T = S_0 \exp\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma W_T\right], \quad W_T \sim \mathcal{N}(0, T)
```

**Black-Scholes Formula** — closed-form European call price:

```math
C_0 = S_0\,\Phi(d_1) - Ke^{-rT}\Phi(d_2)
```

```math
d_1 = \frac{\ln(S_0/K) + (r + \sigma^2/2)\,T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
```

**Monte Carlo Pricing** — law of large numbers applied to risk-neutral expectation:

```math
C_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\max(S_T - K,\, 0)\right] \approx \frac{e^{-rT}}{N}\sum_{i=1}^{N}\max\!\left(S_T^{(i)} - K,\, 0\right)
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| High-performance core | C++17, g++ with -O2 |
| Python interface | pybind11, NumPy, SciPy |
| Analysis & visualization | Jupyter, Matplotlib |
| Version control | Git / GitLab |

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities*
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering*
- López de Prado, M. (2018). *Advances in Financial Machine Learning*
- Almgren, R. & Chriss, N. (2001). *Optimal Execution of Portfolio Transactions*
- Cartea, Á., Jaimungal, S. & Penalva, J. (2015). *Algorithmic and High-Frequency Trading*

---

## Author

**gueth** — Quantitative Developer  
Built as a self-directed deep dive into quantitative finance, numerical methods, and high-performance computing.
