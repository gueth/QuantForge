# Module 2 — Risk Engine

> **Stack:** C++17 · Python 3.12 · pybind11 · NumPy · pandas · SciPy

Portfolio risk analytics engine with a from-scratch C++ numerical backend: VaR, CVaR, PCA, Fama-French 3-factor regression, rolling VaR and stress testing.

## Contents

| File | Description |
|------|-------------|
| `src/portfolio.py` | `Portfolio` class — high-level Python API wrapping the C++ engine |
| `src/visualizer.py` | VaR, PCA, Fama-French and rolling VaR dashboards |
| `src/main.py` | End-to-end demo pipeline |
| `include/linalg.hpp` | From-scratch linear algebra (Jacobi eigensolver, OLS, Cholesky) |
| `include/risk_engine.hpp` | C++ risk engine core |
| `include/risk_types.hpp` | Result structs: `VaRResult`, `PCAResult`, `FFRegressionResult` |
| `cpp/bindings.cpp` | pybind11 bindings |
| `tests/test_risk_engine.py` | 44 unit tests (mathematical invariants) |

## Mathematical Background

### Value at Risk

| Method | Formula |
|--------|---------|
| Historical VaR | $\text{VaR}_\alpha = -Q_{1-\alpha}(r_p)\,\sqrt{h}$ |
| Parametric VaR | $\text{VaR}_\alpha = -(\mu + \Phi^{-1}(1-\alpha)\,\sigma)\,\sqrt{h}$ |
| Historical CVaR | $\text{CVaR}_\alpha = -\mathbb{E}[r_p \mid r_p \leq \text{VaR}_\alpha]\,\sqrt{h}$ |
| Parametric CVaR | $\text{CVaR}_\alpha = -\!\left(\mu - \sigma\,\frac{\phi(\Phi^{-1}(1-\alpha))}{1-\alpha}\right)\!\sqrt{h}$ |

### PCA

Covariance decomposition $\Sigma = V\Lambda V^\top$, solved via the **Jacobi eigenvalue algorithm** (from scratch in C++, no LAPACK dependency). Factor loadings = eigenvectors; factor returns $= R_{\text{centred}} \cdot V_{:,\,k}$.

### Fama-French 3-Factor Regression

$$r_i - r_f = \alpha_i + \beta_i^{\text{Mkt}}(r_m - r_f) + \beta_i^{\text{SMB}}\,\text{SMB} + \beta_i^{\text{HML}}\,\text{HML} + \varepsilon_i$$

Solved via **Gauss-Jordan OLS** (from scratch). Outputs: $\alpha$, betas, $R^2$, systematic and idiosyncratic volatility.

### Rolling VaR

Sliding-window VaR computed by re-running the risk engine on each sub-period $[t-w, t]$.

## Project Structure

```
02_risk_engine/
├── README.md
├── setup.py              # C++ extension build (pybind11, C++17)
├── include/
│   ├── linalg.hpp        # Linear algebra (Jacobi, OLS, Cholesky, norm_ppf)
│   ├── risk_engine.hpp   # C++ risk engine
│   └── risk_types.hpp    # Result structs
├── src/
│   ├── portfolio.py      # Portfolio, RiskReport, PCAResult, FFResult
│   ├── visualizer.py     # Matplotlib dashboards
│   └── main.py           # Runnable demo
├── cpp/
│   └── bindings.cpp      # pybind11 bindings
├── tests/
│   └── test_risk_engine.py
└── data/                 # Fama-French factor cache
```

## Quick Start

```python
import numpy as np
from portfolio import Portfolio, synthetic_portfolio, synthetic_ff_factors

# Create a portfolio
returns = synthetic_portfolio(n_assets=5, n_days=1260)
port    = Portfolio(returns, name="My Portfolio")

# VaR and CVaR
print(port.var_historical(confidence=0.95))   # 1-day 95% VaR
print(port.cvar_historical(confidence=0.99))  # 1-day 99% CVaR

# Full report
print(port.full_report(confidence=0.95, horizon=10))

# PCA
pca = port.pca(n_components=3)
print(pca.variance_table())

# Fama-French regression
factors = synthetic_ff_factors(returns.index)
ff      = port.fama_french(factors)
print(ff.summary())

# Rolling VaR
rv = port.rolling_var(confidence=0.95, window=252)
```

## Build C++ Extension

```bash
cd 02_risk_engine
pip install pybind11
python setup.py build_ext --inplace
```

## Run the Demo

```bash
python 02_risk_engine/src/main.py                   # Synthetic data
python 02_risk_engine/src/main.py --real            # Real data via yfinance
python 02_risk_engine/src/main.py --real --ff       # Real data + Fama-French factors
```

## Run Tests

```bash
# From the project root
pytest 02_risk_engine/tests/ -v
```

## Performance (T=1260 obs, N=5 assets)

| Method | Python | C++ (-O2) | Speedup |
|--------|--------|-----------|---------|
| VaR Historical | 0.12 ms | 0.04 ms | **3×** |
| CVaR Historical | 0.13 ms | 0.04 ms | **3×** |
| PCA (full) | 0.15 ms | 0.24 ms | 0.6× * |

*Jacobi is slower than NumPy/LAPACK on small matrices (educational trade-off).
