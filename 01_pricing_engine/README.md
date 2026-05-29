# Module 1 — Pricing Engine

> **Stack:** C++17 · Python 3.12 · pybind11 · NumPy · SciPy · Matplotlib

Closed-form and Monte Carlo pricing of vanilla and exotic options, with a high-performance C++ backend for path simulation.

## Contents

| File | Description |
|------|-------------|
| `src/black_scholes.py` | BS closed-form pricer, full Greeks, implied volatility (bisection) |
| `src/monte_carlo.py` | MC pricer — antithetic variance reduction, up-and-out barrier |
| `src/visualizer.py` | Pricing, Greeks and MC dashboards (dark-theme matplotlib) |
| `src/main.py` | End-to-end demo pipeline |
| `cpp/mc_bindings.cpp` | pybind11 C++ Monte Carlo engine |
| `tests/test_pricing_engine.py` | 30+ unit tests |

## Mathematical Background

### Black-Scholes Formula

Under risk-neutral dynamics $dS = rS\,dt + \sigma S\,dW$:

$$C = S_0\,\Phi(d_1) - K e^{-rT}\,\Phi(d_2), \quad P = Ke^{-rT}\Phi(-d_2) - S_0\Phi(-d_1)$$

$$d_1 = \frac{\ln(S_0/K) + (r + \frac{1}{2}\sigma^2)T}{\sigma\sqrt{T}}, \quad d_2 = d_1 - \sigma\sqrt{T}$$

### Greeks

| Greek | Call | Put |
|-------|------|-----|
| Delta | $\Phi(d_1)$ | $\Phi(d_1)-1$ |
| Gamma | $\phi(d_1)/(S\sigma\sqrt{T})$ | same |
| Vega  | $S\phi(d_1)\sqrt{T}\;/\;100$ (per 1% vol) | same |
| Theta | $-\tfrac{S\phi(d_1)\sigma}{2\sqrt{T}} - rKe^{-rT}\Phi(d_2)$ | $+rKe^{-rT}\Phi(-d_2)$ |
| Rho   | $KTe^{-rT}\Phi(d_2)\;/\;100$ | $-KTe^{-rT}\Phi(-d_2)\;/\;100$ |

### Monte Carlo

$$S_T = S_0\exp\!\left[\left(r - \tfrac{\sigma^2}{2}\right)T + \sigma\sqrt{T}\,Z\right], \quad Z\sim\mathcal{N}(0,1)$$

**Antithetic variates** — pair each draw $(Z, -Z)$ to halve estimator variance at no extra cost.

**Up-and-out barrier** — monitored at $n_\text{steps}$ equally-spaced dates; option pays zero if $S_t \geq B$ at any step.

**Implied volatility** — bisection search on $\sigma \in (10^{-6},\,10)$, tolerance $10^{-6}$.

## Project Structure

```
01_pricing_engine/
├── README.md
├── setup.py              # C++ extension build (pybind11)
├── src/
│   ├── black_scholes.py  # BS pricing, Greeks, implied vol
│   ├── monte_carlo.py    # MC pricing, antithetic, barrier
│   ├── visualizer.py     # Matplotlib dashboards
│   └── main.py           # Runnable demo
├── cpp/
│   ├── mc_pricer.cpp     # Standalone C++ pricer
│   └── mc_bindings.cpp   # pybind11 bindings
└── tests/
    ├── test_pricing_engine.py
    └── test_binding.py
```

## Quick Start

```python
from black_scholes import bs_call_price, bs_greeks, bs_implied_vol
from monte_carlo   import mc_call_price, mc_call_price_antithetic

# Closed-form ATM call — should be ≈ 10.4506
price = bs_call_price(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)

# All 8 Greeks
g = bs_greeks(S0=100, K=100, r=0.05, sigma=0.20, T=1.0)
print(g["delta_call"], g["gamma"], g["theta_call"])

# Implied volatility round-trip
iv = bs_implied_vol(price, S0=100, K=100, r=0.05, T=1.0)
assert abs(iv - 0.20) < 1e-5

# Monte Carlo with antithetic variance reduction
import numpy as np
np.random.seed(42)
mc = mc_call_price_antithetic(100, 100, 0.05, 0.20, 1.0, n_paths=50_000)
```

## Build C++ Extension

```bash
cd 01_pricing_engine
pip install pybind11
python setup.py build_ext --inplace
```

## Run the Demo

```bash
python 01_pricing_engine/src/main.py
```

Charts are saved to `01_pricing_engine/outputs/`.

## Run Tests

```bash
# From the project root
pytest 01_pricing_engine/tests/ -v
```

## Performance

| Method | 100k paths | Notes |
|--------|-----------|-------|
| Python (NumPy, vectorised) | ~30 ms | Single-step GBM |
| C++ (-O2) | ~5 ms | Same algorithm |
| **Speedup** | **~6×** | |
