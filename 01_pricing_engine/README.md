# Module 1 — Derivative Pricing Engine

> **Stack:** C++17 · Python 3.12 · pybind11 · NumPy · SciPy · Matplotlib

A high-performance options pricing engine implementing two complementary approaches: an analytical Black-Scholes pricer and a Monte Carlo simulator, with a C++ backend exposed to Python via pybind11.

---

## Mathematical Background

### The Asset Model

We model the underlying asset price $`S_t`$ as a **Geometric Brownian Motion**:

```math
dS_t = S_t\left(\mu \, dt + \sigma \, dW_t\right)
```

whose closed-form solution is:

```math
S_T = S_0 \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma W_T\right], \quad W_T \sim \mathcal{N}(0, T)
```

Under the **risk-neutral measure** $`\mathbb{Q}`$, the drift $`\mu`$ is replaced by the risk-free rate $`r`$. This is the key insight of Black-Scholes: option prices are independent of the expected return of the underlying — they depend only on its volatility $`\sigma`$.

### Black-Scholes Formula

The fair price of a European call option with strike $`K`$ and maturity $`T`$ is:

```math
C_0 = S_0\,\Phi(d_1) - Ke^{-rT}\Phi(d_2)
```

where:

```math
d_1 = \frac{\ln(S_0/K) + \left(r + \frac{\sigma^2}{2}\right)T}{\sigma\sqrt{T}}, \qquad d_2 = d_1 - \sigma\sqrt{T}
```

and $`\Phi`$ is the standard normal CDF.

### Monte Carlo Pricing

For options without closed-form solutions (e.g. barrier options), we use Monte Carlo simulation:

```math
C_0 = e^{-rT}\,\mathbb{E}^{\mathbb{Q}}\!\left[\max(S_T - K, 0)\right] \approx e^{-rT} \cdot \frac{1}{N}\sum_{i=1}^{N}\max\!\left(S_T^{(i)} - K,\, 0\right)
```

Each path is simulated as:

```math
S_T^{(i)} = S_0 \exp\!\left[\left(r - \frac{\sigma^2}{2}\right)T + \sigma \sqrt{T}\, Z_i\right], \quad Z_i \sim \mathcal{N}(0,1)
```

---

## Greeks

The Greeks measure the sensitivity of the option price to its parameters:

| Greek | Definition | Formula |
|---|---|---|
| $`\Delta`$ | $`\partial C / \partial S_0`$ | $`\Phi(d_1)`$ |
| $`\Gamma`$ | $`\partial^2 C / \partial S_0^2`$ | $`\phi(d_1) / (S_0 \sigma \sqrt{T})`$ |
| $`\mathcal{V}`$ | $`\partial C / \partial \sigma`$ | $`S_0\, \phi(d_1)\sqrt{T}`$ |

where $`\phi`$ is the standard normal PDF.

---

## Implementation

### File Structure

```
01_pricing_engine/
├── black_scholes.py      # Analytical pricer + Greeks
├── monte_carlo.py        # Vectorized Monte Carlo (vanilla + barrier)
├── mc_bindings.cpp       # C++ Monte Carlo engine
├── mc_bindings.pyd       # Compiled Python extension
├── setup.py              # Build script for pybind11 binding
└── README.md
```

### Python — Analytical Pricer

```python
from black_scholes import bs_call_price, bs_greeks

# European call price
price = bs_call_price(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
# → 10.4506

# Greeks
delta, gamma, vega = bs_greeks(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)
# → delta=0.6368, gamma=0.0188, vega=37.52
```

### Python — Monte Carlo

```python
from monte_carlo import mc_call_price, mc_barrier_call_price

# Vanilla call
price = mc_call_price(S0=100, K=100, r=0.05, sigma=0.2, T=1.0)

# Knock-out barrier call (cancelled if S touches B=120 before maturity)
price = mc_barrier_call_price(S0=100, K=100, B=120, r=0.05, sigma=0.2, T=1.0)
```

### C++ Backend via pybind11

```python
import mc_pricer  # compiled C++ extension

price = mc_pricer.mc_call_price(S0=100, K=100, r=0.05, sigma=0.2, T=1.0, n_paths=100000)
```

To rebuild the C++ extension:

```bash
cd 01_pricing_engine
python setup.py build_ext --inplace
```

---

## Performance Benchmark

Monte Carlo pricing of a European call — 100,000 paths, averaged over 10 runs:

| Implementation | Avg. Latency | Speedup |
|---|---|---|
| Python (vectorized NumPy) | 1710 ms | 1x |
| C++ (g++, `-O2`) via pybind11 | **4.9 ms** | **348x** |

The C++ engine uses `std::mt19937_64` (Mersenne Twister 64-bit) for random number generation and is compiled with `-O2` optimization.

---

## Results

### Price Convergence — Black-Scholes vs Monte Carlo

Both methods converge to the same price across all strikes, confirming the correctness of the simulation.

| S₀ | K | r | σ | T | BS Price | MC Price |
|---|---|---|---|---|---|---|
| 100 | 100 | 5% | 20% | 1y | 10.4506 | ~10.45 |
| 100 | 110 | 5% | 20% | 1y | 6.0401 | ~6.04 |
| 100 | 90 | 5% | 20% | 1y | 16.6994 | ~16.70 |

### Greeks Behavior

- **Delta** decreases monotonically from ~1 (deep in-the-money) to ~0 (deep out-of-the-money)
- **Gamma** and **Vega** peak at-the-money (K = S₀), where uncertainty and volatility sensitivity are maximal

### Terminal Price Distribution

The simulated terminal prices follow a **log-normal distribution**, consistent with the GBM assumption:

- E[S_T] simulated: 105.15 (theoretical: S₀ × e^(rT) = 105.13) ✅
- Median S_T: 103.10 < Mean: 105.15 — right skew confirms log-normality

---

## References

- Black, F. & Scholes, M. (1973). *The Pricing of Options and Corporate Liabilities.* Journal of Political Economy.
- Glasserman, P. (2003). *Monte Carlo Methods in Financial Engineering.* Springer.
- Wilmott, P. (2006). *Paul Wilmott on Quantitative Finance.* Wiley.
