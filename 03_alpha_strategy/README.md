# Module 3 — Alpha Strategy

> **Stack:** C++17 · Python 3.12 · pybind11 · NumPy · pandas · SciPy

Statistical arbitrage and alpha signal generation pipeline: Kalman filter pairs trading, cross-sectional momentum, vectorized backtesting, and purged cross-validation to avoid look-ahead bias.

## Contents

| File | Description |
|------|-------------|
| `src/signals.py` | `KalmanPairsSignal`, `CrossSectionalMomentum`, `ZScoreSignal` |
| `src/backtest.py` | Vectorized backtest engine + `PerformanceMetrics` |
| `src/validation.py` | `PurgedKFold` CV, `WalkForwardAnalysis` |
| `src/visualizer.py` | Performance, signal, comparison and walk-forward dashboards |
| `src/main.py` | End-to-end demo pipeline |
| `include/kalman.hpp` | C++ Kalman filter + rolling z-score + performance metrics |
| `cpp/kalman_bindings.cpp` | pybind11 bindings |
| `tests/test_alpha_strategy.py` | 40+ unit tests |

## Mathematical Background

### Kalman Filter Pairs Trading

We model the spread between two assets $y_t$ and $x_t$ as:

$$y_t = \beta_t\,x_t + \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, V_e)$$

where the hedge ratio $\beta_t$ evolves as a random walk:

$$\beta_t = \beta_{t-1} + w_t, \qquad w_t \sim \mathcal{N}(0, Q)$$

with $Q = \delta/(1-\delta)$ controlled by a small parameter $\delta$.

**Kalman update at time $t$:**

| Step | Formula |
|------|---------|
| Predict | $P_{t\mid t-1} = P_{t-1} + Q$ |
| Innovation | $\nu_t = y_t - \hat\beta_{t-1} x_t$, $\;S_t = x_t^2 P_{t\mid t-1} + V_e$ |
| Gain | $K_t = P_{t\mid t-1}\,x_t / S_t$ |
| Update | $\hat\beta_t = \hat\beta_{t-1} + K_t\nu_t$, $\;P_t = (1 - K_t x_t)P_{t\mid t-1}$ |

**Signal:** rolling z-score of the spread $s_t = y_t - \hat\beta_t x_t$:
- **Long spread** ($s_t$ too low): buy $y_t$, sell $\hat\beta_t\,x_t$
- **Short spread** ($s_t$ too high): sell $y_t$, buy $\hat\beta_t\,x_t$
- **Flat**: close position when $|z_t| < z_\text{exit}$

### Cross-Sectional Momentum

Rank $N$ assets by their trailing $L$-day return; go long the top $q$ fraction and equally short the bottom $q$ fraction (dollar-neutral):

$$w_i = \begin{cases} +1/\lfloor qN\rfloor & \text{top } q\text{-quantile} \\ -1/\lfloor qN\rfloor & \text{bottom } q\text{-quantile} \\ 0 & \text{otherwise} \end{cases}$$

### Purged K-Fold Cross-Validation

Standard k-fold leaks information via serial correlation. Purged KFold (Lopez de Prado 2018):
1. Create $K$ sequential folds.
2. For each test fold, **purge** the $n_\text{purge}$ observations immediately before it from the training set.
3. Add an **embargo** gap of $n_\text{embargo}$ observations after the test fold.

### Performance Metrics

| Metric | Formula |
|--------|---------|
| Sharpe | $(\mu_\text{ann} - r_f) / \sigma_\text{ann}$ |
| Sortino | $(\mu_\text{ann} - r_f) / (\sigma_\text{down}\sqrt{252})$ |
| Calmar | $\mu_\text{ann} / \text{MaxDD}$ |
| Hit Rate | $N_\text{win} / N_\text{trades}$ |
| Profit Factor | $\sum\text{wins} / |\sum\text{losses}|$ |

## Project Structure

```
03_alpha_strategy/
├── README.md
├── setup.py              # C++ extension build (pybind11, C++17)
├── include/
│   ├── alpha_types.hpp   # KalmanResult, PerformanceMetrics structs
│   └── kalman.hpp        # Kalman filter, rolling z-score, metrics
├── src/
│   ├── signals.py        # Signal generators
│   ├── backtest.py       # Backtesting engine
│   ├── validation.py     # PurgedKFold, WalkForwardAnalysis
│   ├── visualizer.py     # Matplotlib dashboards
│   └── main.py           # Runnable demo
├── cpp/
│   └── kalman_bindings.cpp  # pybind11 bindings
└── tests/
    └── test_alpha_strategy.py
```

## Quick Start

```python
from src.signals    import KalmanPairsSignal, synthetic_pair
from src.backtest   import Backtest
from src.validation import PurgedKFold

# Generate a cointegrated pair
y, x = synthetic_pair(n_days=1000, beta=0.8, seed=42)

# Fit Kalman signal
sig = KalmanPairsSignal(delta=1e-5, Ve=1e-3, enter_z=2.0, exit_z=0.5)
sig.fit(y, x)

# Backtest with transaction costs
pnl = sig.pnl_series(y, x)
bt  = Backtest(pnl, signal_series=sig.signal_, transaction_cost=2e-4)
m   = bt.run()
print(m)   # Sharpe, Sortino, Calmar, Max DD, Hit Rate ...

# Purged K-Fold
pkf = PurgedKFold(n_splits=5, purge_pct=0.02)
```

## Build C++ Extension

```bash
cd 03_alpha_strategy
pip install pybind11
python setup.py build_ext --inplace
```

The module falls back to pure NumPy if the extension is not compiled.

## Run the Demo

```bash
cd 03_alpha_strategy/src
python main.py
```

## Run Tests

```bash
cd 03_alpha_strategy
python -m pytest tests/ -v
```

## References

- Kalman, R.E. (1960). *A new approach to linear filtering and prediction problems*.
- Lopez de Prado, M. (2018). *Advances in Financial Machine Learning*, Chapter 7.
- Avellaneda, M. & Lee, J.H. (2010). *Statistical Arbitrage in the US Equities Market*.
