# Module 2 — Risk Engine

> *Portfolio VaR · CVaR · PCA · Fama-French · C++17 · pybind11*

Moteur de risque quantitatif complet avec backend C++ haute-performance exposé à Python via pybind11.

---

## Architecture

```
02_risk_engine/
├── include/
│   ├── risk_types.hpp      # Structures de données (VaRResult, PCAResult, FFResult)
│   ├── linalg.hpp          # Algèbre linéaire from-scratch (Jacobi, OLS, Cholesky, norm_ppf)
│   └── risk_engine.hpp     # Moteur C++ (VaR, CVaR, PCA, FF, Rolling VaR)
├── bindings/
│   └── bindings.cpp        # pybind11 : C++ → Python
├── python/
│   ├── risk_engine_cpp.so  # Extension compilée
│   ├── portfolio.py        # Interface pandas + wrapper haut-niveau
│   ├── visualizer.py       # Dashboards matplotlib (dark theme)
│   └── main.py             # Pipeline complet
├── tests/
│   └── test_risk_engine.py # 44 tests unitaires
├── notebooks/              # Graphiques générés
└── data/                   # Cache Fama-French
```

---

## Compilation

```bash
cd 02_risk_engine

PY_INC=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))")
PB_INC=$(python3 -c "import pybind11; print(pybind11.get_include())")
PY_EXT=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

g++ -O2 -std=c++17 -shared -fPIC \
    -I include -I "$PY_INC" -I "$PB_INC" \
    bindings/bindings.cpp \
    -o "python/risk_engine_cpp${PY_EXT}"
```

---

## Utilisation

```bash
# Pipeline complet (données synthétiques)
cd python && python main.py

# Données réelles + Fama-French
python main.py --real --ff
```

### API Python

```python
import numpy as np
from portfolio import Portfolio, synthetic_portfolio, synthetic_ff_factors

# Créer un portefeuille
returns = synthetic_portfolio(n_assets=5, n_days=1260)
port = Portfolio(returns, name="Mon Portefeuille")

# VaR / CVaR
report = port.full_report(confidence=0.99, horizon=10)
print(report)

# PCA
pca = port.pca(n_components=3)
print(pca.variance_table())

# Fama-French
factors = synthetic_ff_factors(returns.index)
ff = port.fama_french(factors)
print(ff.summary())

# Stress test
stress = port.stress_test({"GFC": {"ASSET_1": -0.40, "ASSET_2": -0.35}})

# Rolling VaR
rolling = port.rolling_var(confidence=0.95, window=252)
```

---

## Résultats

### VaR / CVaR (portefeuille 5 actifs, 1260 jours, équipondéré)

| Métrique | 95% | 99% |
|---|---|---|
| VaR Historique (1j) | 1.11% | 1.70% |
| VaR Paramétrique (1j) | 1.12% | 1.58% |
| CVaR Historique (1j) | 1.50% | 2.16% |
| CVaR Paramétrique (1j) | 1.40% | 1.82% |
| VaR Historique 10j (Basel III) | 3.51% | 5.36% |

### PCA — Variance expliquée

| Composante | Valeur propre | Expliqué | Cumulé |
|---|---|---|---|
| PC1 | 0.000271 | 22.6% | 22.6% |
| PC2 | 0.000245 | 20.5% | 43.1% |
| PC3 | 0.000238 | 19.9% | 63.0% |
| PC4 | 0.000230 | 19.2% | 82.1% |
| PC5 | 0.000214 | 17.9% | 100% |

### Benchmark C++ vs Python pur

| Méthode | Python | C++ (-O2) | Accélération |
|---|---|---|---|
| VaR Historique | 0.116 ms | 0.041 ms | **×2.8** |
| CVaR Historique | 0.133 ms | 0.042 ms | **×3.1** |
| PCA (Jacobi) | 0.152 ms | 0.242 ms | ×0.6 (*) |

(*) La PCA C++ utilise l'algorithme de Jacobi from-scratch, plus lent que LAPACK (numpy) sur de petites matrices. Sur de grandes matrices ou en production, le C++ permettrait d'intégrer LAPACK directement.

---

## Fondements mathématiques

**VaR Historique**
```math
\text{VaR}_\alpha = -Q_{1-\alpha}(r_p) \cdot \sqrt{h}
```

**VaR Paramétrique (Gaussien)**
```math
\text{VaR}_\alpha = -(\mu_p + z_\alpha \cdot \sigma_p) \cdot \sqrt{h}, \quad z_\alpha = \Phi^{-1}(1-\alpha)
```

**CVaR / Expected Shortfall**
```math
\text{CVaR}_\alpha = -\mathbb{E}[r_p \mid r_p \leq \text{VaR}_\alpha] \cdot \sqrt{h}
```

**CVaR Paramétrique**
```math
\text{CVaR}_\alpha = -\left(\mu_p - \sigma_p \cdot \frac{\varphi(z_\alpha)}{1-\alpha}\right) \cdot \sqrt{h}
```

**PCA — décomposition spectrale**
```math
\Sigma = V \Lambda V^\top, \quad F = R_{\text{centré}} \cdot V_{[:K]}
```

**Modèle Fama-French 3 facteurs**
```math
r_{i,t} - r_{f,t} = \alpha_i + \beta_{i,\text{Mkt}} \cdot \text{MKT}_t + \beta_{i,\text{SMB}} \cdot \text{SMB}_t + \beta_{i,\text{HML}} \cdot \text{HML}_t + \varepsilon_{i,t}
```

---

## Tests

```bash
cd 02_risk_engine
python -m pytest tests/ -v
# 44 passed in 1.33s
```

Couvre : poids du portefeuille, VaR positive, CVaR ≥ VaR, monotonie en α, scaling √T, formules analytiques exactes, orthogonalité des vecteurs propres, résidus OLS de moyenne nulle, stress test, rolling VaR.
