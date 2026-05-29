"""
QuantForge - Module 2: Python API Layer
========================================
High-level Python wrapper around the risk engine.

Classes
-------
Portfolio      : Multi-asset portfolio — delegates to C++ (or Python fallback).
RiskReport     : Structured VaR/CVaR result.
PCAResult      : PCA decomposition result.
FFResult       : Fama-French regression result.

Utilities
---------
synthetic_portfolio()   : Correlated synthetic returns for offline testing.
synthetic_ff_factors()  : Synthetic Fama-French factors for testing.
load_fama_french()      : Download real FF3 daily factors from Kenneth French.
"""
from __future__ import annotations

import io
import sys
import zipfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.stats import norm as _norm

# ---------------------------------------------------------------------------
# Engine Protocol — structural interface shared by C++ and Python engines
# ---------------------------------------------------------------------------

@runtime_checkable
class _RiskEngineProtocol(Protocol):
    """
    Structural interface that both the compiled C++ engine and the pure-Python
    fallback must satisfy.  Using a Protocol (PEP 544) means Pylance infers
    correct types for every ``self._engine`` call without requiring an ABC or
    inheritance hierarchy.
    """

    def portfolio_returns(self) -> np.ndarray: ...

    @property
    def n_obs(self) -> int: ...

    @property
    def n_assets(self) -> int: ...

    def var_historical(
        self, confidence: float = ..., horizon: int = ...
    ) -> float: ...

    def var_parametric(
        self, confidence: float = ..., horizon: int = ...
    ) -> float: ...

    def cvar_historical(
        self, confidence: float = ..., horizon: int = ...
    ) -> float: ...

    def cvar_parametric(
        self, confidence: float = ..., horizon: int = ...
    ) -> float: ...

    def full_report(
        self, confidence: float = ..., horizon: int = ...
    ) -> dict[str, Any]: ...

    def pca(self, n_components: int = ...) -> dict[str, Any]: ...

    def fama_french(
        self,
        factors: np.ndarray,
        rf:      np.ndarray,
        names:   list[str],
    ) -> list[dict[str, Any]]: ...

    def rolling_var(
        self,
        confidence: float = ...,
        window:     int   = ...,
        parametric: bool  = ...,
    ) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# C++ backend — optional, falls back to pure Python when not compiled
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

_HAS_CPP: bool = False
try:
    import risk_engine_cpp as _cpp_module  # type: ignore[import-untyped]
    _HAS_CPP = True
except ImportError:
    _cpp_module = None  # type: ignore[assignment]

CACHE_DIR = _HERE.parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FF3_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french"
    "/ftp/F-F_Research_Data_Factors_daily_CSV.zip"
)


# ---------------------------------------------------------------------------
# Pure-Python risk engine  (satisfies _RiskEngineProtocol via duck typing)
# ---------------------------------------------------------------------------

class _PyRiskEngine:
    """
    NumPy / SciPy implementation of the risk engine.
    Mirrors the C++ RiskEngine API so Portfolio works without compilation.
    """

    def __init__(self, returns: np.ndarray, weights: np.ndarray) -> None:
        self._returns:      np.ndarray = returns
        self._weights:      np.ndarray = weights
        self._T:            int        = returns.shape[0]
        self._N:            int        = returns.shape[1]
        self._port_returns: np.ndarray = returns @ weights

    # -- Accessors -----------------------------------------------------------
    def portfolio_returns(self) -> np.ndarray:
        return self._port_returns.copy()

    @property
    def n_obs(self) -> int:
        return self._T

    @property
    def n_assets(self) -> int:
        return self._N

    # -- VaR / CVaR ----------------------------------------------------------
    def var_historical(self, confidence: float = 0.95, horizon: int = 1) -> float:
        q = float(np.quantile(self._port_returns, 1.0 - confidence))
        return -q * float(np.sqrt(horizon))

    def var_parametric(self, confidence: float = 0.95, horizon: int = 1) -> float:
        mu    = float(self._port_returns.mean())
        sigma = float(self._port_returns.std(ddof=1))
        z     = float(_norm.ppf(1.0 - confidence))
        return -(mu + z * sigma) * float(np.sqrt(horizon))

    def cvar_historical(self, confidence: float = 0.95, horizon: int = 1) -> float:
        q    = float(np.quantile(self._port_returns, 1.0 - confidence))
        tail = self._port_returns[self._port_returns <= q]
        if len(tail) == 0:
            return 0.0
        return float(-tail.mean()) * float(np.sqrt(horizon))

    def cvar_parametric(self, confidence: float = 0.95, horizon: int = 1) -> float:
        mu    = float(self._port_returns.mean())
        sigma = float(self._port_returns.std(ddof=1))
        alpha = 1.0 - confidence
        z     = float(_norm.ppf(alpha))
        es    = -(mu - sigma * float(_norm.pdf(z)) / alpha)
        return es * float(np.sqrt(horizon))

    def full_report(
        self, confidence: float = 0.95, horizon: int = 1
    ) -> dict[str, Any]:
        return {
            "confidence":      confidence,
            "horizon_days":    horizon,
            "n_obs":           self._T,
            "var_historical":  self.var_historical(confidence, horizon),
            "var_parametric":  self.var_parametric(confidence, horizon),
            "cvar_historical": self.cvar_historical(confidence, horizon),
            "cvar_parametric": self.cvar_parametric(confidence, horizon),
            "portfolio_vol":   float(self._port_returns.std(ddof=1) * np.sqrt(252)),
            "portfolio_mean":  float(self._port_returns.mean() * 252),
        }

    # -- PCA -----------------------------------------------------------------
    def pca(self, n_components: int = -1) -> dict[str, Any]:
        k = self._N if n_components < 0 or n_components > self._N else n_components

        R_c                       = self._returns - self._returns.mean(axis=0)
        cov                       = np.cov(R_c.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        order        = np.argsort(eigenvalues)[::-1]
        eigenvalues  = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        total_var = float(eigenvalues.sum())
        evr       = eigenvalues / total_var
        cumvar    = np.cumsum(evr)

        evecs_k = eigenvectors[:, :k]
        return {
            "eigenvalues":              eigenvalues[:k],
            "explained_variance_ratio": evr[:k],
            "cumulative_variance":      cumvar[:k],
            "factor_loadings":          evecs_k,
            "factor_returns":           R_c @ evecs_k,
            "n_components":             k,
            "n_assets":                 self._N,
        }

    # -- Fama-French ---------------------------------------------------------
    def fama_french(
        self,
        factors: np.ndarray,
        rf:      np.ndarray,
        names:   list[str],
    ) -> list[dict[str, Any]]:
        T_common = min(self._T, len(factors))
        K        = factors.shape[1]
        X        = np.column_stack([np.ones(T_common), factors[:T_common]])
        results: list[dict[str, Any]] = []

        for n in range(self._N):
            rf_vec = rf[:T_common] if len(rf) > 0 else np.zeros(T_common)
            y      = self._returns[:T_common, n] - rf_vec

            coef, *_ = np.linalg.lstsq(X, y, rcond=None)
            y_hat    = X @ coef
            eps      = y - y_hat

            ss_res = float((eps ** 2).sum())
            ss_tot = float(((y - y.mean()) ** 2).sum())
            r2     = 1.0 - ss_res / ss_tot if ss_tot > 1e-14 else 0.0

            systematic = factors[:T_common] @ coef[1:]
            results.append({
                "asset":          names[n] if n < len(names) else f"Asset_{n}",
                "alpha":          float(coef[0]),
                "betas":          coef[1:].astype(np.float64),
                "r_squared":      r2,
                "systematic_vol": float(systematic.std(ddof=1) * np.sqrt(252)),
                "idio_vol":       float(eps.std(ddof=1) * np.sqrt(252)),
                "residuals":      eps.astype(np.float64),
            })

        return results

    # -- Rolling VaR ---------------------------------------------------------
    def rolling_var(
        self,
        confidence: float = 0.95,
        window:     int   = 252,
        parametric: bool  = False,
    ) -> np.ndarray:
        if window >= self._T:
            raise ValueError(
                f"window ({window}) must be strictly less than T ({self._T})"
            )
        result = np.empty(self._T - window + 1)
        for i in range(window, self._T + 1):
            sub           = _PyRiskEngine(self._returns[i - window:i], self._weights)
            result[i - window] = (
                sub.var_parametric(confidence) if parametric
                else sub.var_historical(confidence)
            )
        return result


# ---------------------------------------------------------------------------
# Engine factory
# ---------------------------------------------------------------------------

def _make_engine(
    returns_arr: np.ndarray,
    weights_arr: np.ndarray,
) -> _RiskEngineProtocol:
    """Return the C++ engine when compiled, otherwise the Python fallback."""
    if _HAS_CPP and _cpp_module is not None:
        return _cpp_module.RiskEngine(returns_arr, weights_arr)  # type: ignore[no-any-return]
    return _PyRiskEngine(returns_arr, weights_arr)


# ---------------------------------------------------------------------------
# Portfolio
# ---------------------------------------------------------------------------

class Portfolio:
    """
    Multi-asset portfolio.

    Parameters
    ----------
    returns : pd.DataFrame
        Daily log-returns, shape (T, N). Index must be a DatetimeIndex.
    weights : array-like, optional
        Portfolio weights, shape (N,). Must sum to 1. Defaults to equal-weight.
    name : str
        Display name used in reports and charts.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        name:    str = "Portfolio",
    ) -> None:
        self.returns: pd.DataFrame = returns.astype(float)
        self.name:    str          = name
        n = returns.shape[1]

        if weights is None:
            self.weights: np.ndarray = np.ones(n) / n
        else:
            self.weights = np.asarray(weights, dtype=float)

        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(
                f"Weights must sum to 1 (got {self.weights.sum():.6f})"
            )
        if len(self.weights) != n:
            raise ValueError(
                f"weights length {len(self.weights)} != n_assets {n}"
            )

        self._engine: _RiskEngineProtocol = _make_engine(
            returns.values.astype(np.float64),
            self.weights.astype(np.float64),
        )

    # -- Read-only properties ------------------------------------------------
    @property
    def assets(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def portfolio_returns(self) -> pd.Series:
        arr = self._engine.portfolio_returns()
        return pd.Series(arr, index=self.returns.index, name=self.name)

    @property
    def T(self) -> int:
        return self._engine.n_obs

    @property
    def N(self) -> int:
        return self._engine.n_assets

    # -- VaR / CVaR ----------------------------------------------------------
    def var_historical(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Historical VaR: -Q_{1-alpha}(r_p) * sqrt(horizon)."""
        return self._engine.var_historical(confidence, horizon)

    def var_parametric(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Parametric (Gaussian) VaR: -(mu + z_alpha * sigma) * sqrt(horizon)."""
        return self._engine.var_parametric(confidence, horizon)

    def cvar_historical(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Historical CVaR (Expected Shortfall): -E[r | r <= VaR]."""
        return self._engine.cvar_historical(confidence, horizon)

    def cvar_parametric(self, confidence: float = 0.95, horizon: int = 1) -> float:
        """Parametric CVaR: -(mu - sigma * phi(z_alpha) / (1-alpha))."""
        return self._engine.cvar_parametric(confidence, horizon)

    def full_report(self, confidence: float = 0.95, horizon: int = 1) -> "RiskReport":
        """Return a complete VaR/CVaR RiskReport."""
        d = self._engine.full_report(confidence, horizon)
        return RiskReport(
            portfolio_name  = self.name,
            confidence      = d["confidence"],
            horizon_days    = d["horizon_days"],
            n_obs           = d["n_obs"],
            var_historical  = d["var_historical"],
            var_parametric  = d["var_parametric"],
            cvar_historical = d["cvar_historical"],
            cvar_parametric = d["cvar_parametric"],
            portfolio_vol   = d["portfolio_vol"],
            portfolio_mean  = d["portfolio_mean"],
        )

    # -- PCA -----------------------------------------------------------------
    def pca(self, n_components: Optional[int] = None) -> "PCAResult":
        """
        Principal Component Analysis of the return covariance matrix.

        Parameters
        ----------
        n_components : int, optional
            Number of principal components to retain. Defaults to N (all).
        """
        k     = n_components if n_components is not None else self.N
        d     = self._engine.pca(k)
        names = [f"PC{i + 1}" for i in range(k)]

        return PCAResult(
            eigenvalues              = np.asarray(d["eigenvalues"]),
            explained_variance_ratio = np.asarray(d["explained_variance_ratio"]),
            cumulative_variance      = np.asarray(d["cumulative_variance"]),
            factor_loadings          = pd.DataFrame(
                d["factor_loadings"], index=self.assets, columns=names
            ),
            factor_returns           = pd.DataFrame(
                d["factor_returns"], index=self.returns.index, columns=names
            ),
            asset_names              = self.assets,
        )

    # -- Fama-French ---------------------------------------------------------
    def fama_french(self, factors: pd.DataFrame) -> "FFResult":
        """
        Fama-French k-factor OLS regression for each asset.

        Parameters
        ----------
        factors : pd.DataFrame
            Factor returns aligned to the same calendar as portfolio returns.
            Must include a column named 'RF' or 'RFR' for the risk-free rate.
            Requires at least 60 overlapping observations.
        """
        common = self.returns.index.intersection(factors.index)
        if len(common) < 60:
            raise ValueError(
                f"Only {len(common)} common dates — need at least 60."
            )

        rf_col = next(
            (c for c in factors.columns if c.upper() in ("RF", "RFR")), None
        )
        f_cols = [c for c in factors.columns if c != rf_col]

        F: np.ndarray  = np.asarray(factors.loc[common, f_cols], dtype=np.float64)
        rf: np.ndarray = (
            np.asarray(factors.loc[common, rf_col], dtype=np.float64)
            if rf_col else np.zeros(len(common))
        )

        sub_port = Portfolio(self.returns.loc[common], self.weights, self.name)
        raw      = sub_port._engine.fama_french(F, rf, self.assets)

        return FFResult(raw=raw, factor_names=f_cols, asset_names=self.assets)

    # -- Rolling VaR ---------------------------------------------------------
    def rolling_var(
        self,
        confidence: float = 0.95,
        window:     int   = 252,
        parametric: bool  = False,
    ) -> pd.Series:
        """
        Sliding-window VaR series.

        Returns a Series of length T - window + 1, indexed from
        returns.index[window-1] to returns.index[-1].
        """
        arr = self._engine.rolling_var(confidence, window, parametric)
        idx = self.returns.index[window - 1:]
        tag = "param" if parametric else "hist"
        return pd.Series(arr, index=idx, name=f"Rolling VaR ({tag})")

    # -- Stress test ---------------------------------------------------------
    def stress_test(
        self, scenarios: dict[str, dict[str, float]]
    ) -> pd.DataFrame:
        """
        Apply named shock scenarios and compute portfolio P&L.

        Parameters
        ----------
        scenarios : dict mapping scenario name -> {asset: shock_return}
            Assets not in the portfolio are silently ignored.

        Returns
        -------
        pd.DataFrame with a single column 'P&L (%)'.
        """
        rows: dict[str, dict[str, float]] = {}
        for scenario_name, shocks in scenarios.items():
            pnl = sum(
                self.weights[self.assets.index(a)] * shock
                for a, shock in shocks.items()
                if a in self.assets
            )
            rows[scenario_name] = {"P&L (%)": round(pnl * 100, 2)}
        return pd.DataFrame(rows).T

    def __repr__(self) -> str:
        pr = self.portfolio_returns
        ann_ret = float(pr.mean()) * 252 * 100
        ann_vol = float(pr.std())  * float(np.sqrt(252)) * 100
        return (
            f"Portfolio('{self.name}' | T={self.T}, N={self.N} | "
            f"ann_ret={ann_ret:.2f}%, ann_vol={ann_vol:.2f}%)"
        )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass
class RiskReport:
    """Structured VaR/CVaR summary for a portfolio."""
    portfolio_name  : str
    confidence      : float
    horizon_days    : int
    n_obs           : int
    var_historical  : float
    var_parametric  : float
    cvar_historical : float
    cvar_parametric : float
    portfolio_vol   : float    # annualised
    portfolio_mean  : float    # annualised

    def __str__(self) -> str:
        bar = "-" * 52
        return (
            f"\n{bar}\n"
            f"  {self.portfolio_name}  |  "
            f"{self.confidence * 100:.0f}%  |  {self.horizon_days}d\n"
            f"{bar}\n"
            f"  Ann. Vol             {self.portfolio_vol * 100:>10.2f} %\n"
            f"  Ann. Return          {self.portfolio_mean * 100:>10.2f} %\n"
            f"  Observations         {self.n_obs:>10d}\n"
            f"{bar}\n"
            f"  VaR  Historical      {self.var_historical * 100:>10.4f} %\n"
            f"  VaR  Parametric      {self.var_parametric * 100:>10.4f} %\n"
            f"  CVaR Historical      {self.cvar_historical * 100:>10.4f} %\n"
            f"  CVaR Parametric      {self.cvar_parametric * 100:>10.4f} %\n"
            f"{bar}"
        )


@dataclass
class PCAResult:
    """Principal Component Analysis result."""
    eigenvalues              : np.ndarray
    explained_variance_ratio : np.ndarray
    cumulative_variance      : np.ndarray
    factor_loadings          : pd.DataFrame   # shape (n_assets, k)
    factor_returns           : pd.DataFrame   # shape (T, k)
    asset_names              : list[str]

    def variance_table(self) -> pd.DataFrame:
        """Return a DataFrame summarising explained variance per component."""
        k = len(self.eigenvalues)
        return pd.DataFrame({
            "Component"          : [f"PC{i + 1}" for i in range(k)],
            "Eigenvalue"         : self.eigenvalues.round(6),
            "Explained Var (%)"  : (self.explained_variance_ratio * 100).round(2),
            "Cumulative Var (%)" : (self.cumulative_variance * 100).round(2),
        })

    def n_factors_for(self, threshold: float = 0.90) -> int:
        """Minimum number of components to explain >= threshold of variance."""
        return int(np.searchsorted(self.cumulative_variance, threshold)) + 1


@dataclass
class FFResult:
    """Fama-French k-factor regression result."""
    raw          : list[dict[str, Any]]
    factor_names : list[str]
    asset_names  : list[str]

    @property
    def betas(self) -> np.ndarray:
        """Beta matrix, shape (n_assets, n_factors)."""
        return np.array([r["betas"] for r in self.raw])

    @property
    def alphas(self) -> np.ndarray:
        """Daily alpha vector, shape (n_assets,)."""
        return np.array([r["alpha"] for r in self.raw])

    def summary(self) -> pd.DataFrame:
        """Return a DataFrame with alpha, R2, betas, systematic and idio vol."""
        rows = []
        for r in self.raw:
            row: dict[str, float] = {
                "Alpha ann. (%)": round(r["alpha"] * 252 * 100, 4),
                "R2"             : round(r["r_squared"],          4),
                "Sys. Vol (%)"   : round(r["systematic_vol"] * 100, 4),
                "Idio. Vol (%)"  : round(r["idio_vol"] * 100,     4),
            }
            for i, fname in enumerate(self.factor_names):
                row[f"beta_{fname}"] = round(float(r["betas"][i]), 4)
            rows.append(row)
        return pd.DataFrame(rows, index=self.asset_names)

    def residuals(self) -> pd.DataFrame:
        """Return a DataFrame of OLS residuals, one column per asset."""
        return pd.DataFrame(
            {r["asset"]: r["residuals"] for r in self.raw}
        )


# ---------------------------------------------------------------------------
# Synthetic data generators
# ---------------------------------------------------------------------------

def synthetic_portfolio(
    n_assets: int = 5,
    n_days:   int = 1260,
    start:    str = "2019-01-01",
    seed:     int = 42,
) -> pd.DataFrame:
    """
    Generate correlated log-returns for offline testing.

    Returns a DataFrame of shape (n_days, n_assets) with a business-day
    DatetimeIndex starting at ``start``.
    """
    rng     = np.random.default_rng(seed)
    tickers = [f"ASSET_{i + 1}" for i in range(n_assets)]

    # Random positive-definite covariance matrix
    A   = rng.standard_normal((n_assets, n_assets))
    cov = A @ A.T / n_assets + np.eye(n_assets) * 0.1
    L   = np.linalg.cholesky(cov)

    mu      = rng.uniform(1e-4, 8e-4, n_assets)
    vol     = rng.uniform(0.008, 0.020, n_assets)
    z       = rng.standard_normal((n_days, n_assets))
    returns = mu + (z @ L.T) * vol
    dates   = pd.bdate_range(start=start, periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=tickers)


def synthetic_ff_factors(
    index: pd.DatetimeIndex,
    seed:  int = 99,
) -> pd.DataFrame:
    """Synthetic Fama-French 3 factors for unit testing."""
    rng = np.random.default_rng(seed)
    T   = len(index)
    return pd.DataFrame(
        {
            "Mkt-RF": rng.normal(3e-4, 0.010, T),
            "SMB"   : rng.normal(1e-4, 0.005, T),
            "HML"   : rng.normal(1e-4, 0.005, T),
            "RF"    : np.full(T, 1.5e-4),
        },
        index=index,
    )


def load_fama_french(start: str, end: str) -> pd.DataFrame:
    """
    Download daily Fama-French 3-factor data from Kenneth French's website.

    Results are cached locally at ``data/ff3_daily.parquet``.
    Requires an internet connection on the first call.
    """
    cache = CACHE_DIR / "ff3_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache).loc[start:end]

    print("  Downloading Fama-French 3-factor data...")
    req = urllib.request.Request(FF3_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        raw_text = zf.open(csv_name).read().decode("latin-1")

    lines:      list[str] = raw_text.split("\n")
    in_data:    bool      = False
    data_lines: list[str] = []
    for line in lines:
        s = line.strip()
        if not s:
            if in_data:
                break
            continue
        if s[0].isdigit() and len(s.split(",")) >= 4:
            in_data = True
            data_lines.append(s)
        elif in_data:
            break

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None,
        names=["Date", "Mkt-RF", "SMB", "HML", "RF"],
        dtype={"Date": str},
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce").dropna() / 100.0
    df.to_parquet(cache)
    print(f"  Cached -> {cache}")
    return df.loc[start:end]
