"""
QuantForge — Module 2 : Couche Python
========================================
Wrapper haut-niveau autour du moteur C++ :
- Portfolio  (DataFrame pandas + poids)
- RiskReport (résultats formatés)
- Génération de données synthétiques
- Chargement Fama-French
"""
from __future__ import annotations

import sys
import io
import zipfile
import urllib.request
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Import du moteur C++
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
import risk_engine_cpp as _cpp

CACHE_DIR = _HERE.parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FF3_URL = ("https://mba.tuck.dartmouth.edu/pages/faculty/ken.french"
           "/ftp/F-F_Research_Data_Factors_daily_CSV.zip")

# ──────────────────────────────────────────────────────────────
# Portfolio
# ──────────────────────────────────────────────────────────────

class Portfolio:
    """
    Portefeuille multi-actifs.

    Parameters
    ----------
    returns : pd.DataFrame  T × N  (index = dates, colonnes = tickers)
    weights : array-like     N     (somme = 1, défaut équipondéré)
    name    : str
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        weights: Optional[np.ndarray] = None,
        name: str = "Portfolio",
    ):
        self.returns = returns.astype(float)
        self.name    = name
        n = returns.shape[1]

        if weights is None:
            self.weights = np.ones(n) / n
        else:
            self.weights = np.asarray(weights, dtype=float)

        if not np.isclose(self.weights.sum(), 1.0):
            raise ValueError(f"Les poids doivent sommer à 1 (somme={self.weights.sum():.4f})")
        if len(self.weights) != n:
            raise ValueError(f"Dimension weights {len(self.weights)} ≠ n_assets {n}")

        # Moteur C++
        self._engine = _cpp.RiskEngine(
            returns.values.astype(np.float64),
            self.weights.astype(np.float64),
        )

    # ── Propriétés ──────────────────────────────────────────
    @property
    def assets(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def portfolio_returns(self) -> pd.Series:
        r = self._engine.portfolio_returns()
        return pd.Series(r, index=self.returns.index, name=self.name)

    @property
    def T(self) -> int:
        return self._engine.n_obs

    @property
    def N(self) -> int:
        return self._engine.n_assets

    # ── VaR / CVaR  (délégués au C++) ───────────────────────
    def var_historical(self, confidence: float = 0.95, horizon: int = 1) -> float:
        return self._engine.var_historical(confidence, horizon)

    def var_parametric(self, confidence: float = 0.95, horizon: int = 1) -> float:
        return self._engine.var_parametric(confidence, horizon)

    def cvar_historical(self, confidence: float = 0.95, horizon: int = 1) -> float:
        return self._engine.cvar_historical(confidence, horizon)

    def cvar_parametric(self, confidence: float = 0.95, horizon: int = 1) -> float:
        return self._engine.cvar_parametric(confidence, horizon)

    def full_report(self, confidence: float = 0.95, horizon: int = 1) -> "RiskReport":
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

    # ── PCA ─────────────────────────────────────────────────
    def pca(self, n_components: Optional[int] = None) -> "PCAResult":
        k = n_components if n_components is not None else self.N
        d = self._engine.pca(k)
        names = [f"PC{i+1}" for i in range(k)]

        factor_loadings = pd.DataFrame(
            d["factor_loadings"],
            index   = self.assets,
            columns = names,
        )
        factor_returns = pd.DataFrame(
            d["factor_returns"],
            index   = self.returns.index,
            columns = names,
        )
        return PCAResult(
            eigenvalues              = d["eigenvalues"],
            explained_variance_ratio = d["explained_variance_ratio"],
            cumulative_variance      = d["cumulative_variance"],
            factor_loadings          = factor_loadings,
            factor_returns           = factor_returns,
            asset_names              = self.assets,
        )

    # ── Fama-French ─────────────────────────────────────────
    def fama_french(self, factors: pd.DataFrame) -> "FFResult":
        common = self.returns.index.intersection(factors.index)
        if len(common) < 60:
            raise ValueError(f"Seulement {len(common)} dates communes — besoin ≥ 60")

        rf_col  = next((c for c in factors.columns if c.upper() in ("RF", "RFR")), None)
        f_cols  = [c for c in factors.columns if c != rf_col]

        F  = factors.loc[common, f_cols].values.astype(np.float64)
        rf = factors.loc[common, rf_col].values.astype(np.float64) if rf_col else np.zeros(len(common))

        sub_ret = self.returns.loc[common]
        sub_port = Portfolio(sub_ret, self.weights, self.name)
        raw = sub_port._engine.fama_french(F, rf, self.assets)

        return FFResult(raw=raw, factor_names=f_cols, asset_names=self.assets)

    # ── Rolling VaR ─────────────────────────────────────────
    def rolling_var(
        self, confidence: float = 0.95, window: int = 252, parametric: bool = False,
    ) -> pd.Series:
        arr = self._engine.rolling_var(confidence, window, parametric)
        idx = self.returns.index[window - 1:]
        return pd.Series(arr, index=idx, name=f"Rolling VaR ({'param' if parametric else 'hist'})")

    # ── Stress test ─────────────────────────────────────────
    def stress_test(self, scenarios: dict[str, dict[str, float]]) -> pd.DataFrame:
        rows = {}
        for name, shocks in scenarios.items():
            pnl = sum(
                self.weights[self.assets.index(a)] * s
                for a, s in shocks.items()
                if a in self.assets
            )
            rows[name] = {"P&L (%)": round(pnl * 100, 2)}
        return pd.DataFrame(rows).T

    def __repr__(self) -> str:
        pr = self.portfolio_returns
        return (f"Portfolio('{self.name}' | "
                f"T={self.T}, N={self.N} | "
                f"μ_ann={pr.mean()*252*100:.2f}%, "
                f"σ_ann={pr.std()*np.sqrt(252)*100:.2f}%)")


# ──────────────────────────────────────────────────────────────
# Résultats
# ──────────────────────────────────────────────────────────────

@dataclass
class RiskReport:
    portfolio_name  : str
    confidence      : float
    horizon_days    : int
    n_obs           : int
    var_historical  : float
    var_parametric  : float
    cvar_historical : float
    cvar_parametric : float
    portfolio_vol   : float
    portfolio_mean  : float

    def __str__(self) -> str:
        s = "─" * 52
        return (
            f"\n{s}\n"
            f"  {self.portfolio_name}  |  {self.confidence*100:.0f}%  |  {self.horizon_days}j\n"
            f"{s}\n"
            f"  Vol ann.           {self.portfolio_vol*100:>10.2f} %\n"
            f"  Rendement ann.     {self.portfolio_mean*100:>10.2f} %\n"
            f"  Observations       {self.n_obs:>10d}\n"
            f"{s}\n"
            f"  VaR  Historique    {self.var_historical*100:>10.4f} %\n"
            f"  VaR  Paramétrique  {self.var_parametric*100:>10.4f} %\n"
            f"  CVaR Historique    {self.cvar_historical*100:>10.4f} %\n"
            f"  CVaR Paramétrique  {self.cvar_parametric*100:>10.4f} %\n"
            f"{s}"
        )


@dataclass
class PCAResult:
    eigenvalues              : np.ndarray
    explained_variance_ratio : np.ndarray
    cumulative_variance      : np.ndarray
    factor_loadings          : pd.DataFrame   # assets × composantes
    factor_returns           : pd.DataFrame   # T × composantes
    asset_names              : list[str]

    def variance_table(self) -> pd.DataFrame:
        k = len(self.eigenvalues)
        return pd.DataFrame({
            "Composante"   : [f"PC{i+1}" for i in range(k)],
            "Valeur propre": self.eigenvalues.round(6),
            "Expliqué (%)" : (self.explained_variance_ratio * 100).round(2),
            "Cumulé (%)"   : (self.cumulative_variance * 100).round(2),
        })

    def n_factors_for(self, threshold: float = 0.90) -> int:
        """Nombre de composantes pour atteindre `threshold` de variance expliquée."""
        return int(np.searchsorted(self.cumulative_variance, threshold)) + 1


@dataclass
class FFResult:
    raw          : list      # brut C++
    factor_names : list[str]
    asset_names  : list[str]

    def summary(self) -> pd.DataFrame:
        rows = []
        for r in self.raw:
            row = {
                "Alpha ann. (%)": r["alpha"] * 252 * 100,
                "R²"            : r["r_squared"],
                "Vol sys. (%)"  : r["systematic_vol"] * 100,
                "Vol idio. (%)": r["idio_vol"] * 100,
            }
            for i, f in enumerate(self.factor_names):
                row[f"β_{f}"] = r["betas"][i]
            rows.append(row)
        return pd.DataFrame(rows, index=self.asset_names).round(4)

    def residuals(self) -> pd.DataFrame:
        return pd.DataFrame(
            {r["asset"]: r["residuals"] for r in self.raw}
        )


# ──────────────────────────────────────────────────────────────
# Données synthétiques
# ──────────────────────────────────────────────────────────────

def synthetic_portfolio(
    n_assets : int  = 5,
    n_days   : int  = 1260,
    start    : str  = "2019-01-01",
    seed     : int  = 42,
) -> pd.DataFrame:
    """
    Rendements log corrélés pour tests hors-ligne.
    Retourne DataFrame T × N avec index BusinessDay.
    """
    rng = np.random.default_rng(seed)
    tickers = [f"ASSET_{i+1}" for i in range(n_assets)]

    # Matrice de corrélation aléatoire (garantie définie positive)
    A   = rng.standard_normal((n_assets, n_assets))
    cov = A @ A.T / n_assets + np.eye(n_assets) * 0.1  # régularisation
    L   = np.linalg.cholesky(cov)

    mu  = rng.uniform(1e-4, 8e-4, n_assets)
    vol = rng.uniform(0.008, 0.020, n_assets)
    z   = rng.standard_normal((n_days, n_assets))

    returns = mu + (z @ L.T) * vol
    dates   = pd.bdate_range(start=start, periods=n_days)
    return pd.DataFrame(returns, index=dates, columns=tickers)


def synthetic_ff_factors(
    index : pd.DatetimeIndex,
    seed  : int = 99,
) -> pd.DataFrame:
    """Facteurs Fama-French synthétiques pour tests."""
    rng = np.random.default_rng(seed)
    T   = len(index)
    return pd.DataFrame({
        "Mkt-RF": rng.normal(3e-4, 0.010, T),
        "SMB"   : rng.normal(1e-4, 0.005, T),
        "HML"   : rng.normal(1e-4, 0.005, T),
        "RF"    : np.full(T, 1.5e-4),
    }, index=index)


def load_fama_french(start: str, end: str) -> pd.DataFrame:
    """
    Télécharge les facteurs Fama-French 3 quotidiens depuis Kenneth French.
    Cache local dans data/ff3_daily.parquet.
    """
    cache = CACHE_DIR / "ff3_daily.parquet"
    if cache.exists():
        return pd.read_parquet(cache).loc[start:end]

    print("  Téléchargement Fama-French 3-facteurs...")
    req = urllib.request.Request(FF3_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        content = resp.read()

    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.upper().endswith(".CSV"))
        raw = zf.open(csv_name).read().decode("latin-1")

    lines, in_data, data_lines = raw.split("\n"), False, []
    for line in lines:
        s = line.strip()
        if not s:
            if in_data: break
            continue
        if s[0].isdigit() and len(s.split(",")) >= 4:
            in_data = True
            data_lines.append(s)
        elif in_data:
            break

    df = pd.read_csv(
        io.StringIO("\n".join(data_lines)),
        header=None, names=["Date","Mkt-RF","SMB","HML","RF"],
        dtype={"Date": str},
    )
    df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d", errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date")
    df = df.apply(pd.to_numeric, errors="coerce").dropna() / 100.0
    df.to_parquet(cache)
    print(f"  Sauvegardé → {cache}")
    return df.loc[start:end]
