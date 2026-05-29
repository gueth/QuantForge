"""
QuantForge — Module 2 : Tests unitaires
=========================================
Couvre : Portfolio, VaR/CVaR (propriétés mathématiques),
         PCA (invariants spectraux), Fama-French (OLS),
         Stress test, Rolling VaR.

Lancement :
    cd 02_risk_engine
    python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import pytest

from portfolio import (
    Portfolio, PCAResult, FFResult,
    synthetic_portfolio, synthetic_ff_factors,
)


# ─────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def port():
    df = synthetic_portfolio(seed=0, n_assets=4, n_days=500)
    return Portfolio(df, name="Test")

@pytest.fixture(scope="module")
def port_large():
    df = synthetic_portfolio(seed=7, n_assets=8, n_days=1500)
    return Portfolio(df, name="Large")

@pytest.fixture(scope="module")
def factors(port):
    return synthetic_ff_factors(port.returns.index)


# ─────────────────────────────────────────────────────────────
# Portfolio
# ─────────────────────────────────────────────────────────────

class TestPortfolio:

    def test_equal_weight_default(self, port):
        n = port.N
        np.testing.assert_allclose(port.weights, np.ones(n) / n)

    def test_weights_sum_to_one(self, port):
        assert abs(port.weights.sum() - 1.0) < 1e-10

    def test_custom_weights_accepted(self):
        df = synthetic_portfolio(seed=1, n_assets=3, n_days=200)
        w  = np.array([0.5, 0.3, 0.2])
        p  = Portfolio(df, weights=w)
        np.testing.assert_allclose(p.weights, w)

    def test_invalid_weights_raise(self):
        df = synthetic_portfolio(seed=2, n_assets=3, n_days=200)
        with pytest.raises(ValueError, match="sommer à 1"):
            Portfolio(df, weights=np.array([0.5, 0.5, 0.5]))

    def test_portfolio_returns_length(self, port):
        assert len(port.portfolio_returns) == port.T

    def test_portfolio_returns_weighted_sum(self, port):
        """r_p = Σ wᵢ rᵢ  (vérification exacte C++ vs pandas)."""
        expected = port.returns.values @ port.weights
        np.testing.assert_allclose(port.portfolio_returns.values, expected, rtol=1e-6)

    def test_repr_contains_name(self, port):
        assert "Test" in repr(port)


# ─────────────────────────────────────────────────────────────
# VaR — propriétés mathématiques fondamentales
# ─────────────────────────────────────────────────────────────

class TestVaR:

    def test_var_positive(self, port):
        """VaR représente une perte → doit être > 0."""
        assert port.var_historical()  > 0
        assert port.var_parametric()  > 0
        assert port.cvar_historical() > 0
        assert port.cvar_parametric() > 0

    def test_cvar_geq_var_historical(self, port):
        """CVaR ≥ VaR par définition (espérance conditionnelle dans la queue)."""
        for conf in [0.90, 0.95, 0.99]:
            var  = port.var_historical(conf)
            cvar = port.cvar_historical(conf)
            assert cvar >= var - 1e-10, f"CVaR < VaR à {conf:.0%}"

    def test_cvar_geq_var_parametric(self, port):
        for conf in [0.90, 0.95, 0.99]:
            var  = port.var_parametric(conf)
            cvar = port.cvar_parametric(conf)
            assert cvar >= var - 1e-10, f"CVaR param < VaR param à {conf:.0%}"

    def test_var_monotone_in_confidence(self, port):
        """VaR(α) croissant en α."""
        v = [port.var_historical(c) for c in [0.90, 0.95, 0.99, 0.999]]
        assert all(v[i] <= v[i+1] + 1e-10 for i in range(len(v)-1))

    def test_horizon_sqrt_scaling(self, port):
        """Règle racine carrée du temps : VaR(10j) = VaR(1j) × √10."""
        v1  = port.var_historical(0.95, horizon=1)
        v10 = port.var_historical(0.95, horizon=10)
        np.testing.assert_allclose(v10, v1 * np.sqrt(10), rtol=1e-6)

    def test_parametric_formula_exact(self, port):
        """
        Vérification analytique : VaR_param = -(μ + z_α σ).
        z_{0.05} = Φ⁻¹(0.05) ≈ -1.6449
        """
        from scipy.stats import norm
        r       = port.portfolio_returns.values
        mu, sig = r.mean(), r.std(ddof=1)
        z       = norm.ppf(0.05)
        expected = -(mu + z * sig)
        computed = port.var_parametric(0.95)
        np.testing.assert_allclose(computed, expected, rtol=1e-5)

    def test_cvar_parametric_formula_exact(self, port):
        """CVaR_param = -(μ - σ φ(z_α) / (1-α))"""
        from scipy.stats import norm
        r       = port.portfolio_returns.values
        mu, sig = r.mean(), r.std(ddof=1)
        alpha   = 0.05
        z       = norm.ppf(alpha)
        expected = -(mu - sig * norm.pdf(z) / alpha)
        computed = port.cvar_parametric(0.95)
        np.testing.assert_allclose(computed, expected, rtol=1e-5)

    def test_full_report_fields(self, port):
        rep = port.full_report(0.95)
        assert rep.confidence    == 0.95
        assert rep.n_obs         == port.T
        assert rep.portfolio_vol > 0
        assert rep.var_historical == pytest.approx(port.var_historical(0.95), rel=1e-6)

    def test_95_vs_99_var_ordering(self, port):
        """99% VaR doit être supérieur au 95% VaR."""
        assert port.var_historical(0.99) > port.var_historical(0.95)
        assert port.var_parametric(0.99) > port.var_parametric(0.95)

    def test_large_portfolio_consistency(self, port_large):
        """VaR et CVaR cohérents sur un large portefeuille."""
        assert port_large.var_historical()  > 0
        assert port_large.cvar_historical() > port_large.var_historical()


# ─────────────────────────────────────────────────────────────
# PCA — invariants spectraux
# ─────────────────────────────────────────────────────────────

class TestPCA:

    def test_explained_variance_sums_to_one(self, port):
        pca = port.pca()
        np.testing.assert_allclose(
            pca.explained_variance_ratio.sum(), 1.0, atol=1e-6
        )

    def test_cumulative_variance_monotone(self, port):
        pca = port.pca()
        diffs = np.diff(pca.cumulative_variance)
        assert (diffs >= -1e-10).all(), "Variance cumulée non-monotone"

    def test_eigenvalues_descending(self, port):
        pca = port.pca()
        diffs = np.diff(pca.eigenvalues)
        assert (diffs <= 1e-10).all(), "Valeurs propres non-décroissantes"

    def test_eigenvalues_non_negative(self, port):
        """Covariance semi-définie positive → valeurs propres ≥ 0."""
        pca = port.pca()
        assert (pca.eigenvalues >= -1e-10).all()

    def test_factor_loadings_shape(self, port):
        n = port.N
        pca = port.pca()
        assert pca.factor_loadings.shape == (n, n)

    def test_factor_returns_shape(self, port):
        pca = port.pca()
        assert pca.factor_returns.shape == (port.T, port.N)

    def test_truncation(self, port):
        pca = port.pca(n_components=2)
        assert pca.factor_loadings.shape[1] == 2
        assert len(pca.eigenvalues) == 2
        assert pca.factor_returns.shape[1] == 2

    def test_eigenvectors_orthogonal(self, port):
        """
        Vecteurs propres d'une matrice symétrique sont orthogonaux :
        VᵀV = I  (jusqu'à la précision numérique de Jacobi)
        """
        import risk_engine_cpp as cpp
        raw = cpp.RiskEngine(
            port.returns.values.astype(np.float64), port.weights
        ).pca(port.N)
        V = raw["factor_loadings"]      # N × N
        prod = V.T @ V
        np.testing.assert_allclose(prod, np.eye(port.N), atol=1e-5)

    def test_factor_returns_zero_mean(self, port):
        """Les rendements factoriels sont construits sur données centrées → ~0 mean."""
        pca = port.pca()
        means = pca.factor_returns.mean()
        np.testing.assert_allclose(means.values, 0.0, atol=1e-8)

    def test_n_factors_for_threshold(self, port):
        pca = port.pca()
        k   = pca.n_factors_for(0.90)
        assert pca.cumulative_variance[k - 1] >= 0.90 - 1e-6

    def test_large_portfolio_pca(self, port_large):
        pca = port_large.pca()
        np.testing.assert_allclose(
            pca.explained_variance_ratio.sum(), 1.0, atol=1e-5
        )


# ─────────────────────────────────────────────────────────────
# Fama-French OLS
# ─────────────────────────────────────────────────────────────

class TestFamaFrench:

    def test_r2_in_unit_interval(self, port, factors):
        ff = port.fama_french(factors)
        r2s = np.array([r["r_squared"] for r in ff.raw])
        assert (r2s >= -0.01).all()
        assert (r2s <= 1.01).all()

    def test_residuals_zero_mean(self, port, factors):
        """OLS avec constante → résidus de moyenne nulle."""
        ff  = port.fama_french(factors)
        eps = ff.residuals()
        np.testing.assert_allclose(eps.mean().values, 0.0, atol=1e-8)

    def test_betas_shape(self, port, factors):
        ff = port.fama_french(factors)
        factor_cols = [c for c in factors.columns if c.upper() not in ("RF", "RFR")]
        # ff.betas est une propriété numpy (n_assets × n_factors)
        assert ff.betas.shape == (port.N, len(factor_cols))
        # Via summary
        summary = ff.summary()
        assert summary.shape[0] == port.N

    def test_systematic_idio_vol_positive(self, port, factors):
        ff = port.fama_french(factors)
        for r in ff.raw:
            assert r["systematic_vol"] >= 0
            assert r["idio_vol"]       >= 0

    def test_insufficient_data_raises(self, factors):
        short = synthetic_portfolio(seed=3, n_assets=3, n_days=30)
        p     = Portfolio(short)
        with pytest.raises(ValueError, match="dates communes"):
            p.fama_french(factors.iloc[:30])

    def test_summary_has_all_assets(self, port, factors):
        ff  = port.fama_french(factors)
        summ = ff.summary()
        assert list(summ.index) == port.assets


# ─────────────────────────────────────────────────────────────
# Stress Test
# ─────────────────────────────────────────────────────────────

class TestStressTest:

    def test_uniform_shock_equal_portfolio_shock(self, port):
        """Choc uniforme -20% sur tous les actifs → P&L = -20%."""
        scenarios = {"test": {a: -0.20 for a in port.assets}}
        res = port.stress_test(scenarios)
        np.testing.assert_allclose(res.loc["test", "P&L (%)"], -20.0, atol=0.01)

    def test_zero_shock(self, port):
        scenarios = {"plat": {a: 0.0 for a in port.assets}}
        res = port.stress_test(scenarios)
        assert res.loc["plat", "P&L (%)"] == pytest.approx(0.0, abs=1e-6)

    def test_unknown_asset_ignored(self, port):
        scenarios = {"ghost": {"TICKER_INEXISTANT": -1.0}}
        res = port.stress_test(scenarios)
        assert res.loc["ghost", "P&L (%)"] == pytest.approx(0.0, abs=1e-6)

    def test_bull_scenario_positive(self, port):
        scenarios = {"bull": {a: 0.30 for a in port.assets}}
        res = port.stress_test(scenarios)
        assert res.loc["bull", "P&L (%)"] > 0

    def test_multiple_scenarios(self, port):
        scenarios = {
            "krach": {a: -0.40 for a in port.assets},
            "bull":  {a: +0.25 for a in port.assets},
        }
        res = port.stress_test(scenarios)
        assert res.loc["krach", "P&L (%)"] < 0
        assert res.loc["bull",  "P&L (%)"] > 0


# ─────────────────────────────────────────────────────────────
# Rolling VaR
# ─────────────────────────────────────────────────────────────

class TestRollingVaR:

    def test_length(self, port):
        window = 100
        rv = port.rolling_var(window=window)
        assert len(rv) == port.T - window + 1

    def test_all_positive(self, port):
        rv = port.rolling_var(window=100)
        assert (rv.values > 0).all()

    def test_parametric_method(self, port):
        rv = port.rolling_var(window=100, parametric=True)
        assert (rv.values > 0).all()

    def test_index_aligned(self, port):
        window = 100
        rv = port.rolling_var(window=window)
        assert rv.index[0]  == port.returns.index[window - 1]
        assert rv.index[-1] == port.returns.index[-1]

    def test_window_too_large_raises(self, port):
        with pytest.raises(Exception):
            port.rolling_var(window=port.T + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
