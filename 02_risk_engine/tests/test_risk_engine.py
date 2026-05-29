"""
QuantForge - Module 2: Unit Tests
===================================
Covers: Portfolio, VaR/CVaR (mathematical invariants),
        PCA (spectral properties), Fama-French (OLS),
        Stress Testing, Rolling VaR.

Run:
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
    _HAS_CPP,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Portfolio properties
# ---------------------------------------------------------------------------

class TestPortfolio:

    def test_equal_weight_default(self, port):
        np.testing.assert_allclose(port.weights, np.ones(port.N) / port.N)

    def test_weights_sum_to_one(self, port):
        assert abs(port.weights.sum() - 1.0) < 1e-10

    def test_custom_weights_accepted(self):
        df = synthetic_portfolio(seed=1, n_assets=3, n_days=200)
        w  = np.array([0.5, 0.3, 0.2])
        p  = Portfolio(df, weights=w)
        np.testing.assert_allclose(p.weights, w)

    def test_invalid_weights_raise(self):
        df = synthetic_portfolio(seed=2, n_assets=3, n_days=200)
        with pytest.raises(ValueError, match="sum to 1"):
            Portfolio(df, weights=np.array([0.5, 0.5, 0.5]))

    def test_portfolio_returns_length(self, port):
        assert len(port.portfolio_returns) == port.T

    def test_portfolio_returns_weighted_sum(self, port):
        """Verify r_p = sum(w_i * r_i) exactly against numpy."""
        expected = port.returns.values @ port.weights
        np.testing.assert_allclose(
            port.portfolio_returns.values, expected, rtol=1e-6
        )

    def test_repr_contains_name(self, port):
        assert "Test" in repr(port)


# ---------------------------------------------------------------------------
# VaR — fundamental mathematical invariants
# ---------------------------------------------------------------------------

class TestVaR:

    def test_var_positive(self, port):
        """VaR represents a loss magnitude and must be strictly positive."""
        assert port.var_historical()  > 0
        assert port.var_parametric()  > 0
        assert port.cvar_historical() > 0
        assert port.cvar_parametric() > 0

    def test_cvar_geq_var_historical(self, port):
        """CVaR >= VaR by definition (conditional expectation in the tail)."""
        for conf in [0.90, 0.95, 0.99]:
            var  = port.var_historical(conf)
            cvar = port.cvar_historical(conf)
            assert cvar >= var - 1e-10, f"CVaR < VaR at {conf:.0%}"

    def test_cvar_geq_var_parametric(self, port):
        for conf in [0.90, 0.95, 0.99]:
            var  = port.var_parametric(conf)
            cvar = port.cvar_parametric(conf)
            assert cvar >= var - 1e-10, f"Parametric CVaR < VaR at {conf:.0%}"

    def test_var_monotone_in_confidence(self, port):
        """VaR(alpha) is non-decreasing in the confidence level."""
        v = [port.var_historical(c) for c in [0.90, 0.95, 0.99, 0.999]]
        assert all(v[i] <= v[i + 1] + 1e-10 for i in range(len(v) - 1))

    def test_horizon_sqrt_scaling(self, port):
        """Square-root-of-time rule: VaR(10d) = VaR(1d) * sqrt(10)."""
        v1  = port.var_historical(0.95, horizon=1)
        v10 = port.var_historical(0.95, horizon=10)
        np.testing.assert_allclose(v10, v1 * np.sqrt(10), rtol=1e-6)

    def test_parametric_formula_exact(self, port):
        """Verify analytical formula: VaR_param = -(mu + z_alpha * sigma)."""
        from scipy.stats import norm
        r        = port.portfolio_returns.values
        mu, sig  = r.mean(), r.std(ddof=1)
        z        = norm.ppf(0.05)
        expected = -(mu + z * sig)
        np.testing.assert_allclose(port.var_parametric(0.95), expected, rtol=1e-5)

    def test_cvar_parametric_formula_exact(self, port):
        """Verify: CVaR_param = -(mu - sigma * phi(z_alpha) / (1-alpha))."""
        from scipy.stats import norm
        r        = port.portfolio_returns.values
        mu, sig  = r.mean(), r.std(ddof=1)
        alpha    = 0.05
        z        = norm.ppf(alpha)
        expected = -(mu - sig * norm.pdf(z) / alpha)
        np.testing.assert_allclose(port.cvar_parametric(0.95), expected, rtol=1e-5)

    def test_full_report_fields(self, port):
        rep = port.full_report(0.95)
        assert rep.confidence     == 0.95
        assert rep.n_obs          == port.T
        assert rep.portfolio_vol  > 0
        assert rep.var_historical == pytest.approx(port.var_historical(0.95), rel=1e-6)

    def test_99_exceeds_95_var(self, port):
        """99% VaR must be larger than 95% VaR."""
        assert port.var_historical(0.99) > port.var_historical(0.95)
        assert port.var_parametric(0.99) > port.var_parametric(0.95)

    def test_large_portfolio_consistency(self, port_large):
        assert port_large.var_historical()  > 0
        assert port_large.cvar_historical() > port_large.var_historical()


# ---------------------------------------------------------------------------
# PCA — spectral invariants
# ---------------------------------------------------------------------------

class TestPCA:

    def test_explained_variance_sums_to_one(self, port):
        pca = port.pca()
        np.testing.assert_allclose(
            pca.explained_variance_ratio.sum(), 1.0, atol=1e-6
        )

    def test_cumulative_variance_monotone(self, port):
        pca   = port.pca()
        diffs = np.diff(pca.cumulative_variance)
        assert (diffs >= -1e-10).all(), "Cumulative variance is not monotone"

    def test_eigenvalues_descending(self, port):
        pca   = port.pca()
        diffs = np.diff(pca.eigenvalues)
        assert (diffs <= 1e-10).all(), "Eigenvalues are not in descending order"

    def test_eigenvalues_non_negative(self, port):
        """Covariance matrix is PSD, so all eigenvalues must be >= 0."""
        pca = port.pca()
        assert (pca.eigenvalues >= -1e-10).all()

    def test_factor_loadings_shape(self, port):
        pca = port.pca()
        assert pca.factor_loadings.shape == (port.N, port.N)

    def test_factor_returns_shape(self, port):
        pca = port.pca()
        assert pca.factor_returns.shape == (port.T, port.N)

    def test_truncation(self, port):
        pca = port.pca(n_components=2)
        assert pca.factor_loadings.shape[1] == 2
        assert len(pca.eigenvalues) == 2
        assert pca.factor_returns.shape[1] == 2

    @pytest.mark.skipif(not _HAS_CPP, reason="requires compiled C++ extension")
    def test_eigenvectors_orthogonal_cpp(self, port):
        """
        Eigenvectors of a symmetric matrix are orthogonal: V^T V = I.
        Tested only with the Jacobi C++ solver.
        """
        import risk_engine_cpp as cpp
        raw  = cpp.RiskEngine(
            port.returns.values.astype(np.float64), port.weights
        ).pca(port.N)
        V    = raw["factor_loadings"]      # shape (N, N)
        prod = V.T @ V
        np.testing.assert_allclose(prod, np.eye(port.N), atol=1e-5)

    def test_eigenvectors_orthogonal_python(self, port):
        """Same orthogonality check using the NumPy fallback engine."""
        from portfolio import _PyRiskEngine
        eng  = _PyRiskEngine(
            port.returns.values.astype(np.float64), port.weights
        )
        raw  = eng.pca(port.N)
        V    = raw["factor_loadings"]      # shape (N, N)
        prod = V.T @ V
        np.testing.assert_allclose(prod, np.eye(port.N), atol=1e-5)

    def test_factor_returns_zero_mean(self, port):
        """Factor returns are built from centred data, so mean ~ 0."""
        pca   = port.pca()
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


# ---------------------------------------------------------------------------
# Fama-French OLS
# ---------------------------------------------------------------------------

class TestFamaFrench:

    def test_r2_in_unit_interval(self, port, factors):
        ff  = port.fama_french(factors)
        r2s = np.array([r["r_squared"] for r in ff.raw])
        assert (r2s >= -0.01).all()
        assert (r2s <= 1.01).all()

    def test_residuals_zero_mean(self, port, factors):
        """OLS with intercept: residuals must have zero mean."""
        ff  = port.fama_french(factors)
        eps = ff.residuals()
        np.testing.assert_allclose(eps.mean().values, 0.0, atol=1e-8)

    def test_betas_shape(self, port, factors):
        ff          = port.fama_french(factors)
        factor_cols = [c for c in factors.columns if c.upper() not in ("RF", "RFR")]
        assert ff.betas.shape == (port.N, len(factor_cols))
        assert ff.summary().shape[0] == port.N

    def test_systematic_idio_vol_positive(self, port, factors):
        ff = port.fama_french(factors)
        for r in ff.raw:
            assert r["systematic_vol"] >= 0
            assert r["idio_vol"]       >= 0

    def test_insufficient_data_raises(self, factors):
        short = synthetic_portfolio(seed=3, n_assets=3, n_days=30)
        p     = Portfolio(short)
        with pytest.raises(ValueError, match="common dates"):
            p.fama_french(factors.iloc[:30])

    def test_summary_has_all_assets(self, port, factors):
        ff   = port.fama_french(factors)
        summ = ff.summary()
        assert list(summ.index) == port.assets


# ---------------------------------------------------------------------------
# Stress Test
# ---------------------------------------------------------------------------

class TestStressTest:

    def test_uniform_shock_equal_portfolio_shock(self, port):
        """Uniform -20% shock on all assets -> portfolio P&L = -20%."""
        scenarios = {"test": {a: -0.20 for a in port.assets}}
        res       = port.stress_test(scenarios)
        np.testing.assert_allclose(res.loc["test", "P&L (%)"], -20.0, atol=0.01)

    def test_zero_shock_returns_zero(self, port):
        scenarios = {"flat": {a: 0.0 for a in port.assets}}
        res       = port.stress_test(scenarios)
        assert res.loc["flat", "P&L (%)"] == pytest.approx(0.0, abs=1e-6)

    def test_unknown_asset_ignored(self, port):
        scenarios = {"ghost": {"UNKNOWN_TICKER_XYZ": -1.0}}
        res       = port.stress_test(scenarios)
        assert res.loc["ghost", "P&L (%)"] == pytest.approx(0.0, abs=1e-6)

    def test_bull_scenario_positive(self, port):
        scenarios = {"bull": {a: 0.30 for a in port.assets}}
        res       = port.stress_test(scenarios)
        assert res.loc["bull", "P&L (%)"] > 0

    def test_multiple_scenarios(self, port):
        scenarios = {
            "crash": {a: -0.40 for a in port.assets},
            "rally": {a: +0.25 for a in port.assets},
        }
        res = port.stress_test(scenarios)
        assert res.loc["crash", "P&L (%)"] < 0
        assert res.loc["rally", "P&L (%)"] > 0


# ---------------------------------------------------------------------------
# Rolling VaR
# ---------------------------------------------------------------------------

class TestRollingVaR:

    def test_correct_length(self, port):
        window = 100
        rv     = port.rolling_var(window=window)
        assert len(rv) == port.T - window + 1

    def test_all_positive(self, port):
        rv = port.rolling_var(window=100)
        assert (rv.values > 0).all()

    def test_parametric_method(self, port):
        rv = port.rolling_var(window=100, parametric=True)
        assert (rv.values > 0).all()

    def test_index_aligned(self, port):
        window = 100
        rv     = port.rolling_var(window=window)
        assert rv.index[0]  == port.returns.index[window - 1]
        assert rv.index[-1] == port.returns.index[-1]

    def test_window_too_large_raises(self, port):
        with pytest.raises(Exception):
            port.rolling_var(window=port.T + 1)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
