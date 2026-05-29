"""
QuantForge — Module 1 : Tests unitaires (Pricing Engine)
=========================================================
Couvre :
  - Black-Scholes call / put pricing
  - Put-call parity
  - Greeks (Delta, Gamma, Vega, Theta, Rho)
  - Implied volatility (round-trip)
  - Monte Carlo European call / put convergence
  - Monte Carlo antithetic variance reduction
  - Monte Carlo barrier option

Lancement :
    cd 01_pricing_engine
    python -m pytest tests/ -v
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pytest
from scipy.stats import norm

from black_scholes import (
    compute_d1_d2,
    bs_call_price,
    bs_put_price,
    bs_put_call_parity,
    bs_greeks,
    bs_implied_vol,
)
from monte_carlo import (
    mc_call_price,
    mc_put_price,
    mc_call_price_antithetic,
    mc_put_price_antithetic,
    mc_barrier_call_price,
)


# ─────────────────────────────────────────────────────────────
# Paramètres de référence ATM
# ─────────────────────────────────────────────────────────────
S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.20, 1.0
# BS call ATM : ≈ 10.4506
BS_CALL_REF = bs_call_price(S0, K, r, sigma, T)
BS_PUT_REF  = bs_put_price(S0, K, r, sigma, T)


# ─────────────────────────────────────────────────────────────
# Black-Scholes
# ─────────────────────────────────────────────────────────────

class TestBlackScholesCall:

    def test_atm_price_known_value(self):
        """Valeur de référence ATM ≈ 10.4506."""
        price = bs_call_price(100, 100, 0.05, 0.20, 1.0)
        assert price == pytest.approx(10.4506, abs=1e-3)

    def test_otm_price_lower_than_atm(self):
        """Option OTM vaut moins qu'ATM."""
        atm = bs_call_price(100, 100, 0.05, 0.20, 1.0)
        otm = bs_call_price(100, 110, 0.05, 0.20, 1.0)
        assert otm < atm

    def test_itm_price_higher_than_atm(self):
        itm = bs_call_price(100, 90, 0.05, 0.20, 1.0)
        atm = bs_call_price(100, 100, 0.05, 0.20, 1.0)
        assert itm > atm

    def test_call_non_negative(self):
        for K_val in [80, 100, 120]:
            assert bs_call_price(100, K_val, 0.05, 0.20, 1.0) >= 0

    def test_intrinsic_value_lower_bound(self):
        """C ≥ max(S0 - K·e^{-rT}, 0)  (absence d'arbitrage)."""
        C   = bs_call_price(S0, K, r, sigma, T)
        lb  = max(S0 - K * np.exp(-r * T), 0)
        assert C >= lb - 1e-10

    def test_increasing_in_sigma(self):
        """Prix croissant en volatilité."""
        prices = [bs_call_price(S0, K, r, v, T) for v in [0.10, 0.20, 0.30, 0.40]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_increasing_in_S0(self):
        prices = [bs_call_price(s, K, r, sigma, T) for s in [80, 90, 100, 110, 120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_decreasing_in_K(self):
        prices = [bs_call_price(S0, k, r, sigma, T) for k in [80, 90, 100, 110, 120]]
        assert all(prices[i] > prices[i+1] for i in range(len(prices)-1))

    def test_approaches_intrinsic_deep_itm(self):
        """Deep ITM → C ≈ S0 - K·e^{-rT}."""
        C  = bs_call_price(200, 100, 0.05, 0.20, 1.0)
        lb = 200 - 100 * np.exp(-0.05)
        assert abs(C - lb) < 1.0


class TestBlackScholesPut:

    def test_atm_put_positive(self):
        assert bs_put_price(S0, K, r, sigma, T) > 0

    def test_put_non_negative(self):
        for K_val in [80, 100, 120]:
            assert bs_put_price(100, K_val, 0.05, 0.20, 1.0) >= 0

    def test_put_increasing_in_K(self):
        prices = [bs_put_price(S0, k, r, sigma, T) for k in [80, 90, 100, 110, 120]]
        assert all(prices[i] < prices[i+1] for i in range(len(prices)-1))

    def test_otm_put_less_than_atm(self):
        atm = bs_put_price(100, 100, 0.05, 0.20, 1.0)
        otm = bs_put_price(100,  90, 0.05, 0.20, 1.0)   # S > K → OTM put
        assert otm < atm


class TestPutCallParity:

    def test_parity_error_negligible(self):
        """C - P = S0 - K·e^{-rT}  (tolérance 1e-10)."""
        res = bs_put_call_parity(S0, K, r, sigma, T)
        assert res["error"] < 1e-10

    def test_parity_otm_call(self):
        res = bs_put_call_parity(100, 110, 0.05, 0.20, 1.0)
        assert res["error"] < 1e-10

    def test_parity_itm_call(self):
        res = bs_put_call_parity(100, 90, 0.05, 0.20, 1.0)
        assert res["error"] < 1e-10


class TestGreeks:

    @pytest.fixture(autouse=True)
    def greeks(self):
        self.g = bs_greeks(S0, K, r, sigma, T)

    def test_delta_call_in_01(self):
        assert 0.0 <= self.g["delta_call"] <= 1.0

    def test_delta_put_in_minus1_0(self):
        assert -1.0 <= self.g["delta_put"] <= 0.0

    def test_delta_put_call_relation(self):
        """delta_put = delta_call - 1."""
        assert self.g["delta_put"] == pytest.approx(self.g["delta_call"] - 1.0, abs=1e-10)

    def test_delta_call_atm_approx_half(self):
        """ATM delta ≈ 0.5 (légèrement supérieur à cause du drift)."""
        assert 0.50 < self.g["delta_call"] < 0.70

    def test_gamma_positive(self):
        assert self.g["gamma"] > 0

    def test_vega_positive(self):
        assert self.g["vega"] > 0

    def test_theta_call_negative(self):
        """Theta call négatif (time decay)."""
        assert self.g["theta_call"] < 0

    def test_theta_put_negative(self):
        assert self.g["theta_put"] < 0

    def test_rho_call_positive(self):
        """Call : taux élevé → valeur plus élevée."""
        assert self.g["rho_call"] > 0

    def test_rho_put_negative(self):
        """Put : taux élevé → valeur moins élevée."""
        assert self.g["rho_put"] < 0

    def test_delta_call_exact_formula(self):
        """delta_call = N(d1)."""
        d1, _ = compute_d1_d2(S0, K, r, sigma, T)
        assert self.g["delta_call"] == pytest.approx(norm.cdf(d1), abs=1e-10)

    def test_gamma_exact_formula(self):
        """gamma = φ(d1) / (S0 σ √T)."""
        d1, _ = compute_d1_d2(S0, K, r, sigma, T)
        expected = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))
        assert self.g["gamma"] == pytest.approx(expected, abs=1e-10)

    def test_delta_call_deep_itm(self):
        g = bs_greeks(200, 100, 0.05, 0.20, 1.0)
        assert g["delta_call"] > 0.99

    def test_delta_call_deep_otm(self):
        g = bs_greeks(50, 100, 0.05, 0.20, 1.0)
        assert g["delta_call"] < 0.01


class TestImpliedVolatility:

    def test_round_trip_call(self):
        """IV(BS(σ)) ≈ σ."""
        sigma_test = 0.25
        price = bs_call_price(S0, K, r, sigma_test, T)
        iv    = bs_implied_vol(price, S0, K, r, T, "call")
        assert iv == pytest.approx(sigma_test, abs=1e-5)

    def test_round_trip_put(self):
        sigma_test = 0.18
        price = bs_put_price(S0, K, r, sigma_test, T)
        iv    = bs_implied_vol(price, S0, K, r, T, "put")
        assert iv == pytest.approx(sigma_test, abs=1e-5)

    def test_round_trip_otm_call(self):
        sigma_test = 0.30
        price = bs_call_price(100, 110, r, sigma_test, T)
        iv    = bs_implied_vol(price, 100, 110, r, T, "call")
        assert iv == pytest.approx(sigma_test, abs=1e-5)

    def test_below_intrinsic_returns_nan(self):
        """Prix < valeur intrinsèque → pas de solution."""
        iv = bs_implied_vol(-1.0, S0, K, r, T, "call")
        assert np.isnan(iv)


# ─────────────────────────────────────────────────────────────
# Monte Carlo
# ─────────────────────────────────────────────────────────────

# Graine fixe pour tests reproductibles
RNG_SEED = 42


class TestMCCall:

    def test_price_close_to_bs(self):
        """MC call ≈ BS call  (±0.15 avec 100 000 chemins)."""
        np.random.seed(RNG_SEED)
        mc = mc_call_price(S0, K, r, sigma, T, n_paths=100_000)
        assert mc == pytest.approx(BS_CALL_REF, abs=0.15)

    def test_price_positive(self):
        np.random.seed(RNG_SEED)
        assert mc_call_price(S0, K, r, sigma, T) > 0

    def test_otm_cheaper_than_atm(self):
        np.random.seed(RNG_SEED)
        atm = mc_call_price(100, 100, 0.05, 0.20, 1.0, n_paths=50_000)
        np.random.seed(RNG_SEED)
        otm = mc_call_price(100, 110, 0.05, 0.20, 1.0, n_paths=50_000)
        assert otm < atm + 0.5   # léger slack pour variance MC

    def test_deep_otm_near_zero(self):
        np.random.seed(RNG_SEED)
        price = mc_call_price(100, 200, 0.05, 0.20, 1.0, n_paths=100_000)
        assert price < 0.05


class TestMCPut:

    def test_price_close_to_bs(self):
        np.random.seed(RNG_SEED)
        mc = mc_put_price(S0, K, r, sigma, T, n_paths=100_000)
        assert mc == pytest.approx(BS_PUT_REF, abs=0.15)

    def test_price_positive(self):
        np.random.seed(RNG_SEED)
        assert mc_put_price(S0, K, r, sigma, T) > 0

    def test_put_call_parity_mc(self):
        """C_MC - P_MC ≈ S0 - K·e^{-rT}."""
        np.random.seed(RNG_SEED)
        C = mc_call_price(S0, K, r, sigma, T, n_paths=200_000)
        np.random.seed(RNG_SEED + 1)
        P = mc_put_price(S0, K, r, sigma, T, n_paths=200_000)
        expected = S0 - K * np.exp(-r * T)
        assert (C - P) == pytest.approx(expected, abs=0.20)


class TestMCAntithetic:

    def test_antithetic_call_close_to_bs(self):
        np.random.seed(RNG_SEED)
        mc = mc_call_price_antithetic(S0, K, r, sigma, T, n_paths=50_000)
        assert mc == pytest.approx(BS_CALL_REF, abs=0.12)

    def test_antithetic_put_close_to_bs(self):
        np.random.seed(RNG_SEED)
        mc = mc_put_price_antithetic(S0, K, r, sigma, T, n_paths=50_000)
        assert mc == pytest.approx(BS_PUT_REF, abs=0.12)

    def test_antithetic_reduces_variance(self):
        """
        La variance de l'estimateur antithétique doit être inférieure
        à celle du MC naïf pour le même nombre de chemins.
        """
        n_trials, n_paths = 200, 10_000
        prices_naive = []
        prices_anti  = []
        for seed in range(n_trials):
            np.random.seed(seed)
            prices_naive.append(mc_call_price(S0, K, r, sigma, T, n_paths))
            np.random.seed(seed)
            prices_anti.append(mc_call_price_antithetic(S0, K, r, sigma, T, n_paths))
        assert np.var(prices_anti) < np.var(prices_naive)


class TestMCBarrier:

    def test_barrier_cheaper_than_vanilla(self):
        """Up-and-out barrier call ≤ vanilla call."""
        np.random.seed(RNG_SEED)
        barrier = mc_barrier_call_price(100, 100, 120, 0.05, 0.20, 1.0)
        np.random.seed(RNG_SEED)
        vanilla = mc_call_price(100, 100, 0.05, 0.20, 1.0)
        assert barrier < vanilla + 0.5

    def test_high_barrier_approaches_vanilla(self):
        """Barrière très haute → option presque jamais knockée → ≈ vanille."""
        np.random.seed(RNG_SEED)
        barrier = mc_barrier_call_price(100, 100, 1e6, 0.05, 0.20, 1.0,
                                         n_paths=100_000)
        np.random.seed(RNG_SEED)
        vanilla = mc_call_price(100, 100, 0.05, 0.20, 1.0, n_paths=100_000)
        assert abs(barrier - vanilla) < 0.20

    def test_barrier_at_spot_is_zero(self):
        """Barrière = spot → option immédiatement knockée → prix ≈ 0."""
        np.random.seed(RNG_SEED)
        price = mc_barrier_call_price(100, 100, 100, 0.05, 0.20, 1.0)
        assert price == pytest.approx(0.0, abs=0.01)

    def test_barrier_positive(self):
        np.random.seed(RNG_SEED)
        price = mc_barrier_call_price(100, 100, 130, 0.05, 0.20, 1.0)
        assert price >= 0

    def test_lower_barrier_cheaper(self):
        """Barrière plus basse → plus de knock-outs → prix plus faible."""
        np.random.seed(RNG_SEED)
        low  = mc_barrier_call_price(100, 100, 115, 0.05, 0.20, 1.0)
        np.random.seed(RNG_SEED)
        high = mc_barrier_call_price(100, 100, 150, 0.05, 0.20, 1.0)
        assert low <= high + 0.5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
