import numpy as np
from scipy.stats import norm

"""
Black-Scholes Option Pricing Model
===================================

This module implements:
- Closed-form pricing of European call options
- Analytical computation of option sensitivities (Greeks)

The model is based on the Black-Scholes framework, which assumes
log-normal dynamics for the underlying asset price.
"""

# ============================================================
# 1. COMPUTE d1 AND d2
# ============================================================

def compute_d1_d2(S0, K, r, sigma, T):
    """
    Compute Black-Scholes intermediate variables d1 and d2.

    d1 measures:
    - risk-adjusted moneyness
    - sensitivity of option value to underlying asset

    d2 is:
    - risk-adjusted probability of exercise
    """

    sigma_sqrt_T = sigma * np.sqrt(T)

    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T

    return d1, d2


# ============================================================
# 2. EUROPEAN CALL OPTION PRICING (BLACK-SCHOLES)
# ============================================================

def bs_call_price(S0, K, r, sigma, T):
    """
    Closed-form Black-Scholes price of a European call option.

    Parameters
    ----------
    S0 : float
        Current asset price
    K : float
        Strike price
    r : float
        Risk-free interest rate
    sigma : float
        Volatility
    T : float
        Time to maturity

    Returns
    -------
    float
        Call option price
    """

    d1, d2 = compute_d1_d2(S0, K, r, sigma, T)

    # Risk-neutral pricing formula
    C0 = (S0 * norm.cdf(d1)) - (K * np.exp(-r * T) * norm.cdf(d2))

    return C0


# ============================================================
# 3. OPTION GREEKS (SENSITIVITY ANALYSIS)
# ============================================================

def bs_greeks(S0, K, r, sigma, T):
    """
    Compute Black-Scholes Greeks:
    - Delta: sensitivity to underlying price
    - Gamma: curvature of delta
    - Vega: sensitivity to volatility
    """

    d1, _ = compute_d1_d2(S0, K, r, sigma, T)

    delta = norm.cdf(d1)

    gamma = norm.pdf(d1) / (S0 * sigma * np.sqrt(T))

    vega = S0 * norm.pdf(d1) * np.sqrt(T)

    return delta, gamma, vega