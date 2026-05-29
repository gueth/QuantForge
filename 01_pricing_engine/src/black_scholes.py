"""
Black-Scholes Option Pricing Model
===================================

This module implements:
- Closed-form pricing of European call and put options
- Analytical computation of all option Greeks (Delta, Gamma, Vega, Theta, Rho)
- Implied volatility via bisection

The model is based on the Black-Scholes framework, which assumes
log-normal dynamics for the underlying asset price.
"""
import numpy as np
from scipy.stats import norm


# ============================================================
# 1. COMPUTE d1 AND d2
# ============================================================

def compute_d1_d2(S0, K, r, sigma, T):
    """
    Compute Black-Scholes intermediate variables d1 and d2.

    d1 : risk-adjusted moneyness / sensitivity of option value to underlying
    d2 : risk-adjusted probability of exercise under risk-neutral measure
    """
    sigma_sqrt_T = sigma * np.sqrt(T)
    d1 = (np.log(S0 / K) + (r + 0.5 * sigma**2) * T) / sigma_sqrt_T
    d2 = d1 - sigma_sqrt_T
    return d1, d2


# ============================================================
# 2. EUROPEAN OPTION PRICING
# ============================================================

def bs_call_price(S0, K, r, sigma, T):
    """
    Closed-form Black-Scholes price of a European call option.

    Parameters
    ----------
    S0    : float  Current asset price
    K     : float  Strike price
    r     : float  Risk-free interest rate (continuous compounding)
    sigma : float  Annualised volatility
    T     : float  Time to maturity (in years)

    Returns
    -------
    float  Call option price
    """
    d1, d2 = compute_d1_d2(S0, K, r, sigma, T)
    return S0 * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def bs_put_price(S0, K, r, sigma, T):
    """
    Closed-form Black-Scholes price of a European put option.

    Uses put-call parity: P = C - S0 + K·e^{-rT}
    """
    d1, d2 = compute_d1_d2(S0, K, r, sigma, T)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S0 * norm.cdf(-d1)


# ============================================================
# 3. PUT-CALL PARITY VERIFICATION
# ============================================================

def bs_put_call_parity(S0, K, r, sigma, T):
    """
    Verify put-call parity: C - P = S0 - K·e^{-rT}

    Returns
    -------
    dict with 'call', 'put', 'parity_lhs', 'parity_rhs', 'error'
    """
    C = bs_call_price(S0, K, r, sigma, T)
    P = bs_put_price(S0, K, r, sigma, T)
    lhs = C - P
    rhs = S0 - K * np.exp(-r * T)
    return {"call": C, "put": P, "parity_lhs": lhs, "parity_rhs": rhs,
            "error": abs(lhs - rhs)}


# ============================================================
# 4. OPTION GREEKS (COMPLETE SET)
# ============================================================

def bs_greeks(S0, K, r, sigma, T):
    """
    Compute all Black-Scholes Greeks for call and put options.

    Parameters
    ----------
    S0, K, r, sigma, T : same as bs_call_price

    Returns
    -------
    dict
        delta_call  : dC/dS
        delta_put   : dP/dS  (= delta_call - 1)
        gamma       : d²C/dS²  (identical for call and put)
        vega        : dC/d(sigma) per 1 % vol move  (identical for call and put)
        theta_call  : dC/dt per calendar day
        theta_put   : dP/dt per calendar day
        rho_call    : dC/dr per 1 % rate move
        rho_put     : dP/dr per 1 % rate move
    """
    d1, d2 = compute_d1_d2(S0, K, r, sigma, T)
    sqrt_T = np.sqrt(T)
    df     = np.exp(-r * T)      # discount factor

    nd1  = norm.pdf(d1)          # φ(d1)
    Nd1  = norm.cdf(d1)          # Φ(d1)
    Nd2  = norm.cdf(d2)          # Φ(d2)
    Nnd2 = norm.cdf(-d2)         # Φ(-d2)

    delta_call = Nd1
    delta_put  = Nd1 - 1.0

    gamma = nd1 / (S0 * sigma * sqrt_T)

    # vega per 1 % point move in implied vol
    vega = S0 * nd1 * sqrt_T / 100.0

    # theta per calendar day (divided by 365)
    theta_call = (-(S0 * nd1 * sigma) / (2.0 * sqrt_T)
                  - r * K * df * Nd2) / 365.0
    theta_put  = (-(S0 * nd1 * sigma) / (2.0 * sqrt_T)
                  + r * K * df * Nnd2) / 365.0

    # rho per 1 % point move in the risk-free rate
    rho_call =  K * T * df * Nd2  / 100.0
    rho_put  = -K * T * df * Nnd2 / 100.0

    return {
        "delta_call": delta_call,
        "delta_put":  delta_put,
        "gamma":      gamma,
        "vega":       vega,
        "theta_call": theta_call,
        "theta_put":  theta_put,
        "rho_call":   rho_call,
        "rho_put":    rho_put,
    }


# ============================================================
# 5. IMPLIED VOLATILITY (BISECTION)
# ============================================================

def bs_implied_vol(market_price, S0, K, r, T,
                   option_type="call", tol=1e-6, max_iter=300):
    """
    Compute the implied volatility that matches a given market price.

    Parameters
    ----------
    market_price : float  Observed option price
    option_type  : "call" | "put"
    tol          : float  Convergence tolerance on the price
    max_iter     : int    Maximum bisection iterations

    Returns
    -------
    float  Implied volatility, or np.nan if no solution exists
    """
    pricer = bs_call_price if option_type == "call" else bs_put_price

    lo, hi = 1e-6, 10.0

    # Check that a solution exists in [lo, hi]
    if pricer(S0, K, r, lo, T) > market_price:
        return np.nan
    if pricer(S0, K, r, hi, T) < market_price:
        return np.nan

    for _ in range(max_iter):
        mid   = (lo + hi) / 2.0
        price = pricer(S0, K, r, mid, T)
        if abs(price - market_price) < tol:
            return mid
        if price < market_price:
            lo = mid
        else:
            hi = mid

    return (lo + hi) / 2.0
