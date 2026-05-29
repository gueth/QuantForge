"""
Monte Carlo Option Pricing Engine
==================================

This module implements:
- Monte Carlo pricing of European call and put options
- Antithetic variates variance-reduction for European options
- Monte Carlo pricing of path-dependent exotic options (up-and-out barrier)

All simulations use risk-neutral Black-Scholes dynamics.
"""
import numpy as np


# ============================================================
# 1. EUROPEAN CALL OPTION PRICING (MONTE CARLO)
# ============================================================

def mc_call_price(S0, K, r, sigma, T, n_paths=100_000):
    """
    Price a European call option using Monte Carlo simulation.

    Parameters
    ----------
    S0      : float  Initial asset price
    K       : float  Strike price
    r       : float  Risk-free interest rate
    sigma   : float  Volatility
    T       : float  Time to maturity
    n_paths : int    Number of simulated paths

    Returns
    -------
    float  Estimated option price
    """
    Z  = np.random.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.exp(-r * T) * np.mean(np.maximum(ST - K, 0.0))


# ============================================================
# 2. EUROPEAN PUT OPTION PRICING (MONTE CARLO)
# ============================================================

def mc_put_price(S0, K, r, sigma, T, n_paths=100_000):
    """
    Price a European put option using Monte Carlo simulation.

    Parameters
    ----------
    S0, K, r, sigma, T, n_paths : same as mc_call_price

    Returns
    -------
    float  Estimated put option price
    """
    Z  = np.random.standard_normal(n_paths)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    return np.exp(-r * T) * np.mean(np.maximum(K - ST, 0.0))


# ============================================================
# 3. ANTITHETIC VARIATES — VARIANCE REDUCTION
# ============================================================

def mc_call_price_antithetic(S0, K, r, sigma, T, n_paths=100_000):
    """
    European call price with antithetic variates variance reduction.

    For each random draw Z, also evaluates the path with -Z.
    This halves variance for the same number of function evaluations.

    Parameters
    ----------
    n_paths : int  Number of antithetic pairs (total 2 × n_paths simulations)

    Returns
    -------
    float  Estimated call price (lower variance than mc_call_price)
    """
    Z   = np.random.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    vol   = sigma * np.sqrt(T)

    ST_pos = S0 * np.exp(drift + vol *  Z)
    ST_neg = S0 * np.exp(drift + vol * -Z)

    payoff = 0.5 * (np.maximum(ST_pos - K, 0.0) + np.maximum(ST_neg - K, 0.0))
    return np.exp(-r * T) * np.mean(payoff)


def mc_put_price_antithetic(S0, K, r, sigma, T, n_paths=100_000):
    """
    European put price with antithetic variates variance reduction.
    """
    Z   = np.random.standard_normal(n_paths)
    drift = (r - 0.5 * sigma**2) * T
    vol   = sigma * np.sqrt(T)

    ST_pos = S0 * np.exp(drift + vol *  Z)
    ST_neg = S0 * np.exp(drift + vol * -Z)

    payoff = 0.5 * (np.maximum(K - ST_pos, 0.0) + np.maximum(K - ST_neg, 0.0))
    return np.exp(-r * T) * np.mean(payoff)


# ============================================================
# 4. UP-AND-OUT BARRIER CALL OPTION (MONTE CARLO)
# ============================================================

def mc_barrier_call_price(S0, K, B, r, sigma, T,
                           n_paths=100_000, n_steps=252):
    """
    Monte Carlo pricing of an up-and-out knock-out barrier call option.

    The option is worth zero if the asset price crosses B from below
    at any monitoring point during [0, T].

    Parameters
    ----------
    S0      : float  Initial asset price (must satisfy S0 < B)
    K       : float  Strike price
    B       : float  Up-and-out barrier level
    r       : float  Risk-free interest rate
    sigma   : float  Volatility
    T       : float  Time to maturity
    n_paths : int    Number of simulated paths
    n_steps : int    Monitoring frequency (252 = daily for 1-year option)

    Returns
    -------
    float  Estimated barrier call price
    """
    dt = T / n_steps

    # Simulate path increments
    Z          = np.random.standard_normal((n_paths, n_steps))
    increments = np.exp((r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z)

    # Reconstruct full price paths: shape (n_paths, n_steps + 1)
    paths         = np.empty((n_paths, n_steps + 1))
    paths[:, 0]   = S0
    paths[:, 1:]  = S0 * np.cumprod(increments, axis=1)

    # Knock-out: any monitored price >= barrier
    knocked_out = np.any(paths >= B, axis=1)

    # Terminal payoff, zeroed for knocked-out paths
    ST      = paths[:, -1]
    payoffs = np.where(knocked_out, 0.0, np.maximum(ST - K, 0.0))

    return np.exp(-r * T) * np.mean(payoffs)
