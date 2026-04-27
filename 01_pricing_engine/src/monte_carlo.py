import numpy as np

"""
Monte Carlo Option Pricing Engine
==================================

This module implements:
- Monte Carlo pricing of European options
- Monte Carlo pricing of path-dependent exotic options (e.g. barrier options)

The model is based on stochastic simulation of asset prices under the
risk-neutral Black-Scholes dynamics.
"""

# ============================================================
# 1. EUROPEAN CALL OPTION PRICING (MONTE_CARLO)
# ============================================================

def mc_call_price(S0, K, r, sigma, T, n_paths=100000):
    """
    Price a European call option using Monte Carlo simulation.

    Parameters
    ----------
    S0 : float
       Initial asset price
    K : float
       Strike price
    r : float
       Risk-free interest rate
    sigma : float
       Volatility
    T : float
       Time to maturity
    n_paths : int
       Number of simulated paths

    Returns
    -------
    float
       Estimated option price
    """

    #1. Step 1: simulate Brownian motion (W_T)
    WT = np.random.standard_normal(size=n_paths) * np.sqrt(T)

    # Step 2: compute terminal asset prices (S_T)
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * WT)

    # Step 3: compute payoffs
    payoffs = np.maximum(ST - K, 0)

    # Step 4: discount expected payoff
    price = np.exp(-r * T) * np.mean(payoffs)

    return price

# ============================================================
# 2. BARRIER OPTION PRICING (MONTE CARLO)
# ============================================================

def mc_barrier_call_price(S0, K, B, r, sigma, T, n_paths=100000, n_steps=252):
    """
    Monte Carlo pricing of a knock-out barrier call option.
    """

    dt = T / n_steps

    # 1. Simulation des trajectoires
    Z = np.random.standard_normal((n_paths, n_steps))

    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0

    increments = np.exp((r - 0.5 * sigma**2) * dt +sigma * np.sqrt(dt) * Z)

    paths[:, 1:] = S0 * np.cumprod(increments, axis=1)

    # 2. Condition de knock-out
    knock_out = np.any(paths >= B, axis=1)

    # 3. Payoff final
    ST = paths[:, -1]
    payoff = np.where(knock_out, 0, np.maximum(ST - K, 0))

    # 4. Actualisation
    price = np.exp(-r * T) * np.mean(payoff)

    return price

