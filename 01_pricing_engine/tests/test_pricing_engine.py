import numpy as np
import time

"""
Pricing Engine Tests
====================

This script tests:
- Monte Carlo European call pricing
- Black-Scholes closed-form pricing
- Greeks computation
- Barrier option pricing
- Performance benchmarking
"""
# ============================================================
# IMPORTS
# ============================================================
from black_scholes import bs_call_price, bs_greeks
from monte_carlo import mc_call_price, mc_barrier_call_price

# ============================================================
# 1. MONTE CARLO TEST - EUROPEAN CALL
# ============================================================

np.random.seed(42)

print("\n==============================")
print("TEST: Monte Carlo Call Price")
print("==============================")

mc_price = mc_call_price(100, 100, 0.05, 0.2, 1)
print("MC Call Price:", mc_price)

"""
Expected result:
You will NOT get exactly 10.45 due to randomness.

But result should be close:
≈ 10.45 ± 0.1
"""


# ============================================================
# 2. BLACK-SCHOLES TEST
# ============================================================

print("\n==============================")
print("TEST: Black-Scholes Call Price")
print("==============================")

test_cases = [
    (100, 100, 0.05, 0.2, 1.0),
    (100, 110, 0.05, 0.2, 1.0),
    (100, 90, 0.05, 0.2, 1.0)
]

for S0, K, r, sigma, T in test_cases:
    price = bs_call_price(S0, K, r, sigma, T)
    print(f"S0={S0}, K={K} -> Call Price: {price}")

"""
Expected results:
10.4506
6.0401
16.6994
"""


# ============================================================
# 3. GREEKS TEST
# ============================================================

print("\n==============================")
print("TEST: Black-Scholes Greeks")
print("==============================")

delta, gamma, vega = bs_greeks(100, 100, 0.05, 0.2, 1)

print(f"Delta: {delta}")
print(f"Gamma: {gamma}")
print(f"Vega:  {vega}")

"""
Expected:
Delta ≈ 0.6368
"""


# ============================================================
# 4. BARRIER OPTION TEST
# ============================================================

print("\n==============================")
print("TEST: Barrier Call Price")
print("==============================")

barrier_price = mc_barrier_call_price(100, 100, 120, 0.05, 0.2, 1)
print("Barrier Call (B=120):", barrier_price)

"""
Expected:
≈ 6.5 (lower than vanilla call ~10.45)
"""

print("Barrier Call (B=150):",
      mc_barrier_call_price(100, 100, 150, 0.05, 0.2, 1))

print("Vanilla Call (MC):",
      mc_call_price(100, 100, 0.05, 0.2, 1))


# ============================================================
# 5. PERFORMANCE BENCHMARK
# ============================================================

print("\n==============================")
print("BENCHMARK: Barrier MC Speed")
print("==============================")

np.random.seed(42)

start = time.time()

for _ in range(10):
    mc_barrier_call_price(100, 100, 120, 0.05, 0.2, 1)

elapsed = time.time() - start

print(f"Average execution time: {elapsed / 10 * 1000:.2f} ms per call")
