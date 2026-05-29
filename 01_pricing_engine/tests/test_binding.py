"""
QuantForge — Module 1: C++ Extension Binding Tests
====================================================
Vérifie que l'extension pybind11 `mc_pricer` (compilée via setup.py) produit
des résultats cohérents avec le pricer Python.

Ces tests sont skippés automatiquement si l'extension n'est pas compilée,
afin de ne pas bloquer la CI sur les environnements sans compilateur C++.

Build:
    cd 01_pricing_engine
    python setup.py build_ext --inplace

Run:
    pytest 01_pricing_engine/tests/test_binding.py -v
"""
import platform
import sys
import sysconfig
import os

import pytest

# Windows : assure que les DLL Python sont trouvables lors du chargement de l'extension
if platform.system() == "Windows":
    _py_data = sysconfig.get_path("data")
    if _py_data:
        os.add_dll_directory(_py_data)
        _libs = os.path.join(_py_data, "libs")
        if os.path.isdir(_libs):
            os.add_dll_directory(_libs)

mc_pricer = pytest.importorskip(
    "mc_pricer",
    reason="Extension C++ mc_pricer non compilée — lancez : python setup.py build_ext --inplace",
)

# ── Référence Python ────────────────────────────────────────────────────────
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from monte_carlo import mc_call_price as _py_call_price


S0, K, r, sigma, T = 100.0, 100.0, 0.05, 0.2, 1.0
N_PATHS = 500_000
TOL_PRICE = 0.10   # tolérance MC (±10 cts sur ATM ~10.45)
TOL_SPEED = 1000   # ms max par appel


def test_mc_call_price_atm():
    """Le prix C++ est proche du prix Python sur ATM."""
    price_cpp = mc_pricer.mc_call_price(S0, K, r, sigma, T, N_PATHS)
    price_py  = _py_call_price(S0, K, r, sigma, T, n_paths=N_PATHS)
    assert abs(price_cpp - price_py) < TOL_PRICE, (
        f"C++={price_cpp:.4f}  Python={price_py:.4f}  diff={abs(price_cpp - price_py):.4f}"
    )


def test_mc_call_price_positive():
    """Le prix d'un call ne peut pas être négatif."""
    price = mc_pricer.mc_call_price(S0, K, r, sigma, T, 100_000)
    assert price >= 0.0


def test_mc_call_price_otm_less_than_atm():
    """Un call OTM vaut moins qu'un call ATM."""
    atm = mc_pricer.mc_call_price(S0, K,       r, sigma, T, 100_000)
    otm = mc_pricer.mc_call_price(S0, K * 1.2, r, sigma, T, 100_000)
    assert otm < atm


def test_mc_call_price_speed():
    """Un appel C++ doit tenir sous TOL_SPEED ms."""
    import time
    start = time.perf_counter()
    mc_pricer.mc_call_price(S0, K, r, sigma, T, 100_000)
    elapsed_ms = (time.perf_counter() - start) * 1000
    assert elapsed_ms < TOL_SPEED, f"Trop lent : {elapsed_ms:.1f} ms"
