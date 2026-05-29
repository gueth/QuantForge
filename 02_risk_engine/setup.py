"""
Build script for the QuantForge risk engine C++ extension.

Compile & install (from 02_risk_engine/):
    pip install pybind11
    python setup.py build_ext --inplace

The compiled module (risk_engine_cpp.pyd / .so) will be placed in
02_risk_engine/python/ so that portfolio.py can import it directly.
"""
import sys
from pathlib import Path
from setuptools import setup, Extension
import pybind11

HERE = Path(__file__).parent

# C++17 required for structured bindings (auto [a, b] = ...)
if sys.platform == "win32":
    extra_compile = ["/O2", "/std:c++17"]
else:
    extra_compile = ["-O2", "-std=c++17"]

ext = Extension(
    "risk_engine_cpp",
    sources=[str(HERE / "bindings" / "bindings.cpp")],
    include_dirs=[
        pybind11.get_include(),
        str(HERE / "include"),
    ],
    language="c++",
    extra_compile_args=extra_compile,
)

setup(
    name="risk_engine_cpp",
    version="1.0.0",
    description="QuantForge Module 2 — C++ Risk Engine (VaR, CVaR, PCA, Fama-French)",
    ext_modules=[ext],
    options={"build_ext": {"build_lib": str(HERE / "python")}},
)
