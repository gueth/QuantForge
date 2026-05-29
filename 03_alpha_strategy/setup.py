"""
Build script for the QuantForge alpha strategy C++ extension.

Compile & install (from 03_alpha_strategy/):
    pip install pybind11
    python setup.py build_ext --inplace

The compiled module (alpha_cpp.pyd / .so) will be placed in
03_alpha_strategy/src/ so that signals.py can import it directly.
"""
import sys
from pathlib import Path
from setuptools import setup, Extension
import pybind11

HERE = Path(__file__).parent

if sys.platform == "win32":
    extra_compile = ["/O2", "/std:c++17"]
else:
    extra_compile = ["-O2", "-std=c++17"]

ext = Extension(
    "alpha_cpp",
    sources=[str(HERE / "cpp" / "kalman_bindings.cpp")],
    include_dirs=[
        pybind11.get_include(),
        str(HERE / "include"),
    ],
    language="c++",
    extra_compile_args=extra_compile,
)

setup(
    name="alpha_cpp",
    version="1.0.0",
    description="QuantForge Module 3 — C++ Alpha Engine (Kalman filter, metrics)",
    ext_modules=[ext],
    options={"build_ext": {"build_lib": str(HERE / "src")}},
)
