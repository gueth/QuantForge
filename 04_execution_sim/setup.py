"""
Build script for the QuantForge execution simulator C++ extension.

Compile & install (from 04_execution_sim/):
    pip install pybind11
    python setup.py build_ext --inplace

The compiled module (exec_cpp.pyd / .so) will be placed in
04_execution_sim/src/ so that simulator.py can import it directly.
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
    "exec_cpp",
    sources=[str(HERE / "cpp" / "exec_bindings.cpp")],
    include_dirs=[
        pybind11.get_include(),
        str(HERE / "include"),
    ],
    language="c++",
    extra_compile_args=extra_compile,
)

setup(
    name="exec_cpp",
    version="1.0.0",
    description="QuantForge Module 4 — C++ Execution Simulator (Almgren-Chriss, Monte Carlo)",
    ext_modules=[ext],
    options={"build_ext": {"build_lib": str(HERE / "src")}},
)
