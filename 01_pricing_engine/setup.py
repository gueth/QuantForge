from setuptools import setup, Extension
import pybind11

ext = Extension(
    "mc_pricer",
    sources=["cpp/mc_bindings.cpp"],
    include_dirs=[pybind11.get_include()],
    language="c++",
    extra_compile_args=["/O2"] if __import__("sys").platform == "win32" else ["-O2"],
)

setup(
    name="mc_pricer",
    ext_modules=[ext],
)