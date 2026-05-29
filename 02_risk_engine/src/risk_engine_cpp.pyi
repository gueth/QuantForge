"""
Type stub for the compiled C++ extension risk_engine_cpp.

This file tells Pylance / pyright exactly what the pybind11 module exports
so that static analysis, autocompletion, and type checking work correctly
even when the extension has not been compiled yet.

The stub mirrors the PyRiskEngine class defined in cpp/bindings.cpp.
"""
from typing import Any
import numpy as np

class RiskEngine:
    """
    High-performance risk engine implemented in C++ (pybind11).

    Parameters
    ----------
    returns : np.ndarray, shape (T, N)
        Matrix of daily log-returns.
    weights : np.ndarray, shape (N,)
        Portfolio weights. Must sum to 1.
    """

    def __init__(
        self,
        returns: np.ndarray,
        weights: np.ndarray,
    ) -> None: ...

    # -- Portfolio returns ---------------------------------------------------
    def portfolio_returns(self) -> np.ndarray: ...

    @property
    def n_obs(self) -> int: ...

    @property
    def n_assets(self) -> int: ...

    # -- VaR / CVaR ----------------------------------------------------------
    def var_historical(
        self,
        confidence: float = ...,
        horizon:    int   = ...,
    ) -> float: ...

    def var_parametric(
        self,
        confidence: float = ...,
        horizon:    int   = ...,
    ) -> float: ...

    def cvar_historical(
        self,
        confidence: float = ...,
        horizon:    int   = ...,
    ) -> float: ...

    def cvar_parametric(
        self,
        confidence: float = ...,
        horizon:    int   = ...,
    ) -> float: ...

    def full_report(
        self,
        confidence: float = ...,
        horizon:    int   = ...,
    ) -> dict[str, Any]: ...

    # -- PCA -----------------------------------------------------------------
    def pca(self, n_components: int = ...) -> dict[str, Any]: ...

    # -- Fama-French ---------------------------------------------------------
    def fama_french(
        self,
        factors: np.ndarray,
        rf:      np.ndarray,
        names:   list[str],
    ) -> list[dict[str, Any]]: ...

    # -- Rolling VaR ---------------------------------------------------------
    def rolling_var(
        self,
        confidence: float = ...,
        window:     int   = ...,
        parametric: bool  = ...,
    ) -> np.ndarray: ...
