/*
 * QuantForge — Module 3: pybind11 bindings
 * Exposes Kalman filter, rolling z-score, and performance metrics to Python.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "kalman.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace qf;

// ── numpy helpers ────────────────────────────────────────────

static std::vector<double> to_vec(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1)
        throw std::runtime_error("Expected 1-D array");
    auto ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0]);
}

static py::array_t<double> to_ndarray(const std::vector<double>& v) {
    auto out = py::array_t<double>(v.size());
    std::copy(v.begin(), v.end(), static_cast<double*>(out.request().ptr));
    return out;
}


// ── Module ──────────────────────────────────────────────────

PYBIND11_MODULE(alpha_cpp, m) {
    m.doc() = "QuantForge Module 3 — C++ Alpha Strategy Engine (Kalman, metrics)";

    // kalman_filter → dict
    m.def("kalman_filter",
        [](py::array_t<double> y, py::array_t<double> x,
           double delta, double Ve, double beta0, double P0) -> py::dict {
            auto res = kalman_filter(to_vec(y), to_vec(x), delta, Ve, beta0, P0);
            return py::dict(
                "betas"_a            = to_ndarray(res.betas),
                "spreads"_a          = to_ndarray(res.spreads),
                "innovations"_a      = to_ndarray(res.innovations),
                "innovations_var"_a  = to_ndarray(res.innovations_var)
            );
        },
        py::arg("y"), py::arg("x"),
        py::arg("delta") = 1e-5,
        py::arg("Ve")    = 1e-3,
        py::arg("beta0") = 0.0,
        py::arg("P0")    = 1.0,
        R"doc(
        Scalar Kalman filter for dynamic hedge ratio estimation.

        Parameters
        ----------
        y     : array (T,)  Log prices of asset A
        x     : array (T,)  Log prices of asset B
        delta : float       Process noise factor in (0, 1); small = slow drift
        Ve    : float       Observation noise variance
        beta0 : float       Initial hedge ratio
        P0    : float       Initial state variance

        Returns
        -------
        dict with keys: betas, spreads, innovations, innovations_var
        )doc"
    );

    // rolling_zscore
    m.def("rolling_zscore",
        [](py::array_t<double> series, int window) -> py::array_t<double> {
            return to_ndarray(rolling_zscore(to_vec(series), window));
        },
        py::arg("series"), py::arg("window"),
        "Rolling z-score of a time series."
    );

    // compute_metrics → dict
    m.def("compute_metrics",
        [](py::array_t<double> daily_pnl, double rfr, int trading_days) -> py::dict {
            auto m = compute_metrics(to_vec(daily_pnl), rfr, trading_days);
            return py::dict(
                "total_return"_a  = m.total_return,
                "annual_return"_a = m.annual_return,
                "annual_vol"_a    = m.annual_vol,
                "sharpe_ratio"_a  = m.sharpe_ratio,
                "sortino_ratio"_a = m.sortino_ratio,
                "calmar_ratio"_a  = m.calmar_ratio,
                "max_drawdown"_a  = m.max_drawdown,
                "hit_rate"_a      = m.hit_rate,
                "profit_factor"_a = m.profit_factor,
                "avg_win"_a       = m.avg_win,
                "avg_loss"_a      = m.avg_loss,
                "n_trades"_a      = m.n_trades,
                "n_obs"_a         = m.n_obs
            );
        },
        py::arg("daily_pnl"),
        py::arg("risk_free_rate") = 0.0,
        py::arg("trading_days")   = 252,
        "Compute performance metrics from daily P&L series."
    );
}
