/*
 * QuantForge — Module 4: pybind11 bindings
 * Exposes Almgren-Chriss optimal execution and Monte Carlo simulation to Python.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "almgren_chriss.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace qf;

static std::vector<double> to_vec(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) throw std::runtime_error("Expected 1-D array");
    auto ptr = static_cast<double*>(buf.ptr);
    return std::vector<double>(ptr, ptr + buf.shape[0]);
}

static py::array_t<double> to_ndarray(const std::vector<double>& v) {
    auto out = py::array_t<double>(v.size());
    std::copy(v.begin(), v.end(), static_cast<double*>(out.request().ptr));
    return out;
}

static py::dict schedule_to_dict(const Schedule& s) {
    return py::dict(
        "name"_a             = s.name,
        "trades"_a           = to_ndarray(s.trades),
        "total_shares"_a     = s.total_shares,
        "expected_cost"_a    = s.expected_cost,
        "expected_variance"_a= s.expected_variance
    );
}


PYBIND11_MODULE(exec_cpp, m) {
    m.doc() = "QuantForge Module 4 — C++ Execution Simulator (Almgren-Chriss, MC)";

    // ac_optimal_trajectory
    m.def("ac_optimal_trajectory",
        [](double X, double sigma, double eta, double gamma_,
           double lambda, int N, double tau) -> py::dict {
            ACParams p{sigma, eta, gamma_, lambda, N, tau};
            return schedule_to_dict(ac_optimal_trajectory(X, p));
        },
        py::arg("X"), py::arg("sigma"), py::arg("eta"), py::arg("gamma"),
        py::arg("lambda_"), py::arg("N"), py::arg("tau"),
        R"doc(
        Compute the Almgren-Chriss optimal execution trajectory.

        Parameters
        ----------
        X      : float  Total shares to liquidate
        sigma  : float  Daily price volatility (fraction)
        eta    : float  Temporary impact coefficient
        gamma  : float  Permanent impact coefficient
        lambda_: float  Risk aversion parameter
        N      : int    Number of trading intervals
        tau    : float  Interval duration (days)

        Returns
        -------
        dict with: name, trades (array), total_shares, expected_cost, expected_variance
        )doc"
    );

    // twap_trajectory
    m.def("twap_trajectory",
        [](double X, int N) -> py::dict {
            return schedule_to_dict(twap_trajectory(X, N));
        },
        py::arg("X"), py::arg("N"),
        "Uniform TWAP schedule: n_k = X/N for all k."
    );

    // vwap_trajectory
    m.def("vwap_trajectory",
        [](double X, int N) -> py::dict {
            return schedule_to_dict(vwap_trajectory(X, N));
        },
        py::arg("X"), py::arg("N"),
        "VWAP schedule using a U-shaped intraday volume profile."
    );

    // simulate_execution
    m.def("simulate_execution",
        [](py::array_t<double> trades,
           double sigma, double eta, double gamma_,
           double lambda, int N, double tau,
           double arrival_price, int n_paths, unsigned seed) -> py::dict {
            ACParams p{sigma, eta, gamma_, lambda, N, tau};
            auto stats = simulate_execution(
                to_vec(trades), p, arrival_price, n_paths, seed
            );
            return py::dict(
                "mean_is"_a           = stats.mean_is,
                "std_is"_a            = stats.std_is,
                "var_95"_a            = stats.var_95,
                "cvar_95"_a           = stats.cvar_95,
                "mean_perm_impact"_a  = stats.mean_perm_impact,
                "mean_temp_impact"_a  = stats.mean_temp_impact,
                "mean_timing_risk"_a  = stats.mean_timing_risk,
                "n_paths"_a           = stats.n_paths,
                "samples"_a           = to_ndarray(stats.samples)
            );
        },
        py::arg("trades"),
        py::arg("sigma"), py::arg("eta"), py::arg("gamma"),
        py::arg("lambda_"), py::arg("N"), py::arg("tau"),
        py::arg("arrival_price") = 100.0,
        py::arg("n_paths")       = 10000,
        py::arg("seed")          = 42,
        R"doc(
        Monte Carlo simulation of implementation shortfall.

        Returns
        -------
        dict with: mean_is, std_is, var_95, cvar_95, mean_perm_impact,
                   mean_temp_impact, mean_timing_risk, n_paths, samples
        )doc"
    );
}
