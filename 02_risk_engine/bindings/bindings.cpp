/*
 * QuantForge — Module 2: pybind11 bindings
 * Expose RiskEngine C++ → Python
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "risk_engine.hpp"

namespace py = pybind11;
using namespace pybind11::literals;
using namespace qf;

// ── Helpers numpy → C++ ──────────────────────────────────────

static linalg::Matrix ndarray_to_matrix(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 2) throw std::runtime_error("Expected 2D array");
    int rows = buf.shape[0], cols = buf.shape[1];
    auto ptr = static_cast<double*>(buf.ptr);
    linalg::Matrix m(rows, linalg::Vec(cols));
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            m[i][j] = ptr[i * cols + j];
    return m;
}

static linalg::Vec ndarray_to_vec(py::array_t<double> arr) {
    auto buf = arr.request();
    if (buf.ndim != 1) throw std::runtime_error("Expected 1D array");
    auto ptr = static_cast<double*>(buf.ptr);
    return linalg::Vec(ptr, ptr + buf.shape[0]);
}

static py::array_t<double> vec_to_ndarray(const linalg::Vec& v) {
    auto result = py::array_t<double>(v.size());
    auto buf = result.request();
    std::copy(v.begin(), v.end(), static_cast<double*>(buf.ptr));
    return result;
}

static py::array_t<double> matrix_to_ndarray(const linalg::Matrix& m) {
    if (m.empty()) return py::array_t<double>(std::vector<ssize_t>{0, 0});
    int rows = m.size(), cols = m[0].size();
    auto result = py::array_t<double>({rows, cols});
    auto buf = result.request();
    auto ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < cols; ++j)
            ptr[i * cols + j] = m[i][j];
    return result;
}

// ── Wrapper Python-friendly ──────────────────────────────────

class PyRiskEngine {
public:
    PyRiskEngine(py::array_t<double> returns, py::array_t<double> weights)
        : engine_(ndarray_to_matrix(returns), ndarray_to_vec(weights)) {}

    // VaR / CVaR
    double var_historical(double conf = 0.95, int horizon = 1) const {
        return engine_.var_historical(conf, horizon);
    }
    double var_parametric(double conf = 0.95, int horizon = 1) const {
        return engine_.var_parametric(conf, horizon);
    }
    double cvar_historical(double conf = 0.95, int horizon = 1) const {
        return engine_.cvar_historical(conf, horizon);
    }
    double cvar_parametric(double conf = 0.95, int horizon = 1) const {
        return engine_.cvar_parametric(conf, horizon);
    }

    // Rapport complet → dict Python
    py::dict full_report(double conf = 0.95, int horizon = 1) const {
        auto r = engine_.full_report(conf, horizon);
        return py::dict(
            "var_historical"_a  = r.var_historical,
            "var_parametric"_a  = r.var_parametric,
            "cvar_historical"_a = r.cvar_historical,
            "cvar_parametric"_a = r.cvar_parametric,
            "portfolio_vol"_a   = r.portfolio_vol,
            "portfolio_mean"_a  = r.portfolio_mean,
            "confidence"_a      = r.confidence,
            "horizon_days"_a    = r.horizon_days,
            "n_obs"_a           = r.n_obs
        );
    }

    // PCA → dict Python avec arrays numpy
    py::dict pca(int n_components = -1) const {
        auto res = engine_.pca(n_components);
        return py::dict(
            "eigenvalues"_a              = vec_to_ndarray(res.eigenvalues),
            "explained_variance_ratio"_a = vec_to_ndarray(res.explained_variance_ratio),
            "cumulative_variance"_a      = vec_to_ndarray(res.cumulative_variance),
            "factor_loadings"_a          = matrix_to_ndarray(res.factor_loadings),
            "factor_returns"_a           = matrix_to_ndarray(res.factor_returns),
            "n_components"_a             = res.n_components,
            "n_assets"_a                 = res.n_assets
        );
    }

    // Fama-French OLS
    py::list fama_french(
        py::array_t<double> factors,
        py::array_t<double> rf,
        py::list names) const
    {
        auto f  = ndarray_to_matrix(factors);
        auto rf_ = ndarray_to_vec(rf);
        std::vector<std::string> ns;
        for (auto n : names) ns.push_back(n.cast<std::string>());
        auto results = engine_.fama_french(f, rf_, ns);

        py::list out;
        for (auto& r : results) {
            out.append(py::dict(
                "asset"_a          = r.asset_name,
                "alpha"_a          = r.alpha,
                "betas"_a          = vec_to_ndarray(linalg::Vec(r.betas)),
                "r_squared"_a      = r.r_squared,
                "systematic_vol"_a = r.systematic_vol,
                "idio_vol"_a       = r.idio_vol,
                "residuals"_a      = vec_to_ndarray(r.residuals)
            ));
        }
        return out;
    }

    // Rolling VaR
    py::array_t<double> rolling_var(
        double conf = 0.95, int window = 252, bool parametric = false) const
    {
        return vec_to_ndarray(engine_.rolling_var(conf, window, parametric));
    }

    // Portfolio returns
    py::array_t<double> portfolio_returns() const {
        return vec_to_ndarray(engine_.portfolio_returns());
    }

    int n_obs()    const { return engine_.n_obs(); }
    int n_assets() const { return engine_.n_assets(); }

private:
    RiskEngine engine_;
};


// ── Module pybind11 ──────────────────────────────────────────

PYBIND11_MODULE(risk_engine_cpp, m) {
    m.doc() = "QuantForge Module 2 — C++ Risk Engine (VaR, CVaR, PCA, Fama-French)";

    py::class_<PyRiskEngine>(m, "RiskEngine")
        .def(py::init<py::array_t<double>, py::array_t<double>>(),
             py::arg("returns"), py::arg("weights"),
             R"doc(
             Moteur de risque quantitatif.

             Parameters
             ----------
             returns : np.ndarray (T, N)
                 Matrice de rendements log journaliers. T = jours, N = actifs.
             weights : np.ndarray (N,)
                 Poids du portefeuille, doit sommer à 1.
             )doc")
        .def("var_historical",  &PyRiskEngine::var_historical,
             py::arg("confidence") = 0.95, py::arg("horizon") = 1,
             "VaR historique : -Q_{1-alpha}(r_p) * sqrt(horizon)")
        .def("var_parametric",  &PyRiskEngine::var_parametric,
             py::arg("confidence") = 0.95, py::arg("horizon") = 1,
             "VaR paramétrique (Gaussien) : -(mu + z_alpha * sigma) * sqrt(horizon)")
        .def("cvar_historical", &PyRiskEngine::cvar_historical,
             py::arg("confidence") = 0.95, py::arg("horizon") = 1,
             "CVaR / Expected Shortfall historique")
        .def("cvar_parametric", &PyRiskEngine::cvar_parametric,
             py::arg("confidence") = 0.95, py::arg("horizon") = 1,
             "CVaR paramétrique (Gaussien) : -(mu - sigma * phi(z) / (1-alpha))")
        .def("full_report",     &PyRiskEngine::full_report,
             py::arg("confidence") = 0.95, py::arg("horizon") = 1,
             "Rapport complet VaR/CVaR → dict")
        .def("pca",             &PyRiskEngine::pca,
             py::arg("n_components") = -1,
             "Décomposition PCA de la matrice de covariance → dict")
        .def("fama_french",     &PyRiskEngine::fama_french,
             py::arg("factors"), py::arg("rf"), py::arg("names"),
             "Régression OLS Fama-French par actif → list de dicts")
        .def("rolling_var",     &PyRiskEngine::rolling_var,
             py::arg("confidence") = 0.95, py::arg("window") = 252,
             py::arg("parametric") = false,
             "VaR glissante sur fenêtre → np.ndarray")
        .def("portfolio_returns", &PyRiskEngine::portfolio_returns,
             "Rendements journaliers du portefeuille → np.ndarray")
        .def_property_readonly("n_obs",    &PyRiskEngine::n_obs)
        .def_property_readonly("n_assets", &PyRiskEngine::n_assets);
}
