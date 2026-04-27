#include <pybind11/pybind11.h>
#include <cmath>
#include <random>

namespace py = pybind11 ;

double mc_call_price(double S0, double K, double r, double sigma, double T, int n_paths) {

    std::mt19937_64 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);
    double sum_payoffs = 0.0;
    for(int i = 0; i < n_paths; i++){
        double Z = norm(rng);
        double ST = S0 * exp((r - 0.5 * pow(sigma, 2)) * T + sigma * Z);
        double payoff = std::max(ST - K, 0.0);
        sum_payoffs += payoff;
    }
    return exp(-r * T) * (sum_payoffs / n_paths);
}

PYBIND11_MODULE(mc_pricer, m){
    m.doc() = "Monte Carlo pricer - C++ backend";
    m.def("mc_call_price", &mc_call_price,
        py::arg("S0"), py::arg("K"), py::arg("r"),
        py::arg("sigma"), py::arg("T"), py::arg("n_paths")=100000);
}