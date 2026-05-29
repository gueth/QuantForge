#pragma once
#include <vector>
#include <string>

namespace qf {

// ── Almgren-Chriss model parameters ──────────────────────────
struct ACParams {
    double sigma;    // daily price volatility (as a fraction)
    double eta;      // temporary market impact coefficient ($/share²/day)
    double gamma;    // permanent market impact coefficient ($/share)
    double lambda;   // risk aversion (trade-off cost vs. timing risk)
    int    N;        // number of trading intervals
    double tau;      // interval duration in trading days (e.g. 1/N for 1-day horizon)
};

// ── Execution statistics ─────────────────────────────────────
struct ExecutionStats {
    double mean_is;          // mean implementation shortfall ($/share)
    double std_is;           // std deviation of IS
    double var_95;           // 95th percentile IS (VaR of execution cost)
    double cvar_95;          // conditional VaR at 95%
    double mean_perm_impact; // mean permanent impact component
    double mean_temp_impact; // mean temporary impact component
    double mean_timing_risk; // mean timing risk component
    int    n_paths;
    std::vector<double> samples;   // full IS distribution
};

// ── Execution schedule ────────────────────────────────────────
struct Schedule {
    std::string name;
    std::vector<double> trades;    // size of each child order
    double total_shares;
    double expected_cost;          // Almgren-Chriss expected cost formula
    double expected_variance;      // Almgren-Chriss variance formula
};

} // namespace qf
