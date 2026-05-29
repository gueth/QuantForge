#pragma once
#include "exec_types.hpp"
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>
#include <random>
#include <string>

namespace qf {

// ─────────────────────────────────────────────────────────────
// Almgren-Chriss Optimal Execution Model (2001)
//
// Objective: liquidate X shares over N intervals of width τ.
// Minimize: E[cost] + λ · Var[cost]
//
// Cost components (per share):
//   Permanent:  γ · Σ n_k
//   Temporary:  η · Σ (n_k / τ)²  × τ
//   Timing:     σ² · Σ x_k²  × τ
//
// Optimal holdings: x_k = X · sinh(κ (N-k) τ) / sinh(κ N τ)
//   where κ = sqrt(λ σ² / η̃),  η̃ = η - γτ/2
//
// Reference: Almgren & Chriss (2001), "Optimal Execution of
//            Portfolio Transactions", JRI.
// ─────────────────────────────────────────────────────────────


// ── Optimal trajectory ───────────────────────────────────────
inline Schedule ac_optimal_trajectory(double X, const ACParams& p) {
    double eta_tilde = p.eta - p.gamma * p.tau / 2.0;
    if (eta_tilde <= 0)
        throw std::invalid_argument(
            "eta_tilde = eta - gamma*tau/2 must be > 0. Reduce gamma or tau.");

    double kappa = std::sqrt(p.lambda * p.sigma * p.sigma / eta_tilde);

    // Holdings x_k: shares remaining at start of interval k
    std::vector<double> holdings(p.N + 1);
    double sinh_kNt = std::sinh(kappa * p.N * p.tau);

    if (sinh_kNt < 1e-12) {
        // kappa ≈ 0: risk-aversion negligible → uniform (TWAP) solution
        for (int k = 0; k <= p.N; ++k)
            holdings[k] = X * (p.N - k) / static_cast<double>(p.N);
    } else {
        for (int k = 0; k <= p.N; ++k)
            holdings[k] = X * std::sinh(kappa * (p.N - k) * p.tau) / sinh_kNt;
    }

    // Trade sizes n_k = x_k - x_{k+1}
    std::vector<double> trades(p.N);
    for (int k = 0; k < p.N; ++k)
        trades[k] = holdings[k] - holdings[k + 1];

    // Almgren-Chriss analytical expected cost and variance
    // E[cost] = (gamma/2)*X² + eta/tau * Σ n_k²
    //           (permanent impact term + temporary impact)
    double perm_cost = 0.5 * p.gamma * X * X;
    double temp_cost = 0.0;
    for (double n : trades) temp_cost += n * n;
    temp_cost *= p.eta / p.tau;

    double timing_var = 0.0;
    for (int k = 0; k < p.N; ++k)
        timing_var += holdings[k] * holdings[k];
    timing_var *= p.sigma * p.sigma * p.tau;

    Schedule s;
    s.name              = "Almgren-Chriss Optimal";
    s.trades            = trades;
    s.total_shares      = X;
    s.expected_cost     = perm_cost + temp_cost;
    s.expected_variance = timing_var;
    return s;
}


// ── TWAP trajectory ──────────────────────────────────────────
inline Schedule twap_trajectory(double X, int N) {
    Schedule s;
    s.name         = "TWAP";
    s.trades       = std::vector<double>(N, X / N);
    s.total_shares = X;
    // Not computed analytically for TWAP
    s.expected_cost     = 0.0;
    s.expected_variance = 0.0;
    return s;
}


// ── VWAP trajectory (U-shaped intraday volume profile) ───────
inline Schedule vwap_trajectory(double X, int N) {
    // Approximate U-shaped intraday volume profile
    std::vector<double> vol_frac(N);
    double sum = 0.0;
    for (int k = 0; k < N; ++k) {
        double t = (k + 0.5) / N;   // normalised time in [0, 1]
        vol_frac[k] = 0.5 + 2.0 * std::pow(t - 0.5, 2);
        sum += vol_frac[k];
    }
    std::vector<double> trades(N);
    for (int k = 0; k < N; ++k)
        trades[k] = X * vol_frac[k] / sum;

    Schedule s;
    s.name         = "VWAP";
    s.trades       = trades;
    s.total_shares = X;
    s.expected_cost     = 0.0;
    s.expected_variance = 0.0;
    return s;
}


// ── Monte Carlo execution simulation ─────────────────────────
inline ExecutionStats simulate_execution(
    const std::vector<double>& trades,  // child order sizes (shares)
    const ACParams& p,
    double arrival_price,               // mid-price when order arrives
    int    n_paths = 10000,
    unsigned seed  = 42)
{
    const int N = static_cast<int>(trades.size());
    double X = 0.0;
    for (double t : trades) X += t;
    if (X <= 0.0) throw std::invalid_argument("Total shares must be positive");

    std::mt19937_64 rng(seed);
    std::normal_distribution<double> Z(0.0, 1.0);

    std::vector<double> is_samples(n_paths);
    std::vector<double> perm_samples(n_paths);
    std::vector<double> temp_samples(n_paths);

    for (int path = 0; path < n_paths; ++path) {
        double price        = arrival_price;
        double total_exec   = 0.0;
        double perm_impact  = 0.0;
        double temp_impact  = 0.0;

        // Current holdings
        double holdings = X;

        for (int k = 0; k < N; ++k) {
            double n_k = trades[k];

            // Price evolution: BM + permanent impact
            double dp_perm  = p.gamma * n_k;           // permanent impact ($/share)
            double dp_noise = p.sigma * std::sqrt(p.tau) * Z(rng);
            price += dp_perm + dp_noise;

            // Temporary impact: execution premium
            double exec_premium = p.eta * (n_k / p.tau);  // $/share
            double exec_price   = price + exec_premium;

            total_exec  += n_k * exec_price;
            perm_impact += n_k * dp_perm;
            temp_impact += n_k * exec_premium;

            holdings -= n_k;
        }

        // Implementation shortfall = (execution cost / X) - arrival_price
        is_samples[path]   = total_exec / X - arrival_price;
        perm_samples[path] = perm_impact / X;
        temp_samples[path] = temp_impact / X;
    }

    // Statistics
    double sum_is = 0.0, sum_perm = 0.0, sum_temp = 0.0;
    for (int i = 0; i < n_paths; ++i) {
        sum_is   += is_samples[i];
        sum_perm += perm_samples[i];
        sum_temp += temp_samples[i];
    }
    double mean_is   = sum_is   / n_paths;
    double mean_perm = sum_perm / n_paths;
    double mean_temp = sum_temp / n_paths;

    double var = 0.0;
    for (double s : is_samples) var += (s - mean_is) * (s - mean_is);
    var /= (n_paths - 1);

    // 95th percentile VaR and CVaR
    std::vector<double> sorted = is_samples;
    std::sort(sorted.begin(), sorted.end());
    int idx_95 = static_cast<int>(0.95 * n_paths);
    double var_95 = sorted[idx_95];

    double cvar_sum = 0.0; int cnt = 0;
    for (int i = idx_95; i < n_paths; ++i) { cvar_sum += sorted[i]; ++cnt; }
    double cvar_95 = cnt > 0 ? cvar_sum / cnt : var_95;

    double timing_risk = std::sqrt(var) - std::abs(mean_perm + mean_temp);

    ExecutionStats stats;
    stats.mean_is          = mean_is;
    stats.std_is           = std::sqrt(var);
    stats.var_95           = var_95;
    stats.cvar_95          = cvar_95;
    stats.mean_perm_impact = mean_perm;
    stats.mean_temp_impact = mean_temp;
    stats.mean_timing_risk = timing_risk;
    stats.n_paths          = n_paths;
    stats.samples          = is_samples;
    return stats;
}

} // namespace qf
