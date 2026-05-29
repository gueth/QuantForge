#pragma once
#include "alpha_types.hpp"
#include <vector>
#include <cmath>
#include <stdexcept>

namespace qf {

// ── Scalar Kalman filter for dynamic pairs regression ───────
//
// State-space model:
//   State:       beta_t = beta_{t-1} + w_t,   w_t ~ N(0, Q)
//   Observation: y_t = beta_t * x_t + v_t,    v_t ~ N(0, Ve)
//
// Process noise: Q = delta / (1 - delta)
// A small delta (e.g. 1e-5) allows beta to evolve slowly.
// ─────────────────────────────────────────────────────────────

inline KalmanResult kalman_filter(
    const std::vector<double>& y,    // log prices of asset A
    const std::vector<double>& x,    // log prices of asset B
    double delta = 1e-5,             // process noise factor
    double Ve    = 1e-3,             // observation noise variance
    double beta0 = 0.0,              // initial hedge ratio
    double P0    = 1.0               // initial state variance
)
{
    if (y.size() != x.size())
        throw std::invalid_argument("y and x must have the same length");
    if (delta <= 0.0 || delta >= 1.0)
        throw std::invalid_argument("delta must be in (0, 1)");

    const int T = static_cast<int>(y.size());
    KalmanResult res;
    res.betas.resize(T);
    res.spreads.resize(T);
    res.innovations.resize(T);
    res.innovations_var.resize(T);

    const double Q = delta / (1.0 - delta);
    double beta = beta0;
    double P    = P0;

    for (int t = 0; t < T; ++t) {
        // Predict
        double P_pred  = P + Q;
        double innov   = y[t] - beta * x[t];
        double S       = x[t] * x[t] * P_pred + Ve;   // innovation variance

        // Update
        double K   = P_pred * x[t] / S;
        beta       = beta + K * innov;
        P          = (1.0 - K * x[t]) * P_pred;

        res.betas[t]           = beta;
        res.spreads[t]         = y[t] - beta * x[t];
        res.innovations[t]     = innov;
        res.innovations_var[t] = S;
    }

    return res;
}


// ── Rolling z-score ──────────────────────────────────────────
// Compute the z-score of a time series over a rolling window.
// Returns NaN for the first (window - 1) observations.
inline std::vector<double> rolling_zscore(
    const std::vector<double>& series,
    int window)
{
    if (window <= 1)
        throw std::invalid_argument("window must be >= 2");

    const int T = static_cast<int>(series.size());
    std::vector<double> zscores(T, std::numeric_limits<double>::quiet_NaN());

    for (int t = window - 1; t < T; ++t) {
        double sum = 0.0, sum2 = 0.0;
        for (int k = t - window + 1; k <= t; ++k) {
            sum  += series[k];
            sum2 += series[k] * series[k];
        }
        double mean = sum  / window;
        double var  = sum2 / window - mean * mean;
        double std  = var > 1e-14 ? std::sqrt(var) : 1e-14;
        zscores[t]  = (series[t] - mean) / std;
    }

    return zscores;
}


// ── Performance metrics ──────────────────────────────────────
// Compute performance metrics from a daily P&L series.
inline PerformanceMetrics compute_metrics(
    const std::vector<double>& daily_pnl,  // daily returns (not cumulated)
    double risk_free_rate = 0.0,           // annualised risk-free rate
    int trading_days = 252
)
{
    const int T = static_cast<int>(daily_pnl.size());
    if (T == 0) return {};

    // Basic stats
    double sum = 0.0;
    for (double r : daily_pnl) sum += r;
    double mean_ret = sum / T;

    double var = 0.0;
    for (double r : daily_pnl) var += (r - mean_ret) * (r - mean_ret);
    var /= (T > 1 ? T - 1 : 1);
    double daily_vol = std::sqrt(var);

    // Equity curve & drawdown
    double peak = 0.0, cumret = 0.0, max_dd = 0.0;
    for (double r : daily_pnl) {
        cumret += r;
        if (cumret > peak) peak = cumret;
        double dd = peak - cumret;
        if (dd > max_dd) max_dd = dd;
    }

    // Downside deviation (Sortino)
    double rf_daily = risk_free_rate / trading_days;
    double down_var = 0.0; int down_count = 0;
    for (double r : daily_pnl) {
        double excess = r - rf_daily;
        if (excess < 0) { down_var += excess * excess; ++down_count; }
    }
    double down_std = (down_count > 0)
        ? std::sqrt(down_var / down_count)
        : 1e-14;

    // Trade stats
    int n_win = 0, n_loss = 0, n_trades = 0;
    double sum_win = 0.0, sum_loss = 0.0;
    for (double r : daily_pnl) {
        if (r == 0.0) continue;
        ++n_trades;
        if (r > 0) { ++n_win;  sum_win  += r; }
        else        { ++n_loss; sum_loss += r; }
    }

    PerformanceMetrics m;
    m.n_obs         = T;
    m.n_trades      = n_trades;
    m.total_return  = cumret;
    m.annual_return = mean_ret * trading_days;
    m.annual_vol    = daily_vol * std::sqrt(static_cast<double>(trading_days));
    m.sharpe_ratio  = (m.annual_vol > 1e-14)
        ? (m.annual_return - risk_free_rate) / m.annual_vol
        : 0.0;
    m.sortino_ratio = (down_std > 1e-14)
        ? (mean_ret - rf_daily) / down_std * std::sqrt(static_cast<double>(trading_days))
        : 0.0;
    m.max_drawdown  = max_dd;
    m.calmar_ratio  = (max_dd > 1e-14) ? m.annual_return / max_dd : 0.0;
    m.hit_rate      = (n_trades > 0) ? static_cast<double>(n_win) / n_trades : 0.0;
    m.avg_win       = (n_win  > 0)   ? sum_win  / n_win  : 0.0;
    m.avg_loss      = (n_loss > 0)   ? sum_loss / n_loss : 0.0;
    m.profit_factor = (std::abs(sum_loss) > 1e-14) ? sum_win / std::abs(sum_loss) : 0.0;

    return m;
}

} // namespace qf
