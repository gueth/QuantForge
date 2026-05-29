#pragma once
#include <vector>
#include <string>

namespace qf {

// ── Kalman filter result ────────────────────────────────────
struct KalmanResult {
    std::vector<double> betas;             // dynamic hedge ratios
    std::vector<double> spreads;           // y - beta * x
    std::vector<double> innovations;       // one-step forecast errors
    std::vector<double> innovations_var;   // forecast error variances
};

// ── Performance metrics ──────────────────────────────────────
struct PerformanceMetrics {
    double total_return     = 0.0;
    double annual_return    = 0.0;
    double annual_vol       = 0.0;
    double sharpe_ratio     = 0.0;
    double sortino_ratio    = 0.0;
    double calmar_ratio     = 0.0;
    double max_drawdown     = 0.0;
    double hit_rate         = 0.0;
    double profit_factor    = 0.0;
    double avg_win          = 0.0;
    double avg_loss         = 0.0;
    int    n_trades         = 0;
    int    n_obs            = 0;
};

} // namespace qf
