#pragma once
#include <vector>
#include <string>
#include <stdexcept>

namespace qf {

// ── Résultats VaR/CVaR ──────────────────────────────────────
struct VaRResult {
    double var_historical   = 0.0;
    double var_parametric   = 0.0;
    double cvar_historical  = 0.0;
    double cvar_parametric  = 0.0;
    double portfolio_vol    = 0.0;   // annualisé
    double portfolio_mean   = 0.0;   // annualisé
    double confidence       = 0.95;
    int    horizon_days     = 1;
    int    n_obs            = 0;
};

// ── Résultats PCA ────────────────────────────────────────────
struct PCAResult {
    std::vector<double> eigenvalues;           // décroissants
    std::vector<std::vector<double>> eigenvectors;  // [n_assets][n_components]
    std::vector<double> explained_variance_ratio;
    std::vector<double> cumulative_variance;
    // Factor loadings : lignes = assets, colonnes = composantes
    std::vector<std::vector<double>> factor_loadings;
    // Factor returns : lignes = T, colonnes = composantes
    std::vector<std::vector<double>> factor_returns;
    int n_components = 0;
    int n_assets     = 0;
};

// ── Résultats Fama-French OLS ────────────────────────────────
struct FFRegressionResult {
    std::string asset_name;
    double alpha           = 0.0;   // journalier
    std::vector<double> betas;      // [n_factors]
    double r_squared       = 0.0;
    double systematic_vol  = 0.0;   // annualisé
    double idio_vol        = 0.0;   // annualisé
    std::vector<double> residuals;
};

} // namespace qf
