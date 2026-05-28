#pragma once
#include "risk_types.hpp"
#include "linalg.hpp"
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <stdexcept>

namespace qf {

using namespace linalg;

class RiskEngine {
public:
    // ── Constructeur ────────────────────────────────────────
    // returns  : T x N  (T jours, N actifs)
    // weights  : N  (somme = 1)
    RiskEngine(const Matrix& returns, const Vec& weights)
        : returns_(returns)
        , weights_(weights)
        , T_(returns.size())
        , N_(returns.empty() ? 0 : returns[0].size())
    {
        if (weights_.size() != static_cast<size_t>(N_))
            throw std::invalid_argument("weights size != n_assets");
        double ws = std::accumulate(weights_.begin(), weights_.end(), 0.0);
        if (std::abs(ws - 1.0) > 1e-6)
            throw std::invalid_argument("weights must sum to 1");
        compute_portfolio_returns();
    }

    // ── VaR Historique ───────────────────────────────────────
    // VaR_hist(α) = -Q_{1-α}(r_p) × √h
    double var_historical(double confidence = 0.95, int horizon = 1) const {
        double q = quantile(port_returns_, 1.0 - confidence);
        return -q * std::sqrt(static_cast<double>(horizon));
    }

    // ── VaR Paramétrique (Gaussien) ───────────────────────────
    // VaR_param(α) = -(μ + z_α · σ) × √h
    double var_parametric(double confidence = 0.95, int horizon = 1) const {
        double mu    = mean(port_returns_);
        double sigma = stddev(port_returns_);
        double z     = norm_ppf(1.0 - confidence);
        return -(mu + z * sigma) * std::sqrt(static_cast<double>(horizon));
    }

    // ── CVaR Historique (Expected Shortfall) ─────────────────
    // CVaR(α) = -E[r_p | r_p ≤ VaR_hist(α)]
    double cvar_historical(double confidence = 0.95, int horizon = 1) const {
        double var = quantile(port_returns_, 1.0 - confidence);
        double sum = 0.0; int count = 0;
        for (double r : port_returns_) {
            if (r <= var) { sum += r; ++count; }
        }
        if (count == 0) return 0.0;
        return -(sum / count) * std::sqrt(static_cast<double>(horizon));
    }

    // ── CVaR Paramétrique ────────────────────────────────────
    // CVaR_param(α) = -(μ - σ · φ(z_α) / (1-α))
    double cvar_parametric(double confidence = 0.95, int horizon = 1) const {
        double mu    = mean(port_returns_);
        double sigma = stddev(port_returns_);
        double alpha = 1.0 - confidence;
        double z     = norm_ppf(alpha);
        double es    = -(mu - sigma * norm_pdf(z) / alpha);
        return es * std::sqrt(static_cast<double>(horizon));
    }

    // ── Rapport complet VaR/CVaR ──────────────────────────────
    VaRResult full_report(double confidence = 0.95, int horizon = 1) const {
        VaRResult r;
        r.confidence      = confidence;
        r.horizon_days    = horizon;
        r.n_obs           = T_;
        r.var_historical  = var_historical(confidence, horizon);
        r.var_parametric  = var_parametric(confidence, horizon);
        r.cvar_historical = cvar_historical(confidence, horizon);
        r.cvar_parametric = cvar_parametric(confidence, horizon);
        r.portfolio_vol   = stddev(port_returns_) * std::sqrt(252.0);
        r.portfolio_mean  = mean(port_returns_) * 252.0;
        return r;
    }

    // ── PCA ──────────────────────────────────────────────────
    // Décomposition spectrale Σ = V Λ Vᵀ
    // Retourne loadings, factor returns, valeurs/vecteurs propres
    PCAResult pca(int n_components = -1) const {
        if (n_components < 0 || n_components > N_)
            n_components = N_;

        // Centrage
        Vec col_mean(N_, 0.0);
        for (auto& row : returns_)
            for (int j = 0; j < N_; ++j)
                col_mean[j] += row[j];
        for (auto& m : col_mean) m /= T_;

        Matrix R_c(T_, Vec(N_));
        for (int t = 0; t < T_; ++t)
            for (int j = 0; j < N_; ++j)
                R_c[t][j] = returns_[t][j] - col_mean[j];

        // Matrice de covariance
        Matrix cov = covariance(returns_);

        // Valeurs propres (Jacobi)
        auto [evals, evecs] = eigen_symmetric(cov);  // triés décroissants

        double total_var = 0.0;
        for (double e : evals) total_var += e;

        PCAResult res;
        res.n_components = n_components;
        res.n_assets     = N_;

        // Tronquer à n_components
        for (int i = 0; i < n_components; ++i) {
            res.eigenvalues.push_back(evals[i]);
            res.explained_variance_ratio.push_back(evals[i] / total_var);
        }

        // Variance cumulée
        double cum = 0.0;
        for (double e : res.explained_variance_ratio) {
            cum += e;
            res.cumulative_variance.push_back(cum);
        }

        // Factor loadings : N x K (evecs colonnes tronquées)
        res.eigenvectors = zeros(N_, n_components);
        res.factor_loadings = zeros(N_, n_components);
        for (int i = 0; i < N_; ++i)
            for (int k = 0; k < n_components; ++k) {
                res.eigenvectors[i][k]    = evecs[i][k];
                res.factor_loadings[i][k] = evecs[i][k];
            }

        // Factor returns : T x K  =  R_centered × V[:, :K]
        res.factor_returns = zeros(T_, n_components);
        for (int t = 0; t < T_; ++t)
            for (int k = 0; k < n_components; ++k)
                for (int j = 0; j < N_; ++j)
                    res.factor_returns[t][k] += R_c[t][j] * evecs[j][k];

        return res;
    }

    // ── Fama-French OLS ──────────────────────────────────────
    // factors  : T x K  (Mkt-RF, SMB, HML…)
    // rf       : T  (taux sans risque journalier, peut être zéro)
    // names    : N noms d'actifs
    std::vector<FFRegressionResult> fama_french(
        const Matrix& factors,
        const Vec& rf,
        const std::vector<std::string>& names) const
    {
        int K = factors[0].size();
        int Tf = factors.size();
        int T_common = std::min(T_, Tf);

        // Matrice X = [1 | factors]  (T_common x K+1)
        Matrix X(T_common, Vec(K + 1));
        for (int t = 0; t < T_common; ++t) {
            X[t][0] = 1.0;
            for (int k = 0; k < K; ++k)
                X[t][k + 1] = factors[t][k];
        }

        std::vector<FFRegressionResult> results;

        for (int n = 0; n < N_; ++n) {
            // Excès de rendement
            Vec y(T_common);
            for (int t = 0; t < T_common; ++t)
                y[t] = returns_[t][n] - (rf.empty() ? 0.0 : rf[t]);

            // OLS
            Vec coef = ols(X, y);

            // Résidus
            Vec y_hat(T_common), eps(T_common);
            for (int t = 0; t < T_common; ++t) {
                y_hat[t] = 0.0;
                for (int k = 0; k <= K; ++k) y_hat[t] += X[t][k] * coef[k];
                eps[t] = y[t] - y_hat[t];
            }

            // R²
            double ss_res = 0.0, ss_tot = 0.0;
            double y_mean = mean(y);
            for (int t = 0; t < T_common; ++t) {
                ss_res += eps[t] * eps[t];
                ss_tot += (y[t] - y_mean) * (y[t] - y_mean);
            }
            double r2 = (ss_tot > 1e-14) ? 1.0 - ss_res / ss_tot : 0.0;

            // Vol systématique & idiosyncratique
            Vec systematic(T_common, 0.0);
            for (int t = 0; t < T_common; ++t)
                for (int k = 0; k < K; ++k)
                    systematic[t] += factors[t][k] * coef[k + 1];

            double sys_vol  = stddev(systematic) * std::sqrt(252.0);
            double idio_vol = stddev(eps)         * std::sqrt(252.0);

            FFRegressionResult res;
            res.asset_name     = (n < static_cast<int>(names.size())) ? names[n] : "Asset_"+std::to_string(n);
            res.alpha          = coef[0];
            res.betas          = Vec(coef.begin() + 1, coef.end());
            res.r_squared      = r2;
            res.systematic_vol = sys_vol;
            res.idio_vol       = idio_vol;
            res.residuals      = eps;
            results.push_back(std::move(res));
        }
        return results;
    }

    // ── Rolling VaR ─────────────────────────────────────────
    // Retourne une série de T - window + 1 valeurs
    Vec rolling_var(double confidence = 0.95, int window = 252,
                    bool parametric = false) const
    {
        if (window >= T_) throw std::invalid_argument("window >= T");
        Vec result;
        result.reserve(T_ - window + 1);
        for (int i = window; i <= T_; ++i) {
            // Sous-matrice returns_[i-window : i]
            Matrix sub_ret(returns_.begin() + i - window, returns_.begin() + i);
            RiskEngine sub(sub_ret, weights_);
            result.push_back(parametric
                ? sub.var_parametric(confidence)
                : sub.var_historical(confidence));
        }
        return result;
    }

    // ── Accesseurs ───────────────────────────────────────────
    const Vec& portfolio_returns() const { return port_returns_; }
    int n_obs()    const { return T_; }
    int n_assets() const { return N_; }

private:
    Matrix returns_;
    Vec    weights_;
    int    T_, N_;
    Vec    port_returns_;

    void compute_portfolio_returns() {
        port_returns_.resize(T_);
        for (int t = 0; t < T_; ++t) {
            double r = 0.0;
            for (int j = 0; j < N_; ++j)
                r += weights_[j] * returns_[t][j];
            port_returns_[t] = r;
        }
    }
};

} // namespace qf
