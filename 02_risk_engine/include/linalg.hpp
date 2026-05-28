#pragma once
#include <vector>
#include <cmath>
#include <stdexcept>
#include <algorithm>
#include <numeric>

namespace qf::linalg {

using Matrix = std::vector<std::vector<double>>;
using Vec    = std::vector<double>;

// ── Opérations basiques ──────────────────────────────────────

inline Matrix zeros(int rows, int cols) {
    return Matrix(rows, Vec(cols, 0.0));
}

inline Matrix identity(int n) {
    auto I = zeros(n, n);
    for (int i = 0; i < n; ++i) I[i][i] = 1.0;
    return I;
}

inline double dot(const Vec& a, const Vec& b) {
    double s = 0.0;
    for (size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

inline Vec matvec(const Matrix& A, const Vec& x) {
    int m = A.size(), n = x.size();
    Vec y(m, 0.0);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            y[i] += A[i][j] * x[j];
    return y;
}

inline Matrix matmul(const Matrix& A, const Matrix& B) {
    int m = A.size(), k = B.size(), n = B[0].size();
    auto C = zeros(m, n);
    for (int i = 0; i < m; ++i)
        for (int p = 0; p < k; ++p)
            for (int j = 0; j < n; ++j)
                C[i][j] += A[i][p] * B[p][j];
    return C;
}

inline Matrix transpose(const Matrix& A) {
    int m = A.size(), n = A[0].size();
    auto T = zeros(n, m);
    for (int i = 0; i < m; ++i)
        for (int j = 0; j < n; ++j)
            T[j][i] = A[i][j];
    return T;
}

// ── Covariance (colonnes = actifs) ───────────────────────────
// returns : T x N  →  cov : N x N  (non-annualisée)
inline Matrix covariance(const Matrix& returns) {
    int T = returns.size();
    int N = returns[0].size();
    Vec mean(N, 0.0);
    for (auto& row : returns)
        for (int j = 0; j < N; ++j)
            mean[j] += row[j];
    for (auto& m : mean) m /= T;

    auto cov = zeros(N, N);
    for (auto& row : returns) {
        for (int i = 0; i < N; ++i)
            for (int j = 0; j < N; ++j)
                cov[i][j] += (row[i] - mean[i]) * (row[j] - mean[j]);
    }
    double inv = 1.0 / (T - 1);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j)
            cov[i][j] *= inv;
    return cov;
}

// ── Cholesky (L telle que A = L Lᵀ) ─────────────────────────
inline Matrix cholesky(const Matrix& A) {
    int n = A.size();
    auto L = zeros(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            double s = A[i][j];
            for (int k = 0; k < j; ++k) s -= L[i][k] * L[j][k];
            if (i == j) {
                if (s <= 0.0) throw std::runtime_error("Matrix not positive definite");
                L[i][j] = std::sqrt(s);
            } else {
                L[i][j] = s / L[j][j];
            }
        }
    }
    return L;
}

// ── Jacobi eigensolver (matrice symétrique) ──────────────────
// Retourne (eigenvalues, eigenvectors) triés par ordre décroissant
inline std::pair<Vec, Matrix> eigen_symmetric(Matrix A) {
    int n = A.size();
    auto V = identity(n);
    const double eps = 1e-10;
    const int max_iter = 1000 * n * n;

    for (int iter = 0; iter < max_iter; ++iter) {
        // Trouver le plus grand élément hors-diagonale
        int p = 0, q = 1;
        double max_off = std::abs(A[0][1]);
        for (int i = 0; i < n; ++i)
            for (int j = i + 1; j < n; ++j)
                if (std::abs(A[i][j]) > max_off) {
                    max_off = std::abs(A[i][j]);
                    p = i; q = j;
                }

        if (max_off < eps) break;

        // Rotation de Jacobi
        double theta = 0.5 * std::atan2(2.0 * A[p][q], A[q][q] - A[p][p]);
        double c = std::cos(theta), s = std::sin(theta);

        // Mise à jour A
        double app = A[p][p], aqq = A[q][q], apq = A[p][q];
        A[p][p] = c*c*app - 2*s*c*apq + s*s*aqq;
        A[q][q] = s*s*app + 2*s*c*apq + c*c*aqq;
        A[p][q] = A[q][p] = 0.0;

        for (int r = 0; r < n; ++r) {
            if (r == p || r == q) continue;
            double arp = A[r][p], arq = A[r][q];
            A[r][p] = A[p][r] = c*arp - s*arq;
            A[r][q] = A[q][r] = s*arp + c*arq;
        }

        // Mise à jour des vecteurs propres
        for (int r = 0; r < n; ++r) {
            double vrp = V[r][p], vrq = V[r][q];
            V[r][p] = c*vrp - s*vrq;
            V[r][q] = s*vrp + c*vrq;
        }
    }

    // Extraire et trier par ordre décroissant
    Vec eigenvalues(n);
    for (int i = 0; i < n; ++i) eigenvalues[i] = A[i][i];

    std::vector<int> idx(n);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](int a, int b) {
        return eigenvalues[a] > eigenvalues[b];
    });

    Vec sorted_evals(n);
    auto sorted_evecs = zeros(n, n);
    for (int i = 0; i < n; ++i) {
        sorted_evals[i] = eigenvalues[idx[i]];
        for (int r = 0; r < n; ++r)
            sorted_evecs[r][i] = V[r][idx[i]];
    }
    return {sorted_evals, sorted_evecs};
}

// ── OLS : β = (XᵀX)⁻¹ Xᵀy ──────────────────────────────────
// X : T x K,  y : T  →  β : K
inline Vec ols(const Matrix& X, const Vec& y) {
    int T = X.size(), K = X[0].size();
    // XᵀX
    auto XtX = zeros(K, K);
    for (int i = 0; i < K; ++i)
        for (int j = 0; j < K; ++j)
            for (int t = 0; t < T; ++t)
                XtX[i][j] += X[t][i] * X[t][j];

    // Xᵀy
    Vec Xty(K, 0.0);
    for (int i = 0; i < K; ++i)
        for (int t = 0; t < T; ++t)
            Xty[i] += X[t][i] * y[t];

    // Résolution par élimination de Gauss-Jordan
    int n = K;
    Matrix aug = zeros(n, n + 1);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) aug[i][j] = XtX[i][j];
        aug[i][n] = Xty[i];
    }
    for (int col = 0; col < n; ++col) {
        // Pivot
        int pivot = col;
        for (int row = col+1; row < n; ++row)
            if (std::abs(aug[row][col]) > std::abs(aug[pivot][col])) pivot = row;
        std::swap(aug[col], aug[pivot]);
        double piv = aug[col][col];
        if (std::abs(piv) < 1e-14) throw std::runtime_error("Singular matrix in OLS");
        for (int j = col; j <= n; ++j) aug[col][j] /= piv;
        for (int row = 0; row < n; ++row) {
            if (row == col) continue;
            double f = aug[row][col];
            for (int j = col; j <= n; ++j) aug[row][j] -= f * aug[col][j];
        }
    }
    Vec beta(n);
    for (int i = 0; i < n; ++i) beta[i] = aug[i][n];
    return beta;
}

// ── Statistiques basiques ────────────────────────────────────
inline double mean(const Vec& v) {
    return std::accumulate(v.begin(), v.end(), 0.0) / v.size();
}

inline double variance(const Vec& v, int ddof = 1) {
    double m = mean(v);
    double s = 0.0;
    for (double x : v) s += (x - m) * (x - m);
    return s / (v.size() - ddof);
}

inline double stddev(const Vec& v, int ddof = 1) {
    return std::sqrt(variance(v, ddof));
}

inline double quantile(Vec v, double q) {
    std::sort(v.begin(), v.end());
    double idx = q * (v.size() - 1);
    int lo = static_cast<int>(idx);
    int hi = lo + 1;
    if (hi >= static_cast<int>(v.size())) return v.back();
    return v[lo] + (idx - lo) * (v[hi] - v[lo]);
}

// Φ⁻¹(p)  — algorithme de Beasley-Springer-Moro
inline double norm_ppf(double p) {
    static const double a[] = {-3.969683028665376e+01,  2.209460984245205e+02,
                                -2.759285104469687e+02,  1.383577518672690e+02,
                                -3.066479806614716e+01,  2.506628277459239e+00};
    static const double b[] = {-5.447609879822406e+01,  1.615858368580409e+02,
                                -1.556989798598866e+02,  6.680131188771972e+01,
                                -1.328068155288572e+01};
    static const double c[] = {-7.784894002430293e-03, -3.223964580411365e-01,
                                -2.400758277161838e+00, -2.549732539343734e+00,
                                 4.374664141464968e+00,  2.938163982698783e+00};
    static const double d[] = { 7.784695709041462e-03,  3.224671290700398e-01,
                                  2.445134137142996e+00,  3.754408661907416e+00};
    const double p_lo = 0.02425, p_hi = 1 - p_lo;
    double q, r;
    if (p < p_lo) {
        q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
               ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    } else if (p <= p_hi) {
        q = p - 0.5; r = q*q;
        return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q /
               (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1);
    } else {
        q = std::sqrt(-2.0 * std::log(1 - p));
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) /
                ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1);
    }
}

// φ(x) — densité normale standard
inline double norm_pdf(double x) {
    return std::exp(-0.5 * x * x) / std::sqrt(2.0 * M_PI);
}

} // namespace qf::linalg
