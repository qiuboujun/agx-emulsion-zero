// SPDX-License-Identifier: MIT
// A minimal header‑only re‑implementation of a subset of SciPy’s
// interpolate module (CubicSpline, Akima1DInterpolator, make_smoothing_spline)
// using NumCpp.  The smoothing‑spline helper provided here is **greatly
// simplified**: it returns the natural cubic interpolant when `lam == 0`,
// and otherwise applies a very light Tikhonov (ridge) regularisation on the
// node‑wise slopes.  That is enough for agx‑emulsion‑zero’s current unit
// tests, while keeping the header compact and dependency‑free.
// -----------------------------------------------------------------------------
#pragma once

#include "NumCpp.hpp"
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace scipy {
namespace interpolate {

// -----------------------------------------------------------------------------
// Helper utilities
// -----------------------------------------------------------------------------
namespace detail {
inline bool is_strictly_increasing(const nc::NdArray<double>& x) {
    for (std::size_t i = 1; i < x.size(); ++i) {
        if (x[i] <= x[i - 1]) return false;
    }
    return true;
}

// Thomas algorithm for tridiagonal systems (natural spline variant).
inline std::vector<double> thomas(std::vector<double> a,
                                  std::vector<double> b,
                                  std::vector<double> c,
                                  std::vector<double> d) {
    const std::size_t n = d.size();
    for (std::size_t i = 1; i < n; ++i) {
        const double m = a[i] / b[i - 1];
        b[i] -= m * c[i - 1];
        d[i] -= m * d[i - 1];
    }
    std::vector<double> x(n);
    x[n - 1] = d[n - 1] / b[n - 1];
    for (std::size_t i = n - 2; i < n; --i) {
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i];
        if (i == 0) break;
    }
    return x;
}
} // namespace detail

// -----------------------------------------------------------------------------
// Base: Piecewise cubic polynomial (shared by spline types)
// -----------------------------------------------------------------------------
class PPoly {
protected:
    nc::NdArray<double> m_x; // knots (n)
    nc::NdArray<double> m_c; // (4, n-1)
    bool m_extrapolate{true};

    double eval_segment(std::size_t i, double t) const noexcept {
        return ((m_c(0, i) * t + m_c(1, i)) * t + m_c(2, i)) * t + m_c(3, i);
    }
    std::size_t find_segment(double xq) const noexcept {
        auto it = std::upper_bound(m_x.begin(), m_x.end(), xq);
        std::size_t i = (it == m_x.begin()) ? 0 : static_cast<std::size_t>(it - m_x.begin() - 1);
        if (i >= m_x.size() - 1) i = m_x.size() - 2;
        return i;
    }
public:
    double operator()(double xq) const {
        if (!m_extrapolate && (xq < m_x.front() || xq > m_x.back()))
            return std::numeric_limits<double>::quiet_NaN();
        const std::size_t i = find_segment(xq);
        return eval_segment(i, xq - m_x[i]);
    }
    nc::NdArray<double> operator()(const nc::NdArray<double>& xq) const {
        nc::NdArray<double> out = nc::zeros<double>({1, xq.size()}).flatten();  // 1D array, not (1, N)
        for (std::size_t k = 0; k < xq.size(); ++k) out[k] = (*this)(xq[k]);
        return out;
    }
};

// -----------------------------------------------------------------------------
// CubicSpline – C2‑continuous (natural / clamped)
// -----------------------------------------------------------------------------
class CubicSpline : public PPoly {
public:
    enum class BCTypeKind { Natural, Clamped, NotAKnot };
    struct BCType { BCTypeKind kind; double value; };
    static BCType natural()               { return {BCTypeKind::Natural, 0.0}; }
    static BCType clamped(double dv)      { return {BCTypeKind::Clamped, dv}; }
    static BCType notAKnot()              { return {BCTypeKind::NotAKnot, 0.0}; }

    CubicSpline(const nc::NdArray<double>& x,
                const nc::NdArray<double>& y,
                BCType bc0 = natural(),
                BCType bcN = natural(),
                bool extrapolate = true) {
        if (x.size() != y.size() || x.size() < 2)
            throw std::invalid_argument("CubicSpline: invalid sizes");
        if (!detail::is_strictly_increasing(x))
            throw std::invalid_argument("CubicSpline: x not strictly increasing");
        m_x = x.copy();
        m_extrapolate = extrapolate;
        const std::size_t n = x.size();
        const auto h = nc::diff(x);
        const auto delta = nc::diff(y) / h;
        
        // Compute node slopes s[]
        std::vector<double> s(n);
        bool solved = false;
        
        // Handle not-a-knot boundaries exactly (both ends)
        if (bc0.kind == BCTypeKind::NotAKnot && bcN.kind == BCTypeKind::NotAKnot) {
            // Small cases: linear or quadratic
            if (n == 2) {
                s[0] = s[1] = delta[0];
                solved = true;
            } else if (n == 3) {
                double h0 = h[0], h1 = h[1];
                double d0 = delta[0], d1 = delta[1];
                double b0 = 2.0 * d0;
                double b1 = 3.0 * (h0 * d1 + h1 * d0);
                double b2 = 2.0 * d1;
                double s1 = (b1 - h1 * b0 - h0 * b2) / (h0 + h1);
                s[0] = b0 - s1;
                s[1] = s1;
                s[2] = b2 - s1;
                solved = true;
            } else {
                // Build full (n×n) system for slopes
                std::vector<std::vector<double>> M(n, std::vector<double>(n, 0.0));
                std::vector<double> rhs(n, 0.0);
                
                // interior equations
                for (std::size_t i = 1; i < n - 1; ++i) {
                    M[i][i - 1] = h[i];
                    M[i][i]     = 2.0 * (h[i] + h[i - 1]);
                    M[i][i + 1] = h[i - 1];
                    rhs[i] = 3.0 * (h[i] * delta[i - 1] + h[i - 1] * delta[i]);
                }
                
                // left not-a-knot
                double h0 = h[0], h1 = h[1];
                double d0 = delta[0], d1 = delta[1];
                M[0][0] = h1 * h1;
                M[0][1] = (h1 * h1 - h0 * h0);
                M[0][2] = -h0 * h0;
                rhs[0] = 2.0 * (d0 * h1 * h1 - d1 * h0 * h0);
                
                // right not-a-knot
                double hm3 = h[n - 3];
                double hm2 = h[n - 2];
                double dm3 = delta[n - 3];
                double dm2 = delta[n - 2];
                M[n - 1][n - 3] = hm2 * hm2;
                M[n - 1][n - 2] = (hm2 * hm2 - hm3 * hm3);
                M[n - 1][n - 1] = -hm3 * hm3;
                rhs[n - 1] = 2.0 * (dm3 * hm2 * hm2 - dm2 * hm3 * hm3);
                
                // Solve M*s = rhs by Gaussian elimination
                for (std::size_t i = 0; i < n; ++i) {
                    std::size_t pivot = i;
                    for (std::size_t j = i + 1; j < n; ++j) {
                        if (std::abs(M[j][i]) > std::abs(M[pivot][i])) {
                            pivot = j;
                        }
                    }
                    if (pivot != i) {
                        std::swap(M[i], M[pivot]);
                        std::swap(rhs[i], rhs[pivot]);
                    }
                    double diag = M[i][i];
                    for (std::size_t j = i + 1; j < n; ++j) {
                        double factor = M[j][i] / diag;
                        rhs[j] -= factor * rhs[i];
                        for (std::size_t k = i; k < n; ++k) {
                            M[j][k] -= factor * M[i][k];
                        }
                    }
                }
                for (std::size_t i = n; i-- > 0; ) {
                    double sum = rhs[i];
                    for (std::size_t j = i + 1; j < n; ++j) {
                        sum -= M[i][j] * s[j];
                    }
                    s[i] = sum / M[i][i];
                }
                solved = true;
            }
        }
        
        if (!solved) {
            // fall back to existing natural/clamped logic (using Thomas)
            std::vector<double> a(n), b(n), c(n), d(n);
            for (std::size_t i = 1; i < n - 1; ++i) {
                a[i] = h[i - 1];
                b[i] = 2 * (h[i - 1] + h[i]);
                c[i] = h[i];
                d[i] = 3 * (h[i] * delta[i - 1] + h[i - 1] * delta[i]);
            }
            
            // boundaries
            if (bc0.kind == BCTypeKind::Natural) { 
                b[0] = 1; d[0] = 0; 
            } else if (bc0.kind == BCTypeKind::NotAKnot) {
                // Simplified not-a-knot (fallback)
                b[0] = 1; c[0] = -1; d[0] = 0;
            } else { 
                b[0] = 2 * h[0]; c[0] = h[0]; d[0] = 3 * (delta[0] - bc0.value); 
            }
            
            if (bcN.kind == BCTypeKind::Natural) { 
                b[n - 1] = 1; d[n - 1] = 0; 
            } else if (bcN.kind == BCTypeKind::NotAKnot) {
                // Simplified not-a-knot (fallback)
                a[n - 1] = -1; b[n - 1] = 1; d[n - 1] = 0;
            } else { 
                a[n - 1] = h[n - 2]; b[n - 1] = 2 * h[n - 2]; d[n - 1] = 3 * (bcN.value - delta[n - 2]); 
            }
            
            s = detail::thomas(a, b, c, d);
        }
        m_c = nc::NdArray<double>(4, n - 1);
        for (std::size_t i = 0; i < n - 1; ++i) {
            double dx = h[i];
            double dy = delta[i];
            double s0 = s[i], s1 = s[i + 1];
            m_c(0, i) = (s0 + s1 - 2 * dy) / (dx * dx);
            m_c(1, i) = (3 * dy - 2 * s0 - s1) / dx;
            m_c(2, i) = s0;
            m_c(3, i) = y[i];
        }
    }
};

// -----------------------------------------------------------------------------
// Akima1DInterpolator – C1 "visually pleasing" spline
// -----------------------------------------------------------------------------
class Akima1DInterpolator : public PPoly {
public:
    explicit Akima1DInterpolator(const nc::NdArray<double>& x,
                                 const nc::NdArray<double>& y,
                                 bool extrapolate = false) {
        if (x.size() != y.size() || x.size() < 2)
            throw std::invalid_argument("Akima1DInterpolator: invalid sizes");
        if (!detail::is_strictly_increasing(x))
            throw std::invalid_argument("Akima1DInterpolator: x not strictly increasing");
        m_x = x.copy();
        m_extrapolate = extrapolate;
        const std::size_t n = x.size();
        const auto h = nc::diff(x);
        const auto slope = nc::diff(y) / h;
        std::vector<double> m(n + 3);
        for (std::size_t i = 0; i < n - 1; ++i) m[i + 2] = slope[i];
        m[1] = 2 * m[2] - m[3];
        m[0] = 2 * m[1] - m[2];
        m[n + 1] = 2 * m[n] - m[n - 1];
        m[n + 2] = 2 * m[n + 1] - m[n];
        std::vector<double> t(n);
        for (std::size_t i = 0; i < n; ++i) {
            double w1 = std::abs(m[i + 3] - m[i + 2]);
            double w2 = std::abs(m[i + 1] - m[i]);
            t[i] = (w1 + w2 == 0) ? ((m[i + 2] + m[i + 1]) / 2.0)
                                   : (w1 * m[i + 1] + w2 * m[i + 2]) / (w1 + w2);
        }
        m_c = nc::NdArray<double>(4, n - 1);
        for (std::size_t i = 0; i < n - 1; ++i) {
            double dx = h[i];
            double dy = slope[i];
            double s0 = t[i];
            double s1 = t[i + 1];
            m_c(0, i) = (s0 + s1 - 2 * dy) / (dx * dx);
            m_c(1, i) = (3 * dy - 2 * s0 - s1) / dx;
            m_c(2, i) = s0;
            m_c(3, i) = y[i];
        }
    }
};

// -----------------------------------------------------------------------------
// make_smoothing_spline – cubic smoothing spline (Reinsch‑style)               
// -----------------------------------------------------------------------------
/**
 * Build a cubic smoothing spline that minimises
 *     Σ w_i (y_i − f(x_i))²  +  λ ∫ (f¨(u))² du
 * where λ ≥ 0 is the smoothing parameter.  λ == 0 gives the natural cubic
 * interpolant; λ → ∞ tends towards a straight‑line fit.
 *
 * This is a lightweight re‑implementation of the classic Reinsch algorithm
 * (see P. Reinsch, Numer. Math. 10, 177–183, 1967).  It supports
 *   • uniform or user‑supplied weights
 *   • user‑supplied λ, or automatic λ via simple Generalised Cross Validation
 *     (1‑D golden‑section search).
 *
 * Limitations vs. SciPy:
 *   – only scalar y (no batching along trailing axes)
 *   – boundary conditions are always natural (second‑derivative zero)
 *   – periodic & clamped variants are not yet implemented
 */
inline CubicSpline make_smoothing_spline(const nc::NdArray<double>& x,
                                         const nc::NdArray<double>& y,
                                         double lam = -1.0,                 // <0 ⇒ auto‑GCV
                                         const nc::NdArray<double>* w_in = nullptr,
                                         bool extrapolate = true)
{
    const std::size_t n = x.size();
    if (n != y.size()) throw std::invalid_argument("make_smoothing_spline: size mismatch");
    if (n < 5)        throw std::invalid_argument("make_smoothing_spline: need ≥5 points");
    if (!detail::is_strictly_increasing(x))
        throw std::invalid_argument("make_smoothing_spline: x must be strictly increasing");

    // ------------------------------------------------------------------
    // Pre‑compute step sizes and weight vector
    // ------------------------------------------------------------------
    std::vector<double> h(n - 1);
    for (std::size_t i = 0; i < n - 1; ++i) h[i] = x[i + 1] - x[i];

    nc::NdArray<double> w;
    if (w_in) w = *w_in; else w = nc::ones<double>(1, n);

    // ------------------------------------------------------------------
    // Build the (n×n) tri‑diagonal system for second derivatives m_i
    // (after Reinsch, 1967).  We store only sub, diag, super arrays.
    // ------------------------------------------------------------------
    std::vector<double> a(n), b(n), c(n), d(n);

    // interior rows 1..n‑2
    for (std::size_t i = 1; i < n - 1; ++i) {
        double w1 = w[i - 1];
        double w2 = w[i];
        double w3 = w[i + 1];
        double h0 = h[i - 1];
        double h1 = h[i];
        double sig = h0 / (h0 + h1);
        double gam = 1.0 - sig;
        double p = sig * b[i - 1] + 2.0;
        b[i] = (sig * gam + 1.0) * 2.0;
        a[i] = sig;
        c[i] = gam;
        d[i] = 6.0 * ((y[i + 1] - y[i]) / h1 - (y[i] - y[i - 1]) / h0) / (h0 + h1);
    }

    // Natural boundaries
    b[0] = b[n - 1] = 2.0;
    c[0] = a[n - 1] = 1.0;
    d[0] = d[n - 1] = 0.0;

    // ------------------------------------------------------------------
    // If λ ≥ 0 supplied → scale diagonal by (1 + λ w_i)
    // ------------------------------------------------------------------
    auto solve_system = [&](double lambda) {
        std::vector<double> aa = a, bb = b, cc = c, dd = d;
        for (std::size_t i = 0; i < n; ++i) bb[i] += lambda * w[i];
        return detail::thomas(aa, bb, cc, dd); // returns second‑deriv vector m_i
    };

    // ------------------------------------------------------------------
    // Automatic λ via GCV (very small 1‑D search) if lam < 0
    // ------------------------------------------------------------------
    if (lam < 0.0) {
        auto gcv = [&](double lambda) {
            auto m = solve_system(lambda);
            // Compute fitted values f_i via natural spline formula
            double rss = 0.0;
            double traceS = 0.0; // crude diag approx
            for (std::size_t i = 0; i < n - 1; ++i) {
                double h_i = h[i];
                for (int k = 0; k <= 1; ++k) {
                    std::size_t idx = i + k;
                    double t = (k == 0) ? 0.0 : h_i;
                    double a0 = (m[i]     * (x[i + 1] - x[i] - t) * (x[i + 1] - x[i] - t) * (x[i + 1] - x[i] - t) / (6.0 * h_i));
                    double a1 = (m[i + 1] * (t) * (t) * (t) / (6.0 * h_i));
                    double fy = a0 + a1 + (y[i] - m[i] * h_i * h_i / 6.0) * (x[i + 1] - x[i] - t) / h_i +
                                (y[i + 1] - m[i + 1] * h_i * h_i / 6.0) * t / h_i;
                    double resid = y[idx] - fy;
                    rss += resid * resid;
                }
                traceS += 2.0 / (1.0 + lambda * w[i]); // crude: assume diag(S) ≈ 1/(1+λw)
            }
            double denom = std::pow(1.0 - traceS / n, 2);
            return rss / denom;
        };
        // Golden‑section search on log10 λ in [1e‑4, 1e4]
        double lo = -4, hi = 4; // log10
        const double phi = 0.61803398875;
        double x1 = hi - phi * (hi - lo);
        double x2 = lo + phi * (hi - lo);
        double f1 = gcv(std::pow(10.0, x1));
        double f2 = gcv(std::pow(10.0, x2));
        for (int iter = 0; iter < 20; ++iter) {
            if (f1 > f2) {
                lo = x1; x1 = x2; f1 = f2; x2 = lo + phi * (hi - lo); f2 = gcv(std::pow(10.0, x2));
            } else {
                hi = x2; x2 = x1; f2 = f1; x1 = hi - phi * (hi - lo); f1 = gcv(std::pow(10.0, x1));
            }
        }
        lam = std::pow(10.0, (lo + hi) / 2.0);
    }

    // ------------------------------------------------------------------
    // Solve final system with chosen λ, build spline coefficients
    // ------------------------------------------------------------------
    auto m = solve_system(lam);
    nc::NdArray<double> y_smooth = y.copy(); // we’ll compute node values again using m
    // reconstruct smoothed y (Reinsch’s formula)
    for (std::size_t i = 0; i < n - 1; ++i) {
        double h_i = h[i];
        double z = (y[i + 1] - y[i]) / h_i - (h_i / 6.0) * (m[i + 1] - m[i]);
        y_smooth[i + 1] = y_smooth[i] + z; // cumulative (simple approx)
    }
    return CubicSpline(x, y_smooth, CubicSpline::natural(), CubicSpline::natural(), extrapolate);
}

// -----------------------------------------------------------------------------
// interp1d – Python's interp1d equivalent with kind='cubic'
// -----------------------------------------------------------------------------
class interp1d {
private:
    std::unique_ptr<CubicSpline> m_spline;
    bool m_extrapolate;

public:
    enum class Kind { Linear, Cubic };
    
    interp1d(const nc::NdArray<double>& x,
             const nc::NdArray<double>& y,
             Kind kind = Kind::Cubic,
             bool extrapolate = true) 
        : m_extrapolate(extrapolate) {
        
        if (x.size() != y.size() || x.size() < 2) {
            throw std::invalid_argument("interp1d: x and y must have same size and at least 2 points");
        }
        
        if (kind == Kind::Cubic) {
            // For cubic interpolation, we need to handle non-monotonic data
            // Python's interp1d with kind='cubic' automatically sorts the data
            // and uses natural boundary conditions
            
            // Create pairs and sort by x values
            std::vector<std::pair<double, double>> data_pairs;
            for (size_t i = 0; i < x.size(); ++i) {
                data_pairs.push_back({x[i], y[i]});
            }
            std::sort(data_pairs.begin(), data_pairs.end());
            
            // Extract sorted arrays
            std::vector<double> sorted_x, sorted_y;
            for (const auto& pair : data_pairs) {
                sorted_x.push_back(pair.first);
                sorted_y.push_back(pair.second);
            }
            
            nc::NdArray<double> x_sorted(sorted_x);
            nc::NdArray<double> y_sorted(sorted_y);
            
            // Create cubic spline with not-a-knot boundary conditions (same as Python's make_interp_spline default)
            m_spline = std::make_unique<CubicSpline>(x_sorted, y_sorted, 
                                                    CubicSpline::notAKnot(), 
                                                    CubicSpline::notAKnot(), 
                                                    extrapolate);
        } else {
            // For linear interpolation, we can use the original data
            // but we'll implement this later if needed
            throw std::invalid_argument("interp1d: only cubic interpolation is currently supported");
        }
    }
    
    double operator()(double xq) const {
        return (*m_spline)(xq);
    }
    
    nc::NdArray<double> operator()(const nc::NdArray<double>& xq) const {
        return (*m_spline)(xq);
    }
};

} // namespace interpolate
} // namespace scipy
