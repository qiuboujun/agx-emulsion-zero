#include "density_curves.hpp"
#include <limits>

namespace agx_emulsion {

// ------------------------ internal helpers ------------------------
static inline int as_int(CurveType t) {
    switch (t) {
        case CurveType::Negative: return 0;
        case CurveType::Positive: return 1;
        case CurveType::Paper:    return 2;
    }
    return 0;
}

static std::size_t upper_bound_idx(const std::vector<double>& xs, double x) {
    // first index i such that xs[i] > x (like std::upper_bound)
    std::size_t lo = 0, hi = xs.size();
    while (lo < hi) {
        std::size_t mid = (lo + hi) / 2;
        if (xs[mid] > x) hi = mid; else lo = mid + 1;
    }
    return lo;
}

static inline double lerp(double x0, double x1, double y0, double y1, double x) {
    if (x1 == x0) return y0;
    const double t = (x - x0) / (x1 - x0);
    return y0 + (y1 - y0) * t;
}

// ------------------------ Models (CPU) ------------------------
std::vector<double> density_curve_model_norm_cdfs(
    const std::vector<double>& loge,
    const DensityParams& p,
    CurveType type,
    int number_of_layers)
{
    number_of_layers = std::max(0, std::min(number_of_layers, 3));
    std::vector<double> out(loge.size(), 0.0);

    const bool positive = (type == CurveType::Positive);
    for (int i = 0; i < number_of_layers; ++i) {
        const double c = p.center[i];
        const double a = p.amplitude[i];
        const double s = p.sigma[i] > 0.0 ? p.sigma[i] : std::numeric_limits<double>::min();

        for (std::size_t j = 0; j < loge.size(); ++j) {
            const double z = (loge[j] - c) / s;
            const double z_used = positive ? (-z) : (z);
            out[j] += a * normal_cdf(z_used);
        }
    }
    return out;
}

Matrix distribution_model_norm_cdfs(
    const std::vector<double>& loge,
    const DensityParams& p,
    int number_of_layers)
{
    number_of_layers = std::max(0, std::min(number_of_layers, 3));
    Matrix dist(loge.size(), 3);
    for (std::size_t r = 0; r < loge.size(); ++r)
        for (int c = 0; c < 3; ++c) dist(r, c) = 0.0;

    for (int i = 0; i < number_of_layers; ++i) {
        const double c0 = p.center[i];
        const double a0 = p.amplitude[i];
        const double s0 = p.sigma[i] > 0.0 ? p.sigma[i] : std::numeric_limits<double>::min();
        for (std::size_t r = 0; r < loge.size(); ++r) {
            const double z = (loge[r] - c0) / s0;
            const double v = a0 * normal_pdf(z) / s0; // pdf((x-μ)/σ)/σ
            dist(r, i) += v;
        }
    }
    return dist;
}

Matrix compute_density_curves(
    const std::vector<double>& log_exposure,
    const std::array<DensityParams,3>& params_rgb,
    CurveType type)
{
    Matrix m(log_exposure.size(), 3);
    for (int ch = 0; ch < 3; ++ch) {
        auto col = density_curve_model_norm_cdfs(log_exposure, params_rgb[ch], type, 3);
        for (std::size_t r = 0; r < log_exposure.size(); ++r) m(r, ch) = col[r];
    }
    return m;
}

Matrix interpolate_exposure_to_density(
    const Matrix& log_exposure_rgb,
    const Matrix& density_curves,
    const std::vector<double>& log_exposure,
    const std::array<double,3>& gamma_factor)
{
    if (log_exposure_rgb.cols != 3 || density_curves.cols != 3)
        throw std::invalid_argument("Matrices must have 3 columns for RGB/CMY channels.");
    if (log_exposure.size() != density_curves.rows)
        throw std::invalid_argument("log_exposure length must match density_curves rows.");

    Matrix out(log_exposure_rgb.rows, 3);

    for (int ch = 0; ch < 3; ++ch) {
        const double g = gamma_factor[ch];
        for (std::size_t i = 0; i < log_exposure_rgb.rows; ++i) {
            const double x = log_exposure_rgb(i, ch) / g;
            // clamp to endpoints
            if (x <= log_exposure.front()) {
                out(i, ch) = density_curves(0, ch);
                continue;
            }
            if (x >= log_exposure.back()) {
                out(i, ch) = density_curves(density_curves.rows - 1, ch);
                continue;
            }
            const std::size_t hi = upper_bound_idx(log_exposure, x);
            const std::size_t lo = hi - 1;
            out(i, ch) = lerp(log_exposure[lo], log_exposure[hi],
                              density_curves(lo, ch), density_curves(hi, ch), x);
        }
    }
    return out;
}

bool gpu_interpolate_exposure_to_density(
    const Matrix& log_exposure_rgb,
    const Matrix& density_curves,
    const std::vector<double>& log_exposure,
    const std::array<double,3>& gamma_factor,
    Matrix& out)
{
    // CPU fallback if CUDA TU not used; actual GPU impl in density_curves.cu
    out = interpolate_exposure_to_density(log_exposure_rgb, density_curves, log_exposure, gamma_factor);
    return false;
}

Matrix apply_gamma_shift_correction(
    const std::vector<double>& le,
    const Matrix& dc,
    const std::array<double,3>& gc,
    const std::array<double,3>& les)
{
    if (dc.rows != le.size() || dc.cols != 3)
        throw std::invalid_argument("apply_gamma_shift_correction: shape mismatch.");
    Matrix out(dc.rows, 3);

    // For each channel i: dc_out[:,i] = interp(le, le/gc[i] + les[i], dc[:,i])
    for (int ch = 0; ch < 3; ++ch) {
        std::vector<double> srcx(le.size());
        for (std::size_t i = 0; i < le.size(); ++i) srcx[i] = le[i] / gc[ch] + les[ch];

        for (std::size_t i = 0; i < le.size(); ++i) {
            const double x = le[i];
            // binary search on srcx (assume le is sorted; srcx is monotonic if gc>0)
            if (x <= srcx.front()) {
                out(i, ch) = dc(0, ch);
                continue;
            }
            if (x >= srcx.back()) {
                out(i, ch) = dc(dc.rows - 1, ch);
                continue;
            }
            // find interval [lo, hi] s.t. srcx[lo] <= x < srcx[hi]
            std::size_t lo = 0, hi = srcx.size();
            while (lo < hi) {
                std::size_t mid = (lo + hi) / 2;
                if (srcx[mid] > x) hi = mid; else lo = mid + 1;
            }
            const std::size_t h = lo, l = h - 1;
            out(i, ch) = lerp(srcx[l], srcx[h], dc(l, ch), dc(h, ch), x);
        }
    }
    return out;
}



} // namespace agx_emulsion 