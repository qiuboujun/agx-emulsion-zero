#pragma once
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>
#include <algorithm>
#include <stdexcept>

namespace agx_emulsion {

enum class CurveType { Negative, Positive, Paper };

// Parameters per channel (up to 3 "layers" as in the Python default)
struct DensityParams {
    std::array<double,3> center{0.0, 1.0, 2.0};
    std::array<double,3> amplitude{0.5, 0.5, 0.5};
    std::array<double,3> sigma{0.3, 0.5, 0.7};
};

// Lightweight row-major matrix for doubles
struct Matrix {
    std::size_t rows{0}, cols{0};
    std::vector<double> data;
    Matrix() = default;
    Matrix(std::size_t r, std::size_t c) : rows(r), cols(c), data(r*c, 0.0) {}
    double& operator()(std::size_t r, std::size_t c) { return data[r*cols + c]; }
    const double& operator()(std::size_t r, std::size_t c) const { return data[r*cols + c]; }
};

// ------------------------ Math helpers ------------------------
inline double normal_cdf(double z) {
    // Standard normal CDF Φ(z) via erf
    return 0.5 * (1.0 + std::erf(z / std::sqrt(2.0)));
}
inline double normal_pdf(double z) {
    static constexpr double inv_sqrt_2pi = 0.39894228040143267794; // 1/sqrt(2π)
    return inv_sqrt_2pi * std::exp(-0.5*z*z);
}

// ------------------------ Models (CPU) ------------------------
/**
 * density_curve_model_norm_cdfs:
 * Mirrors Python's:
 *   scipy.stats.norm.cdf((loge - center)/sigma)*amplitude
 * If type == Positive, the sign inside CDF is flipped like Python.
 */
std::vector<double> density_curve_model_norm_cdfs(
    const std::vector<double>& loge,
    const DensityParams& p,
    CurveType type = CurveType::Negative,
    int number_of_layers = 3);

/**
 * distribution_model_norm_cdfs:
 * Returns Nx3 matrix where each column i is amplitude[i]*pdf((loge-center[i])/sigma[i]).
 */
Matrix distribution_model_norm_cdfs(
    const std::vector<double>& loge,
    const DensityParams& p,
    int number_of_layers = 3);

/**
 * compute_density_curves:
 * Given three per-channel parameter sets, compute Nx3 density matrix.
 * model currently supports "norm_cdfs".
 */
Matrix compute_density_curves(
    const std::vector<double>& log_exposure,
    const std::array<DensityParams,3>& params_rgb,
    CurveType type = CurveType::Negative);

// -------------------- Interpolation & adjustments --------------------
/**
 * Linear interpolation channel-wise:
 * For each channel c, given an input value x, find y by linearly interpolating
 * (log_exposure/gamma_factor[c] -> density_curves[:,c]).
 * log_exposure_rgb is an HxW x 3 (flattened) matrix for simplicity.
 */
Matrix interpolate_exposure_to_density(
    const Matrix& log_exposure_rgb,         // (H*W) x 3
    const Matrix& density_curves,           // N x 3 (curves per channel)
    const std::vector<double>& log_exposure,// length N
    const std::array<double,3>& gamma_factor);

/**
 * apply_gamma_shift_correction:
 * For each channel i:
 *   dc_out[:,i] = interp(le, le/gamma_correction[i] + log_exposure_correction[i], dc[:,i])
 * where interp is 1D linear interpolation w.r.t le.
 */
Matrix apply_gamma_shift_correction(
    const std::vector<double>& log_exposure,     // le
    const Matrix& density_curves,                // dc (N x 3)
    const std::array<double,3>& gamma_correction,// gc
    const std::array<double,3>& log_exp_correction); // les

// ------------------------ GPU entry points ------------------------
/**
 * gpu_density_curve_model_norm_cdfs:
 * If CUDA is available (compiled with nvcc), run on device; otherwise falls back to CPU.
 * Returns true if GPU path executed, false if CPU fallback was used.
 */
bool gpu_density_curve_model_norm_cdfs(
    const std::vector<double>& loge,
    const DensityParams& p,
    CurveType type,
    int number_of_layers,
    std::vector<double>& out);

} // namespace agx_emulsion 