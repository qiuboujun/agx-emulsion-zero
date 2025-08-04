// couplers.cpp
//
// This source file provides a straightforward, reference implementation of
// the DIR coupler algorithms described in ``agx_emulsion/model/couplers.py``.
// No third party numerical libraries are used here, preferring instead to
// implement the necessary operations by hand.  Although the algorithms are
// computationally simple, they are written clearly to assist any port to
// optimised libraries or GPU backends found in ``couplers.cu``.

#include "couplers.hpp"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace agx_emulsion {

// Helper for computing a 1D Gaussian weight between two integer sample
// positions.  The Gaussian is unnormalised; normalisation occurs later.
static inline double gaussian_weight(int diff, double sigma) {
    if (sigma <= 0.0) {
        // Degenerate case; return Kronecker delta
        return (diff == 0) ? 1.0 : 0.0;
    }
    double x = static_cast<double>(diff);
    double exponent = -0.5 * (x / sigma) * (x / sigma);
    return std::exp(exponent);
}

std::array<std::array<double, 3>, 3>
Couplers::compute_dir_couplers_matrix(const std::array<double, 3> &amount_rgb,
                                      double layer_diffusion) {
    std::array<std::array<double, 3>, 3> result{};
    // For each input layer i (row) we compute a Gaussian centred on
    // position i across the three output layers (columns).  This is
    // equivalent to applying a 1D Gaussian filter with sigma
    // ``layer_diffusion`` on an identity matrix under constant boundary
    // conditions.  We then normalise each row to sum to one and scale by
    // the requested amount per channel.
    for (int i = 0; i < 3; ++i) {
        // Compute unnormalised row values
        double sum_w = 0.0;
        for (int j = 0; j < 3; ++j) {
            double w = gaussian_weight(j - i, layer_diffusion);
            result[i][j] = w;
            sum_w += w;
        }
        // Normalise row to sum to one
        if (sum_w > 0.0) {
            for (int j = 0; j < 3; ++j) {
                result[i][j] /= sum_w;
            }
        }
        // Scale by amount per channel
        for (int j = 0; j < 3; ++j) {
            result[i][j] *= amount_rgb[i];
        }
    }
    return result;
}

// Helper to perform one dimensional linear interpolation on arbitrary
// monotonic data.  Input arrays ``x`` and ``y`` must have the same
// length N.  The function returns a new array containing the values of
// the piecewise linear function defined by (x,y) evaluated at the
// positions given in ``x_new``.  If x is not sorted the routine will
// internally sort a copy of the data.
static std::vector<double> interpolate_1d(const std::vector<double> &x,
                                          const std::vector<double> &y,
                                          const std::vector<double> &x_new) {
    const std::size_t N = x.size();
    if (y.size() != N) {
        throw std::invalid_argument("interpolate_1d: x and y vectors must have the same length");
    }
    // Create sorted indices for x
    std::vector<std::size_t> idx(N);
    std::iota(idx.begin(), idx.end(), 0);
    std::sort(idx.begin(), idx.end(), [&](std::size_t a, std::size_t b) {
        return x[a] < x[b];
    });
    // Create sorted x and y copies
    std::vector<double> x_sorted(N);
    std::vector<double> y_sorted(N);
    for (std::size_t i = 0; i < N; ++i) {
        x_sorted[i] = x[idx[i]];
        y_sorted[i] = y[idx[i]];
    }
    // Lambda to perform binary search for interval
    auto find_interval = [&](double t) -> std::size_t {
        // Return index j such that x_sorted[j] <= t <= x_sorted[j+1]
        // for 0 <= j < N-1.  If t <= x_sorted[0] return 0.  If t >=
        // x_sorted[N-1] return N-2.
        if (t <= x_sorted.front()) {
            return 0;
        }
        if (t >= x_sorted.back()) {
            return N - 2;
        }
        std::size_t lo = 0;
        std::size_t hi = N - 1;
        while (lo + 1 < hi) {
            std::size_t mid = (lo + hi) / 2;
            if (t < x_sorted[mid]) {
                hi = mid;
            } else {
                lo = mid;
            }
        }
        return lo;
    };
    // Interpolate each new point
    std::vector<double> result(x_new.size());
    for (std::size_t i = 0; i < x_new.size(); ++i) {
        double t = x_new[i];
        // Handle constant region or equal endpoints gracefully
        if (N == 1) {
            result[i] = y_sorted[0];
            continue;
        }
        if (t <= x_sorted.front()) {
            result[i] = y_sorted.front();
            continue;
        }
        if (t >= x_sorted.back()) {
            result[i] = y_sorted.back();
            continue;
        }
        std::size_t j = find_interval(t);
        double x0 = x_sorted[j];
        double x1 = x_sorted[j + 1];
        double y0 = y_sorted[j];
        double y1 = y_sorted[j + 1];
        if (x1 == x0) {
            // Avoid division by zero; treat as constant segment
            result[i] = y0;
        } else {
            double alpha = (t - x0) / (x1 - x0);
            result[i] = y0 + alpha * (y1 - y0);
        }
    }
    return result;
}

std::vector<std::vector<double>>
Couplers::compute_density_curves_before_dir_couplers(
    const std::vector<std::vector<double>> &density_curves,
    const std::vector<double> &log_exposure,
    const std::array<std::array<double, 3>, 3> &dir_couplers_matrix,
    double high_exposure_couplers_shift) {
    // Validate shapes
    const std::size_t N = density_curves.size();
    if (N == 0 || log_exposure.size() != N) {
        throw std::invalid_argument("compute_density_curves_before_dir_couplers: mismatched shapes");
    }
    // Ensure inner dimension is three
    for (const auto &row : density_curves) {
        if (row.size() != 3) {
            throw std::invalid_argument("compute_density_curves_before_dir_couplers: each row of density_curves must have length 3");
        }
    }
    // Compute maximum per channel (ignore NaN by treating them as negative infinity)
    std::array<double, 3> d_max = {0.0, 0.0, 0.0};
    for (int c = 0; c < 3; ++c) {
        double max_val = -std::numeric_limits<double>::infinity();
        for (std::size_t i = 0; i < N; ++i) {
            double v = density_curves[i][c];
            if (!std::isnan(v) && v > max_val) {
                max_val = v;
            }
        }
        d_max[c] = max_val;
    }
    // Build normalised and shifted copy
    std::vector<std::vector<double>> dc_norm_shift(N, std::vector<double>(3));
    for (std::size_t i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
            double denom = d_max[c];
            double norm = (denom != 0.0) ? density_curves[i][c] / denom : 0.0;
            double shifted = norm + high_exposure_couplers_shift * norm * norm;
            dc_norm_shift[i][c] = shifted;
        }
    }
    // Multiply by the couplers matrix: (N×3) × (3×3) -> (N×3)
    std::vector<std::vector<double>> couplers_amount_curves(N, std::vector<double>(3, 0.0));
    for (std::size_t i = 0; i < N; ++i) {
        for (int m = 0; m < 3; ++m) {
            double acc = 0.0;
            for (int k = 0; k < 3; ++k) {
                acc += dc_norm_shift[i][k] * dir_couplers_matrix[k][m];
            }
            couplers_amount_curves[i][m] = acc;
        }
    }
    // Compute x0 array: log_exposure[:,None] - couplers_amount_curves
    std::vector<std::vector<double>> x0(N, std::vector<double>(3));
    for (std::size_t i = 0; i < N; ++i) {
        for (int c = 0; c < 3; ++c) {
            x0[i][c] = log_exposure[i] - couplers_amount_curves[i][c];
        }
    }
    // For each channel perform interpolation
    std::vector<std::vector<double>> density_curves_corrected(N, std::vector<double>(3, 0.0));
    for (int c = 0; c < 3; ++c) {
        // Extract x and y for this channel
        std::vector<double> x_channel(N);
        std::vector<double> y_channel(N);
        for (std::size_t i = 0; i < N; ++i) {
            x_channel[i] = x0[i][c];
            y_channel[i] = density_curves[i][c];
        }
        // Interpolate to original log_exposure positions
        std::vector<double> interp = interpolate_1d(x_channel, y_channel, log_exposure);
        // Write back to result
        for (std::size_t i = 0; i < N; ++i) {
            density_curves_corrected[i][c] = interp[i];
        }
    }
    return density_curves_corrected;
}

// Compute a separable Gaussian kernel.  The kernel is generated as the
// outer product of a 1D kernel with itself.  The length is determined
// from sigma via the conventional truncation at three standard
// deviations.  The resulting kernel is normalised to sum to one.
static std::vector<std::vector<double>>
make_gaussian_kernel_2d(double sigma) {
    if (sigma <= 0.0) {
        // Degenerate case; return 1x1 kernel
        return {{1.0}};
    }
    // To better approximate SciPy's gaussian_filter, we extend the
    // kernel support to four standard deviations.  SciPy internally
    // truncates the Gaussian at 4 * sigma which yields a closer match
    // than the more common 3 * sigma cutoff.
    int half_size = static_cast<int>(std::ceil(4.0 * sigma));
    int size = 2 * half_size + 1;
    std::vector<double> kernel_1d(size);
    double sum1 = 0.0;
    for (int i = -half_size; i <= half_size; ++i) {
        double w = std::exp(-0.5 * (static_cast<double>(i) * static_cast<double>(i)) / (sigma * sigma));
        kernel_1d[i + half_size] = w;
        sum1 += w;
    }
    // Normalise 1D kernel
    for (double &v : kernel_1d) {
        v /= sum1;
    }
    // Build 2D kernel as outer product
    std::vector<std::vector<double>> kernel_2d(size, std::vector<double>(size));
    for (int y = 0; y < size; ++y) {
        for (int x = 0; x < size; ++x) {
            kernel_2d[y][x] = kernel_1d[y] * kernel_1d[x];
        }
    }
    return kernel_2d;
}

std::vector<std::vector<std::array<double, 3>>>
Couplers::compute_exposure_correction_dir_couplers(
    const std::vector<std::vector<std::array<double, 3>>> &log_raw,
    const std::vector<std::vector<std::array<double, 3>>> &density_cmy,
    const std::array<double, 3> &density_max,
    const std::array<std::array<double, 3>, 3> &dir_couplers_matrix,
    int diffusion_size_pixel,
    double high_exposure_couplers_shift) {
    // Validate shapes
    const std::size_t H = log_raw.size();
    if (H == 0 || H != density_cmy.size()) {
        throw std::invalid_argument("compute_exposure_correction_dir_couplers: mismatched height");
    }
    const std::size_t W = log_raw[0].size();
    if (W == 0) {
        throw std::invalid_argument("compute_exposure_correction_dir_couplers: zero width");
    }
    for (std::size_t y = 0; y < H; ++y) {
        if (log_raw[y].size() != W || density_cmy[y].size() != W) {
            throw std::invalid_argument("compute_exposure_correction_dir_couplers: inconsistent row length");
        }
    }
    // Build normalised density volume
    std::vector<std::vector<std::array<double, 3>>> norm_density(H,
        std::vector<std::array<double, 3>>(W));
    for (std::size_t i = 0; i < H; ++i) {
        for (std::size_t j = 0; j < W; ++j) {
            for (int c = 0; c < 3; ++c) {
                double denom = density_max[c];
                double norm = (denom != 0.0) ? density_cmy[i][j][c] / denom : 0.0;
                norm += high_exposure_couplers_shift * norm * norm;
                norm_density[i][j][c] = norm;
            }
        }
    }
    // Compute per‑pixel correction via matrix multiplication
    std::vector<std::vector<std::array<double, 3>>> log_raw_correction(H,
        std::vector<std::array<double, 3>>(W));
    for (std::size_t i = 0; i < H; ++i) {
        for (std::size_t j = 0; j < W; ++j) {
            for (int m = 0; m < 3; ++m) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k) {
                    acc += norm_density[i][j][k] * dir_couplers_matrix[k][m];
                }
                log_raw_correction[i][j][m] = acc;
            }
        }
    }
    // Apply spatial diffusion if requested
    if (diffusion_size_pixel > 0) {
        double sigma = static_cast<double>(diffusion_size_pixel);
        std::vector<std::vector<double>> kernel = make_gaussian_kernel_2d(sigma);
        int ksize = static_cast<int>(kernel.size());
        int half = ksize / 2;
        // Allocate blurred volume
        std::vector<std::vector<std::array<double, 3>>> blurred(H,
            std::vector<std::array<double, 3>>(W));
        // Convolve for each channel independently
        for (std::size_t y = 0; y < H; ++y) {
            for (std::size_t x = 0; x < W; ++x) {
                // For each channel
                for (int c = 0; c < 3; ++c) {
                    double acc = 0.0;
                    // Iterate over kernel support
                    for (int dy = -half; dy <= half; ++dy) {
                        int ny = static_cast<int>(y) + dy;
                        // Apply reflective padding by repeatedly folding ny
                        // back into the valid range [0,H-1].  This loop
                        // handles arbitrary kernel sizes.
                        while (ny < 0 || ny >= static_cast<int>(H)) {
                            if (ny < 0) {
                                ny = -ny - 1;
                            } else {
                                ny = 2 * static_cast<int>(H) - ny - 1;
                            }
                        }
                        for (int dx = -half; dx <= half; ++dx) {
                            int nx = static_cast<int>(x) + dx;
                            while (nx < 0 || nx >= static_cast<int>(W)) {
                                if (nx < 0) {
                                    nx = -nx - 1;
                                } else {
                                    nx = 2 * static_cast<int>(W) - nx - 1;
                                }
                            }
                            double w = kernel[dy + half][dx + half];
                            acc += log_raw_correction[ny][nx][c] * w;
                        }
                    }
                    blurred[y][x][c] = acc;
                }
            }
        }
        log_raw_correction.swap(blurred);
    }
    // Subtract correction from raw input
    std::vector<std::vector<std::array<double, 3>>> result(H,
        std::vector<std::array<double, 3>>(W));
    for (std::size_t i = 0; i < H; ++i) {
        for (std::size_t j = 0; j < W; ++j) {
            for (int c = 0; c < 3; ++c) {
                result[i][j][c] = log_raw[i][j][c] - log_raw_correction[i][j][c];
            }
        }
    }
    return result;
}

} // namespace agx_emulsion