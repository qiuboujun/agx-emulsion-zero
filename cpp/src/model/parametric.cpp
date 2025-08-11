// SPDX-License-Identifier: MIT

#include "parametric.hpp"
#include <cmath>

namespace agx {
namespace model {

static inline float log10_safe(float x) {
    return std::log10(x);
}

nc::NdArray<float> parametric_density_curves_model(
    const nc::NdArray<float>& log_exposure,
    const std::array<float, 3>& gamma,
    const std::array<float, 3>& log_exposure_0,
    const std::array<float, 3>& density_max,
    const std::array<float, 3>& toe_size,
    const std::array<float, 3>& shoulder_size) {

    auto le = log_exposure.flatten();
    const size_t N = le.size();
    nc::NdArray<float> out(N, 3);

    for (int i = 0; i < 3; ++i) {
        const float g = gamma[i];
        const float loge0 = log_exposure_0[i];
        const float dmax = density_max[i];
        const float ts = toe_size[i];
        const float ss = shoulder_size[i];
        for (size_t j = 0; j < N; ++j) {
            const float le_j = le[j];
            const float a = g * ts * log10_safe(1.0f + std::pow(10.0f, (le_j - loge0) / ts));
            const float b = g * ss * log10_safe(1.0f + std::pow(10.0f, (le_j - loge0 - dmax / g) / ss));
            out(j, i) = a - b;
        }
    }
    return out;
}

bool parametric_density_curves_model_cuda(
    const nc::NdArray<float>& log_exposure,
    const std::array<float, 3>& gamma,
    const std::array<float, 3>& log_exposure_0,
    const std::array<float, 3>& density_max,
    const std::array<float, 3>& toe_size,
    const std::array<float, 3>& shoulder_size,
    nc::NdArray<float>& out_density_curves) {
    // CPU fallback placeholder; CUDA kernel provided in parametric.cu
    (void)log_exposure; (void)gamma; (void)log_exposure_0; (void)density_max; (void)toe_size; (void)shoulder_size;
    (void)out_density_curves;
    return false;
}

nc::NdArray<float> parametric_density_curves_model_auto(
    const nc::NdArray<float>& log_exposure,
    const std::array<float, 3>& gamma,
    const std::array<float, 3>& log_exposure_0,
    const std::array<float, 3>& density_max,
    const std::array<float, 3>& toe_size,
    const std::array<float, 3>& shoulder_size,
    bool use_cuda) {
    if (use_cuda) {
        nc::NdArray<float> out(log_exposure.size(), 3);
        if (parametric_density_curves_model_cuda(log_exposure.flatten(), gamma, log_exposure_0, density_max, toe_size, shoulder_size, out)) {
            return out;
        }
    }
    return parametric_density_curves_model(log_exposure, gamma, log_exposure_0, density_max, toe_size, shoulder_size);
}

} // namespace model
} // namespace agx


