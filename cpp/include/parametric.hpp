// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include "NumCpp.hpp"

namespace agx {
namespace model {

// CPU implementation
nc::NdArray<float> parametric_density_curves_model(
    const nc::NdArray<float>& log_exposure,            // (N,1) or (N)
    const std::array<float, 3>& gamma,
    const std::array<float, 3>& log_exposure_0,
    const std::array<float, 3>& density_max,
    const std::array<float, 3>& toe_size,
    const std::array<float, 3>& shoulder_size);

// CUDA implementation (returns true if succeeded and fills out)
bool parametric_density_curves_model_cuda(
    const nc::NdArray<float>& log_exposure,            // (N)
    const std::array<float, 3>& gamma,
    const std::array<float, 3>& log_exposure_0,
    const std::array<float, 3>& density_max,
    const std::array<float, 3>& toe_size,
    const std::array<float, 3>& shoulder_size,
    nc::NdArray<float>& out_density_curves);           // (N,3)

// Convenience wrapper: tries CUDA, falls back to CPU
nc::NdArray<float> parametric_density_curves_model_auto(
    const nc::NdArray<float>& log_exposure,
    const std::array<float, 3>& gamma,
    const std::array<float, 3>& log_exposure_0,
    const std::array<float, 3>& density_max,
    const std::array<float, 3>& toe_size,
    const std::array<float, 3>& shoulder_size,
    bool use_cuda = true);

} // namespace model
} // namespace agx


