// SPDX-License-Identifier: MIT

#pragma once

#include "NumCpp.hpp"

namespace agx {
namespace utils {

// Compute spectral density from CMY dye densities and spectral dye-density curves
// density_cmy: (H, W, 3)
// dye_density: (K, 4) where [:,0:3] are spectral dye densities per channel and [:,3] is base density
// dye_density_min_factor: scalar factor applied to base dye density term
// Returns: (H, W, K)
nc::NdArray<float> compute_density_spectral(
    const nc::NdArray<float>& density_cmy,
    const nc::NdArray<float>& dye_density,
    float dye_density_min_factor);

// GPU-accelerated version with CPU fallback. Returns true if GPU ran.
bool compute_density_spectral_gpu(
    const nc::NdArray<float>& density_cmy,
    const nc::NdArray<float>& dye_density,
    float dye_density_min_factor,
    nc::NdArray<float>& out);

} // namespace utils
} // namespace agx


