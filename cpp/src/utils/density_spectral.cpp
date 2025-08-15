// SPDX-License-Identifier: MIT

#include "density_spectral.hpp"

namespace agx { namespace utils {

nc::NdArray<float> compute_density_spectral(
    const nc::NdArray<float>& density_cmy,
    const nc::NdArray<float>& dye_density,
    float dye_density_min_factor) {
    // density_cmy: (H, W, 3)
    // dye_density: (K, C) with [:,0:3] spectral dye densities and optional base in [:,3]
    const int H = (int)density_cmy.shape().rows;
    const int W3 = (int)density_cmy.shape().cols;
    if (W3 % 3 != 0) throw std::runtime_error("density_cmy last dim not multiple of 3");
    const int W = W3 / 3;
    const int K = (int)dye_density.shape().rows;

    // Prepare output (H, W, K)
    nc::NdArray<float> out(H, W * K);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            // Gather CMY per pixel
            const double c = static_cast<double>(density_cmy(y, x * 3 + 0));
            const double m = static_cast<double>(density_cmy(y, x * 3 + 1));
            const double yv = static_cast<double>(density_cmy(y, x * 3 + 2));
            const int cols = (int)dye_density.shape().cols;
            for (int k = 0; k < K; ++k) {
                double spec = c * static_cast<double>(dye_density(k, 0))
                            + m * static_cast<double>(dye_density(k, 1))
                            + yv * static_cast<double>(dye_density(k, 2));
                if (cols > 3) {
                    spec += static_cast<double>(dye_density(k, 3)) * static_cast<double>(dye_density_min_factor);
                }
                out(y, x * K + k) = static_cast<float>(spec);
            }
        }
    }
    return out;
}

#if !defined(AGX_WITH_CUDA)
bool compute_density_spectral_gpu(
    const nc::NdArray<float>& density_cmy,
    const nc::NdArray<float>& dye_density,
    float dye_density_min_factor,
    nc::NdArray<float>& out) {
    // No CUDA build: use CPU path and report false
    out = compute_density_spectral(density_cmy, dye_density, dye_density_min_factor);
    return false;
}
#endif

}} // namespace agx::utils


