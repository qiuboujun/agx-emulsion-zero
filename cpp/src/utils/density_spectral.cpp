// SPDX-License-Identifier: MIT

#include "density_spectral.hpp"

namespace agx { namespace utils {

nc::NdArray<float> compute_density_spectral(
    const nc::NdArray<float>& density_cmy,
    const nc::NdArray<float>& dye_density,
    float dye_density_min_factor) {
    // density_cmy: (H, W, 3)
    // dye_density: (K, 4) with [:,0:3] spectral dye densities and [:,3] base
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
            float c = density_cmy(y, x * 3 + 0);
            float m = density_cmy(y, x * 3 + 1);
            float yv = density_cmy(y, x * 3 + 2);
            for (int k = 0; k < K; ++k) {
                float spec = c * dye_density(k, 0) + m * dye_density(k, 1) + yv * dye_density(k, 2);
                spec += dye_density(k, 3) * dye_density_min_factor;
                out(y, x * K + k) = spec;
            }
        }
    }
    return out;
}

}} // namespace agx::utils


