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

bool compute_density_spectral_gpu(
    const nc::NdArray<float>& density_cmy,
    const nc::NdArray<float>& dye_density,
    float dye_density_min_factor,
    nc::NdArray<float>& out) {
#ifndef __CUDACC__
    out = compute_density_spectral(density_cmy, dye_density, dye_density_min_factor);
    return false;
#else
    const int H = (int)density_cmy.shape().rows;
    const int W3 = (int)density_cmy.shape().cols;
    const int W = W3 / 3;
    const int K = (int)dye_density.shape().rows;
    out = nc::NdArray<float>(H, W*K);

    // Flatten buffers and copy to device
    auto cmy = density_cmy.flatten();
    auto dd = dye_density.flatten();
    auto o = out.flatten();
    float *d_cmy=nullptr, *d_dd=nullptr, *d_o=nullptr;
    cudaMalloc(&d_cmy, cmy.size()*sizeof(float));
    cudaMalloc(&d_dd, dd.size()*sizeof(float));
    cudaMalloc(&d_o,  o.size()*sizeof(float));
    cudaMemcpy(d_cmy, cmy.data(), cmy.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dd,  dd.data(),  dd.size()*sizeof(float),  cudaMemcpyHostToDevice);

    struct Params { int H,W,K,cols; float base; } p{H,W,K,(int)dye_density.shape().cols,dye_density_min_factor};
    int N = H*W*K;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    __global__ void k_cd(const float* cmy, const float* dd, float* out, Params p) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx >= p.H*p.W*p.K) return;
        int tmp = idx;
        int k = tmp % p.K; tmp /= p.K;
        int x = tmp % p.W; tmp /= p.W;
        int y = tmp;
        float c = cmy[(y*p.W + x)*3 + 0];
        float m = cmy[(y*p.W + x)*3 + 1];
        float yv= cmy[(y*p.W + x)*3 + 2];
        float spec = c * dd[k*p.cols + 0] + m * dd[k*p.cols + 1] + yv * dd[k*p.cols + 2];
        if (p.cols > 3) {
            spec += dd[k*p.cols + 3] * p.base;
        }
        out[y*(p.W*p.K) + x*p.K + k] = spec;
    }
    k_cd<<<blocks, threads>>>(d_cmy, d_dd, d_o, p);
    cudaDeviceSynchronize();
    cudaMemcpy(o.data(), d_o, o.size()*sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_cmy); cudaFree(d_dd); cudaFree(d_o);
    return true;
#endif
}

}} // namespace agx::utils


