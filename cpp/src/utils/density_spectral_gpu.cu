// SPDX-License-Identifier: MIT

#include "density_spectral.hpp"
#include <cuda_runtime.h>

namespace agx { namespace utils {

// Kernel: compute spectral density per pixel
__global__ void k_density_spectral(const float* cmy, const float* dd, float* out,
                                   int H, int W, int K, int cols, float base)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = H * W * K;
    if (idx >= N) return;

    int tmp = idx;
    int k = tmp % K; tmp /= K;
    int x = tmp % W; tmp /= W;
    int y = tmp;

    float c = cmy[(y * W + x) * 3 + 0];
    float m = cmy[(y * W + x) * 3 + 1];
    float yv = cmy[(y * W + x) * 3 + 2];

    float spec = c * dd[k * cols + 0] + m * dd[k * cols + 1] + yv * dd[k * cols + 2];
    if (cols > 3) spec += dd[k * cols + 3] * base;

    out[y * (W * K) + x * K + k] = spec;
}

bool compute_density_spectral_gpu(
    const nc::NdArray<float>& density_cmy,
    const nc::NdArray<float>& dye_density,
    float dye_density_min_factor,
    nc::NdArray<float>& out)
{
    const int H = (int)density_cmy.shape().rows;
    const int W3 = (int)density_cmy.shape().cols;
    if (W3 % 3 != 0) return false;
    const int W = W3 / 3;
    const int K = (int)dye_density.shape().rows;
    const int cols = (int)dye_density.shape().cols;

    out = nc::NdArray<float>(H, W * K);

    // Flatten and allocate
    auto cmy = density_cmy.flatten();
    auto dd = dye_density.flatten();
    auto o = out.flatten();

    float *d_cmy = nullptr, *d_dd = nullptr, *d_o = nullptr;
    cudaError_t err;
    err = cudaMalloc(&d_cmy, cmy.size() * sizeof(float)); if (err != cudaSuccess) return false;
    err = cudaMalloc(&d_dd, dd.size() * sizeof(float));   if (err != cudaSuccess) { cudaFree(d_cmy); return false; }
    err = cudaMalloc(&d_o,  o.size() * sizeof(float));    if (err != cudaSuccess) { cudaFree(d_cmy); cudaFree(d_dd); return false; }

    cudaMemcpy(d_cmy, cmy.data(), cmy.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_dd,  dd.data(),  dd.size()  * sizeof(float), cudaMemcpyHostToDevice);

    int N = H * W * K;
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    k_density_spectral<<<blocks, threads>>>(d_cmy, d_dd, d_o, H, W, K, cols, dye_density_min_factor);
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(d_cmy); cudaFree(d_dd); cudaFree(d_o);
        return false;
    }

    cudaMemcpy(o.data(), d_o, o.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_cmy); cudaFree(d_dd); cudaFree(d_o);
    return true;
}

bool compute_density_spectral_cuda(
    const nc::NdArray<float>& density_cmy,
    const nc::NdArray<float>& dye_density,
    float dye_density_min_factor,
    nc::NdArray<float>& out)
{
    // Just call GPU version and require success
    return compute_density_spectral_gpu(density_cmy, dye_density, dye_density_min_factor, out);
}

} } // namespace agx::utils


