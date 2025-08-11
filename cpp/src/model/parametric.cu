// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include "parametric.hpp"

namespace agx {
namespace model {

__global__ void parametric_kernel(
    const float* __restrict__ le, int N,
    float g0, float g1, float g2,
    float le0_0, float le0_1, float le0_2,
    float dmax0, float dmax1, float dmax2,
    float ts0, float ts1, float ts2,
    float ss0, float ss1, float ss2,
    float* __restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    float le_j = le[j];

    auto eval = [&](float g, float le0, float dmax, float ts, float ss) {
        float a = g * ts * log10f(1.0f + powf(10.0f, (le_j - le0) / ts));
        float b = g * ss * log10f(1.0f + powf(10.0f, (le_j - le0 - dmax / g) / ss));
        return a - b;
    };

    out[j * 3 + 0] = eval(g0, le0_0, dmax0, ts0, ss0);
    out[j * 3 + 1] = eval(g1, le0_1, dmax1, ts1, ss1);
    out[j * 3 + 2] = eval(g2, le0_2, dmax2, ts2, ss2);
}

bool parametric_density_curves_model_cuda(
    const nc::NdArray<float>& log_exposure,
    const std::array<float, 3>& gamma,
    const std::array<float, 3>& log_exposure_0,
    const std::array<float, 3>& density_max,
    const std::array<float, 3>& toe_size,
    const std::array<float, 3>& shoulder_size,
    nc::NdArray<float>& out_density_curves) {
    const int N = static_cast<int>(log_exposure.size());
    if (N <= 0) return true;

    float *d_le = nullptr, *d_out = nullptr;
    if (cudaMalloc(&d_le, N * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_out, N * 3 * sizeof(float)) != cudaSuccess) { cudaFree(d_le); return false; }

    cudaMemcpy(d_le, log_exposure.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);
    parametric_kernel<<<grid, block>>>(
        d_le, N,
        gamma[0], gamma[1], gamma[2],
        log_exposure_0[0], log_exposure_0[1], log_exposure_0[2],
        density_max[0], density_max[1], density_max[2],
        toe_size[0], toe_size[1], toe_size[2],
        shoulder_size[0], shoulder_size[1], shoulder_size[2],
        d_out);
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d_le); cudaFree(d_out); return false; }

    // Ensure out array shape
    if (out_density_curves.shape().rows != static_cast<size_t>(N) || out_density_curves.shape().cols != 3) {
        out_density_curves = nc::NdArray<float>(N, 3);
    }
    cudaMemcpy(out_density_curves.data(), d_out, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_le);
    cudaFree(d_out);
    return true;
}

} // namespace model
} // namespace agx


