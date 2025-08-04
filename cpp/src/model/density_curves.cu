// If compiled with a non-CUDA compiler, this file still provides a main()
// and reuses the CPU fallback declared in the header/CPP files.

#include "density_curves.hpp"
#include <iostream>
#include <iomanip>
#include <limits>
#include <cmath>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

namespace agx_emulsion {

#ifdef __CUDACC__
// ------------------------ Device helpers ------------------------
__device__ inline double d_normal_cdf(double z) {
    // 0.5 * (1 + erf(z/sqrt(2)))
    return 0.5 * (1.0 + erf(z / 1.4142135623730951));
}

__global__ void k_norm_cdf_curve(
    const double* loge, int N,
    const double* centers,
    const double* amplitudes,
    const double* sigmas,
    int number_of_layers,
    int curve_type, // 0=Neg,1=Pos,2=Paper (Pos handled like Python sign)
    double* out)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    double v = 0.0;
    for (int i = 0; i < number_of_layers; ++i) {
        const double c = centers[i];
        const double a = amplitudes[i];
        const double s = fmax(sigmas[i], 1e-30);
        const double z = (loge[idx] - c) / s;
        const double z_used = (curve_type == 1) ? (-z) : (z); // Positive flips sign
        v += a * d_normal_cdf(z_used);
    }
    out[idx] = v;
}
#endif // __CUDACC__

bool gpu_density_curve_model_norm_cdfs(
    const std::vector<double>& loge,
    const DensityParams& p,
    CurveType type,
    int number_of_layers,
    std::vector<double>& out)
{
#ifndef __CUDACC__
    // CPU fallback if not compiled with nvcc
    out = density_curve_model_norm_cdfs(loge, p, type, number_of_layers);
    return false;
#else
    const int N = static_cast<int>(loge.size());
    out.assign(N, 0.0);

    double *d_loge=nullptr, *d_centers=nullptr, *d_amps=nullptr, *d_sigmas=nullptr, *d_out=nullptr;
    cudaMalloc(&d_loge,   N * sizeof(double));
    cudaMalloc(&d_centers, 3 * sizeof(double));
    cudaMalloc(&d_amps,    3 * sizeof(double));
    cudaMalloc(&d_sigmas,  3 * sizeof(double));
    cudaMalloc(&d_out,     N * sizeof(double));

    cudaMemcpy(d_loge,   loge.data(),           N * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_centers, p.center.data(),    3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_amps,    p.amplitude.data(), 3 * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sigmas,  p.sigma.data(),     3 * sizeof(double), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks  = (N + threads - 1) / threads;

    k_norm_cdf_curve<<<blocks, threads>>>(
        d_loge, N, d_centers, d_amps, d_sigmas,
        std::max(0, std::min(number_of_layers, 3)),
        (type == CurveType::Positive) ? 1 : 0,
        d_out);
    cudaDeviceSynchronize();

    cudaMemcpy(out.data(), d_out, N * sizeof(double), cudaMemcpyDeviceToHost);

    cudaFree(d_loge);
    cudaFree(d_centers);
    cudaFree(d_amps);
    cudaFree(d_sigmas);
    cudaFree(d_out);
    return true;
#endif
}

} // namespace agx_emulsion 