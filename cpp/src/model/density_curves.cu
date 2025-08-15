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

#ifdef __CUDACC__
// Free function kernel for interpolation (declare before use)
__global__ void k_interp_density(const double* loge_rgb, int P,
                                 const double* le, int N, const double* dc,
                                 double* out) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= P) return;
    for (int c=0;c<3;++c) {
        double x = loge_rgb[idx*3 + c];
        if (x <= le[0]) { out[idx*3+c] = dc[c]; continue; }
        if (x >= le[N-1]) { out[idx*3+c] = dc[(N-1)*3 + c]; continue; }
        int lo = 0, hi = N;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (le[mid] > x) hi = mid; else lo = mid + 1;
        }
        int h = lo, l = h - 1;
        double x0 = le[l], x1 = le[h];
        double t = (x - x0) / (x1 - x0);
        double y0 = dc[l*3 + c];
        double y1 = dc[h*3 + c];
        out[idx*3 + c] = y0 + t * (y1 - y0);
    }
}
#endif

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

// GPU version of interpolate_exposure_to_density: per-channel 1D linear interp
bool gpu_interpolate_exposure_to_density(
    const Matrix& log_exposure_rgb,
    const Matrix& density_curves,
    const std::vector<double>& log_exposure,
    const std::array<double,3>& gamma_factor,
    Matrix& out)
{
#ifndef __CUDACC__
    // Fallback: run CPU and return false
    out = interpolate_exposure_to_density(log_exposure_rgb, density_curves, log_exposure, gamma_factor);
    return false;
#else
    const int P = static_cast<int>(log_exposure_rgb.rows);
    const int N = static_cast<int>(log_exposure.size());
    if (density_curves.rows != static_cast<std::size_t>(N) || density_curves.cols != 3) {
        throw std::invalid_argument("gpu_interpolate_exposure_to_density: shape mismatch");
    }

    out = Matrix(log_exposure_rgb.rows, 3);

    // Copy host data to device
    double* d_loge_rgb = nullptr; // P x 3
    double* d_le = nullptr;       // N
    double* d_dc = nullptr;       // N x 3
    double* d_out = nullptr;      // P x 3
    cudaMalloc(&d_loge_rgb, P * 3 * sizeof(double));
    cudaMalloc(&d_le,       N * sizeof(double));
    cudaMalloc(&d_dc,       N * 3 * sizeof(double));
    cudaMalloc(&d_out,      P * 3 * sizeof(double));

    // Pack inputs into temporary doubles
    std::vector<double> h_loge_rgb(P*3);
    for (int i=0;i<P;++i) { for (int c=0;c<3;++c) h_loge_rgb[i*3+c] = log_exposure_rgb(i,c) / gamma_factor[c]; }
    std::vector<double> h_le(N);
    for (int i=0;i<N;++i) h_le[i] = log_exposure[i];
    std::vector<double> h_dc(N*3);
    for (int i=0;i<N;++i) for (int c=0;c<3;++c) h_dc[i*3+c] = density_curves(i,c);

    cudaMemcpy(d_loge_rgb, h_loge_rgb.data(), P*3*sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_le,       h_le.data(),       N*sizeof(double),   cudaMemcpyHostToDevice);
    cudaMemcpy(d_dc,       h_dc.data(),       N*3*sizeof(double), cudaMemcpyHostToDevice);

    const int threads = 256;
    const int blocks = (P + threads - 1) / threads;
    k_interp_density<<<blocks, threads>>>(d_loge_rgb, P, d_le, N, d_dc, d_out);
    cudaDeviceSynchronize();

    // Copy back
    std::vector<double> h_out(P*3);
    cudaMemcpy(h_out.data(), d_out, P*3*sizeof(double), cudaMemcpyDeviceToHost);
    for (int i=0;i<P;++i) for (int c=0;c<3;++c) out(i,c) = h_out[i*3 + c];

    cudaFree(d_loge_rgb); cudaFree(d_le); cudaFree(d_dc); cudaFree(d_out);
    return true;
#endif
}

// Enforced GPU variant: throws if CUDA not available
bool gpu_interpolate_exposure_to_density_cuda(
    const Matrix& log_exposure_rgb,
    const Matrix& density_curves,
    const std::vector<double>& log_exposure,
    const std::array<double,3>& gamma_factor,
    Matrix& out)
{
#ifndef __CUDACC__
    throw std::runtime_error("gpu_interpolate_exposure_to_density_cuda requires CUDA");
#else
    return gpu_interpolate_exposure_to_density(log_exposure_rgb, density_curves, log_exposure, gamma_factor, out);
#endif
}
} // namespace agx_emulsion 