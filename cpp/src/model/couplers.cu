// couplers.cu
//
// This CUDA source file provides optional GPU accelerated versions of
// the DIR coupler functions defined in ``couplers.hpp``.  When
// compiled with NVCC it will produce device code capable of running
// on a CUDA enabled GPU.  The CPU reference implementations live in
// ``couplers.cpp``; should a GPU not be available these versions may
// simply forward to the CPU code.  The kernels here follow a naive
// implementation for clarity rather than performance.

#include "couplers.hpp"

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <cstdio>
#endif

namespace agx_emulsion {

// The GPU implementation of ``compute_dir_couplers_matrix`` is
// identical to the CPU version since the operation is trivial
// (three rows of three coefficients).  We simply defer to the host
// implementation.
std::array<std::array<double, 3>, 3>
compute_dir_couplers_matrix_cuda(const std::array<double, 3> &amount_rgb,
                                 double layer_diffusion) {
    return Couplers::compute_dir_couplers_matrix(amount_rgb, layer_diffusion);
}

// Similarly, density curve inversion is performed on the host.  The
// per‑channel interpolation does not benefit from GPU parallelism at
// the scale of typical film profiles (on the order of a few dozen
// samples).  Should you wish to accelerate this further you could
// parallelise the outer loop over colour channels.
std::vector<std::vector<double>>
compute_density_curves_before_dir_couplers_cuda(
    const std::vector<std::vector<double>> &density_curves,
    const std::vector<double> &log_exposure,
    const std::array<std::array<double, 3>, 3> &dir_couplers_matrix,
    double high_exposure_couplers_shift = 0.0) {
    return Couplers::compute_density_curves_before_dir_couplers(
        density_curves, log_exposure, dir_couplers_matrix,
        high_exposure_couplers_shift);
}

// A naive CUDA kernel for performing 2D Gaussian convolution on a
// single colour channel.  Each thread processes one output pixel.
#ifdef __CUDACC__
__global__ void gaussian_blur_kernel(const double *input, double *output,
                                     int width, int height,
                                     const double *kernel, int ksize) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half = ksize / 2;
    if (x >= width || y >= height) return;
    double acc = 0.0;
    for (int dy = -half; dy <= half; ++dy) {
        int yy = y + dy;
        if (yy < 0 || yy >= height) continue;
        for (int dx = -half; dx <= half; ++dx) {
            int xx = x + dx;
            if (xx < 0 || xx >= width) continue;
            double w = kernel[(dy + half) * ksize + (dx + half)];
            acc += input[yy * width + xx] * w;
        }
    }
    output[y * width + x] = acc;
}
#endif

// GPU accelerated exposure correction.  If CUDA support is not
// available this function defers to the CPU implementation.  The
// signature matches that of the CPU function.  To enable this
// implementation compile with nvcc and define __CUDACC__.
std::vector<std::vector<std::array<double, 3>>>
compute_exposure_correction_dir_couplers_cuda(
    const std::vector<std::vector<std::array<double, 3>>> &log_raw,
    const std::vector<std::vector<std::array<double, 3>>> &density_cmy,
    const std::array<double, 3> &density_max,
    const std::array<std::array<double, 3>, 3> &dir_couplers_matrix,
    double diffusion_size_pixel,
    double high_exposure_couplers_shift) {
#ifndef __CUDACC__
    throw std::runtime_error("compute_exposure_correction_dir_couplers_cuda requires CUDA");
#else
    // The GPU implementation follows the same steps as the CPU
    // reference: compute per‑pixel inhibitors then optionally blur
    // spatially and subtract from the raw input.  For brevity we
    // implement the blur using a separable kernel but do not exploit
    // shared memory or constant memory; such optimisations are left to
    // the reader.
    const int H = static_cast<int>(log_raw.size());
    const int W = static_cast<int>(log_raw[0].size());
    // Prepare host buffers
    std::vector<std::vector<std::array<double, 3>>> norm_density(H,
        std::vector<std::array<double, 3>>(W));
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int c = 0; c < 3; ++c) {
                double denom = density_max[c];
                double norm = (denom != 0.0) ? density_cmy[i][j][c] / denom : 0.0;
                norm += high_exposure_couplers_shift * norm * norm;
                norm_density[i][j][c] = norm;
            }
        }
    }
    // Compute per pixel correction via matrix multiplication on host
    std::vector<std::vector<std::array<double, 3>>> corr(H,
        std::vector<std::array<double, 3>>(W));
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int m = 0; m < 3; ++m) {
                double acc = 0.0;
                for (int k = 0; k < 3; ++k) {
                    acc += norm_density[i][j][k] * dir_couplers_matrix[k][m];
                }
                corr[i][j][m] = acc;
            }
        }
    }
    // If no diffusion requested we can subtract directly
    if (diffusion_size_pixel <= 0.0) {
        std::vector<std::vector<std::array<double, 3>>> result(H,
            std::vector<std::array<double, 3>>(W));
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                for (int c = 0; c < 3; ++c) {
                    result[i][j][c] = log_raw[i][j][c] - corr[i][j][c];
                }
            }
        }
        return result;
    }
    // Build Gaussian kernel on host
    double sigma = diffusion_size_pixel;
    // Create the kernel manually since make_gaussian_kernel_2d is not accessible from CUDA
    std::vector<std::vector<double>> kernel2d;
    if (sigma <= 0.0) {
        kernel2d = {{1.0}};
    } else {
        int half_size = static_cast<int>(std::ceil(4.0 * sigma));
        int size = 2 * half_size + 1;
        std::vector<double> kernel_1d(size);
        double sum1 = 0.0;
        for (int i = -half_size; i <= half_size; ++i) {
            double w = std::exp(-0.5 * (static_cast<double>(i) * static_cast<double>(i)) / (sigma * sigma));
            kernel_1d[i + half_size] = w;
            sum1 += w;
        }
        // Normalise 1D kernel
        for (double &v : kernel_1d) {
            v /= sum1;
        }
        // Build 2D kernel as outer product
        kernel2d.resize(size, std::vector<double>(size));
        for (int y = 0; y < size; ++y) {
            for (int x = 0; x < size; ++x) {
                kernel2d[y][x] = kernel_1d[y] * kernel_1d[x];
            }
        }
    }
    int ksize = static_cast<int>(kernel2d.size());
    int kernel_size_sq = ksize * ksize;
    // Flatten kernel into 1D for device transfer
    std::vector<double> kernel_flat(kernel_size_sq);
    for (int y = 0; y < ksize; ++y) {
        for (int x = 0; x < ksize; ++x) {
            kernel_flat[y * ksize + x] = kernel2d[y][x];
        }
    }
    // Allocate device buffers for one channel at a time
    double *d_input = nullptr;
    double *d_output = nullptr;
    double *d_kernel = nullptr;
    cudaMalloc(&d_input, sizeof(double) * H * W);
    cudaMalloc(&d_output, sizeof(double) * H * W);
    cudaMalloc(&d_kernel, sizeof(double) * kernel_size_sq);
    cudaMemcpy(d_kernel, kernel_flat.data(), sizeof(double) * kernel_size_sq, cudaMemcpyHostToDevice);
    // Output volume
    std::vector<std::vector<std::array<double, 3>>> blurred(H,
        std::vector<std::array<double, 3>>(W));
    // Configure kernel launch
    dim3 block(16, 16);
    dim3 grid((W + block.x - 1) / block.x, (H + block.y - 1) / block.y);
    for (int c = 0; c < 3; ++c) {
        // Copy input channel to device
        std::vector<double> channel_in(H * W);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                channel_in[y * W + x] = corr[y][x][c];
            }
        }
        cudaMemcpy(d_input, channel_in.data(), sizeof(double) * H * W, cudaMemcpyHostToDevice);
        // Launch kernel
        gaussian_blur_kernel<<<grid, block>>>(d_input, d_output, W, H, d_kernel, ksize);
        cudaDeviceSynchronize();
        // Copy result back
        std::vector<double> channel_out(H * W);
        cudaMemcpy(channel_out.data(), d_output, sizeof(double) * H * W, cudaMemcpyDeviceToHost);
        // Store into blurred volume
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                blurred[y][x][c] = channel_out[y * W + x];
            }
        }
    }
    // Free device buffers
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_kernel);
    // Subtract from raw
    std::vector<std::vector<std::array<double, 3>>> result(H,
        std::vector<std::array<double, 3>>(W));
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            for (int c = 0; c < 3; ++c) {
                result[i][j][c] = log_raw[i][j][c] - blurred[i][j][c];
            }
        }
    }
    return result;
#endif
}

} // namespace agx_emulsion