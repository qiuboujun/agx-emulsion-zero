// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include "autoexposure.hpp"

namespace agx {
namespace utils {

// Simple CUDA reduction computing center-weighted mean of Y channel.
// For brevity, we assume image is small/medium and use a naive two-pass approach.

__global__ void compute_weights_kernel(int H, int W, float sx, float sy, float sigma, float* w) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;
    float xx = (float(x) / W) - 0.5f; xx *= sx;
    float yy = (float(y) / H) - 0.5f; yy *= sy;
    w[y * W + x] = __expf(-(xx * xx + yy * yy) / (2.0f * sigma * sigma));
}

__global__ void sum_weighted_kernel(const float* Y, const float* w, int N, float* out_sum_w, float* out_sum_yw) {
    __shared__ float sw[256];
    __shared__ float syw[256];
    int tid = threadIdx.x;
    float accw = 0.0f;
    float accyw = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < N; i += gridDim.x * blockDim.x) {
        float wi = w[i];
        accw += wi;
        accyw += wi * Y[i];
    }
    sw[tid] = accw; syw[tid] = accyw;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) { sw[tid] += sw[tid + s]; syw[tid] += syw[tid + s]; }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out_sum_w, sw[0]);
        atomicAdd(out_sum_yw, syw[0]);
    }
}

bool measure_autoexposure_ev_cuda_center_weighted(
    const nc::NdArray<float>& image_hwc, bool apply_cctf_decoding, float& out_ev) {
    (void)apply_cctf_decoding; // decode on CPU for simplicity; CUDA path expects linear RGB already
    const int H = (int)image_hwc.shape().rows;
    const int W3 = (int)image_hwc.shape().cols;
    if (W3 % 3 != 0) return false;
    const int W = W3 / 3;

    // Extract approximate Y = 0.2126 R + 0.7152 G + 0.0722 B (assumes linear)
    std::vector<float> hostY(H * W);
    for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
        float R = image_hwc(y, x * 3 + 0);
        float G = image_hwc(y, x * 3 + 1);
        float B = image_hwc(y, x * 3 + 2);
        hostY[y * W + x] = 0.2126f * R + 0.7152f * G + 0.0722f * B;
    }

    float *dY = nullptr, *dw = nullptr;
    float *d_sum_w = nullptr, *d_sum_yw = nullptr;
    const int N = H * W;
    if (cudaMalloc(&dY, N * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&dw, N * sizeof(float)) != cudaSuccess) { cudaFree(dY); return false; }
    if (cudaMalloc(&d_sum_w, sizeof(float)) != cudaSuccess) { cudaFree(dY); cudaFree(dw); return false; }
    if (cudaMalloc(&d_sum_yw, sizeof(float)) != cudaSuccess) { cudaFree(dY); cudaFree(dw); cudaFree(d_sum_w); return false; }
    cudaMemcpy(dY, hostY.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemset(d_sum_w, 0, sizeof(float));
    cudaMemset(d_sum_yw, 0, sizeof(float));

    const float smax = (float)std::max(H, W);
    const float sy = (float)H / smax;
    const float sx = (float)W / smax;
    const float sigma = 0.2f;

    dim3 block2d(16, 16);
    dim3 grid2d((W + block2d.x - 1) / block2d.x, (H + block2d.y - 1) / block2d.y);
    compute_weights_kernel<<<grid2d, block2d>>>(H, W, sx, sy, sigma, dw);
    cudaDeviceSynchronize();

    dim3 block1d(256);
    dim3 grid1d((N + block1d.x - 1) / block1d.x);
    sum_weighted_kernel<<<grid1d, block1d>>>(dY, dw, N, d_sum_w, d_sum_yw);
    cudaDeviceSynchronize();

    float h_sum_w = 0.0f, h_sum_yw = 0.0f;
    cudaMemcpy(&h_sum_w, d_sum_w, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&h_sum_yw, d_sum_yw, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(dY); cudaFree(dw); cudaFree(d_sum_w); cudaFree(d_sum_yw);
    if (h_sum_w <= 0.0) return false;
    float Y_exposure = h_sum_yw / h_sum_w;
    float exposure = Y_exposure / 0.184f;
    out_ev = (exposure > 0.0f) ? -log2f(exposure) : 0.0f;
    return true;
}

} // namespace utils
} // namespace agx


