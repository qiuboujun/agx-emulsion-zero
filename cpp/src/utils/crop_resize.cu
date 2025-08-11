// SPDX-License-Identifier: MIT

#include <cuda_runtime.h>
#include "crop_resize.hpp"

namespace agx { namespace utils {

__device__ inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

__global__ void resize_bilinear_kernel(const float* __restrict__ src, int H, int W,
                                       float* __restrict__ dst, int newH, int newW) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= newW || y >= newH) return;
    float scaleY = (float)H / (float)newH;
    float scaleX = (float)W / (float)newW;
    float srcY = (y + 0.5f) * scaleY - 0.5f;
    int y0 = (int)floorf(srcY);
    int y1 = y0 + 1; if (y1 >= H) y1 = H - 1; if (y0 < 0) y0 = 0;
    float ty = srcY - y0; if (ty < 0.f) ty = 0.f;
    float srcX = (x + 0.5f) * scaleX - 0.5f;
    int x0 = (int)floorf(srcX);
    int x1 = x0 + 1; if (x1 >= W) x1 = W - 1; if (x0 < 0) x0 = 0;
    float tx = srcX - x0; if (tx < 0.f) tx = 0.f;
    for (int c = 0; c < 3; ++c) {
        float p00 = src[(y0 * W + x0) * 3 + c];
        float p01 = src[(y0 * W + x1) * 3 + c];
        float p10 = src[(y1 * W + x0) * 3 + c];
        float p11 = src[(y1 * W + x1) * 3 + c];
        float a = lerp(p00, p01, tx);
        float b = lerp(p10, p11, tx);
        dst[(y * newW + x) * 3 + c] = lerp(a, b, ty);
    }
}

bool resize_image_bilinear_cuda(
    const nc::NdArray<float>& image_hwc, int newH, int newW, nc::NdArray<float>& out_hwc) {
    const int H = (int)image_hwc.shape().rows;
    const int W3 = (int)image_hwc.shape().cols;
    if (W3 % 3 != 0) return false;
    const int W = W3 / 3;
    const int Nsrc = H * W * 3;
    const int Ndst = newH * newW * 3;
    float *d_src = nullptr, *d_dst = nullptr;
    if (cudaMalloc(&d_src, Nsrc * sizeof(float)) != cudaSuccess) return false;
    if (cudaMalloc(&d_dst, Ndst * sizeof(float)) != cudaSuccess) { cudaFree(d_src); return false; }
    cudaMemcpy(d_src, image_hwc.data(), Nsrc * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(16, 16);
    dim3 grid((newW + block.x - 1) / block.x, (newH + block.y - 1) / block.y);
    resize_bilinear_kernel<<<grid, block>>>(d_src, H, W, d_dst, newH, newW);
    if (cudaDeviceSynchronize() != cudaSuccess) { cudaFree(d_src); cudaFree(d_dst); return false; }

    out_hwc = nc::NdArray<float>(newH, newW * 3);
    cudaMemcpy(out_hwc.data(), d_dst, Ndst * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_src); cudaFree(d_dst);
    return true;
}

}} // namespace agx::utils


