// diffusion.cu
#include "diffusion.hpp"

#if defined(__CUDACC__) || defined(CUDA_VERSION)
#include <cuda_runtime.h>

__device__ __forceinline__ int reflect(int x, int n) {
    if (n <= 1) return 0;
    while (x < 0 || x >= n) {
        if (x < 0) x = -x - 1;
        else       x = 2 * n - x - 1;
    }
    return x;
}

namespace agx_emulsion {

namespace {

__global__ void sep_h(const float* in, float* tmp,
                      int H, int W, int channel,
                      const float* k, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.f;
    for (int t = -radius; t <= radius; ++t) {
        int xr = reflect(x + t, W);
        size_t idx = ((size_t)y * W + xr) * 3 + channel;
        acc += in[idx] * k[t + radius];
    }
    tmp[y * W + x] = acc;
}

__global__ void sep_v(const float* tmp, float* out,
                      int H, int W, int channel,
                      const float* k, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= W || y >= H) return;

    float acc = 0.f;
    for (int t = -radius; t <= radius; ++t) {
        int yr = reflect(y + t, H);
        acc += tmp[yr * W + x] * k[t + radius];
    }
    out[((size_t)y * W + x) * 3 + channel] = acc;
}

// Shared-memory optimised separable planar Gaussian blur
__global__ void gauss_h_planar_shared(const float* __restrict__ inP,
                                      float* __restrict__ tmpP,
                                      int Hh, int Ww,
                                      const float* __restrict__ kP,
                                      int rad) {
    extern __shared__ float sdata[]; // size: blockDim.y * (blockDim.x + 2*rad)
    const int stride = blockDim.x + 2 * rad;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (y >= Hh) return;
    const int xBase = blockIdx.x * blockDim.x;
    float* srow = sdata + threadIdx.y * stride;

    // Load tile with halo using reflection
    for (int ox = threadIdx.x; ox < blockDim.x + 2 * rad; ox += blockDim.x) {
        int gx = xBase + ox - rad;
        int xr = reflect(gx, Ww);
        srow[ox] = inP[y * Ww + xr];
    }
    __syncthreads();

    int x = xBase + threadIdx.x;
    if (x >= Ww) return;
    float acc = 0.f;
    const int lx = threadIdx.x + rad;
    for (int t = -rad; t <= rad; ++t) {
        acc += srow[lx + t] * kP[t + rad];
    }
    tmpP[y * Ww + x] = acc;
}

__global__ void gauss_v_planar_shared(const float* __restrict__ tmpP,
                                      float* __restrict__ outP,
                                      int Hh, int Ww,
                                      const float* __restrict__ kP,
                                      int rad) {
    extern __shared__ float sdata[]; // size: (blockDim.y + 2*rad) * blockDim.x
    const int stride = blockDim.x;
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    if (x >= Ww) return;
    const int yBase = blockIdx.y * blockDim.y;

    // Load column tile with halo using reflection
    for (int oy = threadIdx.y; oy < blockDim.y + 2 * rad; oy += blockDim.y) {
        int gy = yBase + oy - rad;
        int yr = reflect(gy, Hh);
        sdata[oy * stride + threadIdx.x] = tmpP[yr * Ww + x];
    }
    __syncthreads();

    int y = yBase + threadIdx.y;
    if (y >= Hh) return;
    float acc = 0.f;
    const int ly = threadIdx.y + rad;
    for (int t = -rad; t <= rad; ++t) {
        acc += sdata[(ly + t) * stride + threadIdx.x] * kP[t + rad];
    }
    outP[y * Ww + x] = acc;
}

} // anonymous namespace

bool diffusion_cuda::gaussian_blur_rgb(const float* in, float* out,
                                       int H, int W,
                                       const float* k1d, int ksize)
{
    if (ksize <= 0 || H <= 0 || W <= 0) return false;
    const int radius = ksize / 2;

    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_tmp = nullptr;
    float* d_k = nullptr;

    size_t n_pix = (size_t)H * W;
    size_t n_rgb = n_pix * 3;

    cudaError_t err = cudaSuccess;
    if ((err = cudaMalloc(&d_in,  n_rgb * sizeof(float))) != cudaSuccess) return false;
    if ((err = cudaMalloc(&d_out, n_rgb * sizeof(float))) != cudaSuccess) { cudaFree(d_in); return false; }
    if ((err = cudaMalloc(&d_tmp, n_pix * sizeof(float))) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); return false; }
    if ((err = cudaMalloc(&d_k,   ksize * sizeof(float))) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); return false; }

    if ((err = cudaMemcpy(d_in, in, n_rgb * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); cudaFree(d_k); return false;
    }
    if ((err = cudaMemcpy(d_k, k1d, ksize * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); cudaFree(d_k); return false;
    }

    dim3 block(16,16);
    dim3 grid((W + block.x - 1)/block.x, (H + block.y - 1)/block.y);

    for (int c = 0; c < 3; ++c) {
        sep_h<<<grid, block>>>(d_in, d_tmp, H, W, c, d_k, radius);
        sep_v<<<grid, block>>>(d_tmp, d_out, H, W, c, d_k, radius);
    }
    err = cudaDeviceSynchronize();

    bool ok = (err == cudaSuccess);
    if (ok) cudaMemcpy(out, d_out, n_rgb * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); cudaFree(d_k);
    return ok;
}

bool diffusion_cuda::gaussian_blur_planar(const float* in, float* out,
                              int H, int W,
                              const float* k1d, int ksize) {
    if (ksize <= 0 || H <= 0 || W <= 0) return false;
    const int radius = ksize / 2;

    float* d_in = nullptr;
    float* d_out = nullptr;
    float* d_tmp = nullptr;
    float* d_k = nullptr;

    size_t n_pix = (size_t)H * W;

    cudaError_t err = cudaSuccess;
    if ((err = cudaMalloc(&d_in,  n_pix * sizeof(float))) != cudaSuccess) return false;
    if ((err = cudaMalloc(&d_out, n_pix * sizeof(float))) != cudaSuccess) { cudaFree(d_in); return false; }
    if ((err = cudaMalloc(&d_tmp, n_pix * sizeof(float))) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); return false; }
    if ((err = cudaMalloc(&d_k,   ksize * sizeof(float))) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); return false; }

    if ((err = cudaMemcpy(d_in, in, n_pix * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); cudaFree(d_k); return false; }
    if ((err = cudaMemcpy(d_k, k1d, ksize * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) { cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); cudaFree(d_k); return false; }

    // Launch shared-memory optimised separable passes
    dim3 blockH(64, 4);
    dim3 gridH((W + blockH.x - 1)/blockH.x, (H + blockH.y - 1)/blockH.y);
    size_t smemH = (blockH.x + 2 * radius) * blockH.y * sizeof(float);
    gauss_h_planar_shared<<<gridH, blockH, smemH>>>(d_in, d_tmp, H, W, d_k, radius);

    dim3 blockV(8, 32);
    dim3 gridV((W + blockV.x - 1)/blockV.x, (H + blockV.y - 1)/blockV.y);
    size_t smemV = (blockV.y + 2 * radius) * blockV.x * sizeof(float);
    gauss_v_planar_shared<<<gridV, blockV, smemV>>>(d_tmp, d_out, H, W, d_k, radius);
    err = cudaDeviceSynchronize();

    bool ok = (err == cudaSuccess);
    if (ok) cudaMemcpy(out, d_out, n_pix * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in); cudaFree(d_out); cudaFree(d_tmp); cudaFree(d_k);
    return ok;
}

} // namespace agx_emulsion

#else
// No CUDA toolchain available; stub returns false so CPU path is used.
namespace agx_emulsion {
namespace diffusion_cuda {
bool gaussian_blur_rgb(const float*, float*, int, int, const float*, int) {
    return false;
}
} // namespace diffusion_cuda
} // namespace agx_emulsion
#endif 