// diffusion.cu
#include "diffusion.hpp"

#if defined(__CUDACC__) || defined(CUDA_VERSION)
#include <cuda_runtime.h>

namespace agx_emulsion {

namespace {

__device__ __forceinline__ int reflect(int x, int n) {
    if (n <= 1) return 0;
    while (x < 0 || x >= n) {
        if (x < 0) x = -x - 1;
        else       x = 2 * n - x - 1;
    }
    return x;
}

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