#include "grain.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>

// Simple Poisson and Binomial samplers on device.
// For Poisson we use the Knuth algorithm (sufficient for moderate lambda).
// For Binomial, we do N Bernoulli trials (good for small/medium N).

__device__ int poisson_knuth(curandStatePhilox4_32_10_t* state, float lambda) {
    if (lambda <= 0.0f) return 0;
    const float L = expf(-lambda);
    int k = 0;
    float p = 1.0f;
    do {
        ++k;
        float u = curand_uniform(state);
        p *= u;
    } while (p > L);
    return k - 1;
}

__device__ int binomial_naive(curandStatePhilox4_32_10_t* state, int n, float p) {
    if (n <= 0) return 0;
    int x = 0;
    for (int i = 0; i < n; ++i) {
        float u = curand_uniform(state);
        if (u < p) ++x;
    }
    return x;
}

__global__ void kernel_layer_particle(const float* density,
                                      float* out,
                                      int W, int H,
                                      float density_max,
                                      float npp,
                                      float grain_uniformity,
                                      unsigned long long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = W * H;
    if (idx >= N) return;

    int y = idx / W;
    int x = idx - y * W;
    float d = density[idx];
    float p = d / density_max;
    p = fminf(fmaxf(p, 1e-6f), 1.0f - 1e-6f);

    float saturation = 1.0f - p * grain_uniformity * (1.0f - 1e-6f);
    float lambda = npp / fmaxf(saturation, 1e-6f);
    float od_particle = density_max / fmaxf(npp, 1.0f);

    curandStatePhilox4_32_10_t state;
    curand_init(seed, idx, 0, &state);

    int seeds = poisson_knuth(&state, lambda);
    int developed = binomial_naive(&state, seeds, p);

    out[idx] = float(developed) * od_particle * saturation;
}

namespace agx_emulsion {
namespace Grain {

Image2D layer_particle_model_cuda(const Image2D& density,
                                  float density_max,
                                  float n_particles_per_pixel,
                                  float grain_uniformity,
                                  uint64_t seed,
                                  float /*blur_particle*/) {
    // blur on GPU is intentionally omitted for simplicity
    if (density.channels != 1) {
        throw std::runtime_error("CUDA layer_particle_model expects 1-channel density.");
    }
    const int W = density.width, H = density.height;
    const int N = W * H;

    float *d_in = nullptr, *d_out = nullptr;
    cudaMalloc(&d_in,  N * sizeof(float));
    cudaMalloc(&d_out, N * sizeof(float));
    cudaMemcpy(d_in, density.data.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x);

    kernel_layer_particle<<<grid, block>>>(d_in, d_out, W, H,
                                           density_max,
                                           n_particles_per_pixel,
                                           grain_uniformity,
                                           (unsigned long long)(seed ? seed : 12345ULL));
    cudaDeviceSynchronize();

    Image2D out(W, H, 1);
    cudaMemcpy(out.data.data(), d_out, N * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_in);
    cudaFree(d_out);
    return out;
}

} // namespace Grain
} // namespace agx_emulsion 