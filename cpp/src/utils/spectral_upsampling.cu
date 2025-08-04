// Copyright (c) 2025
//
// This CUDA source provides GPU accelerated versions of the simple
// coordinate transforms defined in `spectral_upsampling.hpp`.  The
// device kernels mirror the logic of the CPU implementation but can
// operate on large batches of coordinates in parallel.  The entry
// functions here accept flat arrays of x/y pairs and write the results
// in‑place into the provided output arrays.  They are intended to be
// called from a host wrapper or via a higher level API; no device
// memory management is performed here.

#include <cuda_runtime.h>
#include "spectral_upsampling.hpp"

// Kernel to convert from triangular to square coordinates.  Each
// thread processes a single (tx, ty) pair stored in the input array.
__global__ static void tri2quad_kernel(float *out, const float *in, std::size_t n_coords)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_coords) {
        const std::size_t i = idx * 2;
        float tx = in[i];
        float ty = in[i + 1];
        const float denom = fmaxf(1.0f - tx, 1e-10f);
        float y = ty / denom;
        float x = (1.0f - tx) * (1.0f - tx);
        // Clamp to [0,1]
        x = fminf(fmaxf(x, 0.0f), 1.0f);
        y = fminf(fmaxf(y, 0.0f), 1.0f);
        out[i]     = x;
        out[i + 1] = y;
    }
}

// Kernel to convert from square to triangular coordinates.  Each
// thread operates on a single (x, y) pair.
__global__ static void quad2tri_kernel(float *out, const float *in, std::size_t n_coords)
{
    const std::size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_coords) {
        const std::size_t i = idx * 2;
        float x = in[i];
        float y = in[i + 1];
        float sqrt_x = sqrtf(x);
        float tx = 1.0f - sqrt_x;
        float ty = y * sqrt_x;
        out[i]     = tx;
        out[i + 1] = ty;
    }
}

// Host wrappers.  These allocate device memory, copy input data,
// launch the kernels and copy results back.  They throw std::runtime_error
// on CUDA errors.  For brevity no CUDA stream support is provided.

static void check_cuda(cudaError_t result)
{
    if (result != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(result));
    }
}

void tri2quad_cuda(std::vector<float> &out, const std::vector<float> &in)
{
    const std::size_t n_coords = in.size() / 2;
    out.resize(in.size());
    float *d_in  = nullptr;
    float *d_out = nullptr;
    const std::size_t nbytes = in.size() * sizeof(float);
    check_cuda(cudaMalloc(&d_in, nbytes));
    check_cuda(cudaMalloc(&d_out, nbytes));
    check_cuda(cudaMemcpy(d_in, in.data(), nbytes, cudaMemcpyHostToDevice));
    const int blockSize = 256;
    const int gridSize = (n_coords + blockSize - 1) / blockSize;
    tri2quad_kernel<<<gridSize, blockSize>>>(d_out, d_in, n_coords);
    check_cuda(cudaDeviceSynchronize());
    check_cuda(cudaMemcpy(out.data(), d_out, nbytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(d_in));
    check_cuda(cudaFree(d_out));
}

void quad2tri_cuda(std::vector<float> &out, const std::vector<float> &in)
{
    const std::size_t n_coords = in.size() / 2;
    out.resize(in.size());
    float *d_in  = nullptr;
    float *d_out = nullptr;
    const std::size_t nbytes = in.size() * sizeof(float);
    check_cuda(cudaMalloc(&d_in, nbytes));
    check_cuda(cudaMalloc(&d_out, nbytes));
    check_cuda(cudaMemcpy(d_in, in.data(), nbytes, cudaMemcpyHostToDevice));
    const int blockSize = 256;
    const int gridSize = (n_coords + blockSize - 1) / blockSize;
    quad2tri_kernel<<<gridSize, blockSize>>>(d_out, d_in, n_coords);
    check_cuda(cudaDeviceSynchronize());
    check_cuda(cudaMemcpy(out.data(), d_out, nbytes, cudaMemcpyDeviceToHost));
    check_cuda(cudaFree(d_in));
    check_cuda(cudaFree(d_out));
}