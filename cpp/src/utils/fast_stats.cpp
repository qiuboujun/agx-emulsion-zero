#include "fast_stats.hpp"

#ifndef FAST_STATS_NO_CUDA
#include <cuda_runtime.h>
#include <stdexcept>

// Forward declaration of CUDA kernel
extern "C" __global__ void reduce_sum_and_sq(const float* data, float* out, size_t size);

namespace agx_emulsion {

// GPU entrypoint implementation
std::pair<double, double> FastStats::compute_gpu(const float* data, size_t size) {
    if (size == 0) return {0.0, 0.0};

    float* d_data = nullptr;
    float* d_result = nullptr;
    float h_result[2]; // sum, sum_sq

    cudaError_t err;
    
    // Allocate device memory
    if ((err = cudaMalloc(&d_data, size * sizeof(float))) != cudaSuccess) {
        throw std::runtime_error("CUDA memory allocation failed for input data");
    }
    
    if ((err = cudaMalloc(&d_result, 2 * sizeof(float))) != cudaSuccess) {
        cudaFree(d_data);
        throw std::runtime_error("CUDA memory allocation failed for result");
    }

    // Copy data to device
    if ((err = cudaMemcpy(d_data, data, size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        throw std::runtime_error("CUDA memory copy failed");
    }

    // Initialize result to zero
    if ((err = cudaMemset(d_result, 0, 2 * sizeof(float))) != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        throw std::runtime_error("CUDA memory memset failed");
    }

    // Launch kernel
    dim3 block(256);
    dim3 grid((size + block.x - 1) / block.x);
    reduce_sum_and_sq<<<grid, block, 2 * block.x * sizeof(float)>>>(d_data, d_result, size);

    // Check for kernel launch errors
    if ((err = cudaGetLastError()) != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        throw std::runtime_error("CUDA kernel launch failed");
    }

    // Copy results back to host
    if ((err = cudaMemcpy(h_result, d_result, 2 * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess) {
        cudaFree(d_data);
        cudaFree(d_result);
        throw std::runtime_error("CUDA memory copy failed");
    }

    // Clean up device memory
    cudaFree(d_data);
    cudaFree(d_result);

    // Calculate mean and standard deviation
    double mean = h_result[0] / size;
    double variance = h_result[1] / size;
    double stddev = std::sqrt(variance - mean * mean);
    
    return {mean, stddev};
}

} // namespace agx_emulsion

#else
// No CUDA implementation available
namespace agx_emulsion {
std::pair<double, double> FastStats::compute_gpu(const float* data, size_t size) {
    // Fallback to CPU implementation when CUDA is not available
    std::vector<float> vec(data, data + size);
    return mean_stddev(vec);
}
} // namespace agx_emulsion
#endif 