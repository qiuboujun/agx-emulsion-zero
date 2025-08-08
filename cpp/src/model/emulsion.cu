// emulsion.cu
//
// CUDA kernels for the AgX emulsion implementation.  This file provides a
// device implementation of a separable Gaussian blur on three channel
// floating point images.  The host wrapper function accepts pointers to
// contiguous memory and performs all necessary device allocations and
// transfers.  If CUDA is unavailable in your build environment you may
// ignore this file; the C++ implementation in emulsion.cpp contains
// portable CPU versions of all algorithms.

#include "emulsion.hpp"
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>

namespace agx_emulsion {

// CUDA kernel for 2D Gaussian blur
__global__ void gaussian_blur_2d_kernel(const float* input, float* output,
                                       int width, int height, int channels,
                                       const float* kernel, int kernel_size) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int radius = kernel_size / 2;
    
    for (int c = 0; c < channels; ++c) {
        float sum = 0.0f;
        float weight_sum = 0.0f;
        
        for (int ky = -radius; ky <= radius; ++ky) {
            for (int kx = -radius; kx <= radius; ++kx) {
                int nx = x + kx;
                int ny = y + ky;
                
                // Reflect boundary
                if (nx < 0) nx = -nx - 1;
                if (nx >= width) nx = 2 * width - nx - 1;
                if (ny < 0) ny = -ny - 1;
                if (ny >= height) ny = 2 * height - ny - 1;
                
                float weight = kernel[(ky + radius) * kernel_size + (kx + radius)];
                sum += weight * input[(ny * width + nx) * channels + c];
                weight_sum += weight;
            }
        }
        
        output[(y * width + x) * channels + c] = sum / weight_sum;
    }
}

// CUDA kernel for grain simulation using fixed arrays
__global__ void grain_simulation_kernel(const float* density, float* output,
                                       int width, int height, int depth, int channels,
                                       const float* fixed_rands, size_t rand_offset,
                                       float od_particle, float n_particles_per_pixel,
                                       const float* grain_uniformity) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    for (int c = 0; c < channels; ++c) {
        int idx = ((z * height + y) * width + x) * channels + c;
        float d = density[idx];
        
        float p = fmaxf(1e-6f, fminf(1.0f - 1e-6f, d / 2.2f));
        float saturation = 1.0f - p * grain_uniformity[c] * (1.0f - 1e-6f);
        float lambda = n_particles_per_pixel / fmaxf(1e-6f, saturation);
        
        // Use fixed random values
        size_t rand_idx = rand_offset + idx * 2;
        float rand1 = fixed_rands[rand_idx];
        float rand2 = fixed_rands[rand_idx + 1];
        
        // Simple Poisson approximation
        int n = (int)(lambda * rand1);
        n = max(0, n);
        
        // Simple binomial approximation
        int developed = (int)(n * p * rand2);
        developed = max(0, min(developed, n));
        
        float grain_val = (float)developed * od_particle * saturation;
        output[idx] = d + grain_val;
    }
}

// CUDA kernel for DIR coupler simulation using fixed arrays
__global__ void dir_coupler_kernel(const float* density, float* output,
                                  int width, int height, int depth, int channels,
                                  const float* fixed_rands, size_t rand_offset,
                                  const float* dir_matrix, float dir_scale) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;
    
    if (x >= width || y >= height || z >= depth) return;
    
    float input_density[3] = {0.0f, 0.0f, 0.0f};
    float output_density[3] = {0.0f, 0.0f, 0.0f};
    
    // Get input densities for RGB channels
    for (int c = 0; c < 3 && c < channels; ++c) {
        int idx = ((z * height + y) * width + x) * channels + c;
        input_density[c] = density[idx];
    }
    
    // Apply DIR coupler matrix
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            output_density[i] += dir_matrix[i * 3 + j] * input_density[j] * dir_scale;
        }
    }
    
    // Add fixed noise and write output
    for (int c = 0; c < 3 && c < channels; ++c) {
        int idx = ((z * height + y) * width + x) * channels + c;
        float noise = fixed_rands[rand_offset + idx] * 0.01f; // Small fixed noise
        output[idx] = fmaxf(0.0f, fminf(2.2f, output_density[c] + noise));
    }
}

// Host wrapper for CUDA grain simulation
Image3D apply_grain_cuda(const Image3D& density, const GrainParams& params) {
    Image3D result = density;
    
    // Get fixed random values
    std::vector<float> fixed_rands = AgXEmulsion::get_fixed_random_values(density.size() * 2);
    
    // Allocate device memory
    thrust::device_vector<float> d_density(density.data);
    thrust::device_vector<float> d_output(density.data);
    thrust::device_vector<float> d_fixed_rands(fixed_rands);
    thrust::device_vector<float> d_grain_uniformity(params.grain_uniformity.begin(), params.grain_uniformity.end());
    
    // Kernel parameters
    const float od_particle = 0.22f;
    const float n_particles_per_pixel = 10.0f;
    
    // Launch kernel
    dim3 block_size(16, 16, 4);
    dim3 grid_size((density.width + block_size.x - 1) / block_size.x,
                   (density.height + block_size.y - 1) / block_size.y,
                   (density.depth + block_size.z - 1) / block_size.z);
    
    grain_simulation_kernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_density.data()),
        thrust::raw_pointer_cast(d_output.data()),
        density.width, density.height, density.depth, density.channels,
        thrust::raw_pointer_cast(d_fixed_rands.data()), 0,
        od_particle, n_particles_per_pixel,
        thrust::raw_pointer_cast(d_grain_uniformity.data())
    );
    
    // Copy result back to host
    thrust::copy(d_output.begin(), d_output.end(), result.data.begin());
    
    // Apply Gaussian blur if specified
    if (params.grain_blur > 0.0f) {
        // Note: This would require a separate CUDA blur implementation
        // For now, fall back to CPU blur
        Image3D blurred(result.width, result.height, result.depth, result.channels);
        AgXEmulsion::gaussian_blur_3d(result, blurred, params.grain_blur, params.grain_blur, 0.0f);
        result = blurred;
    }
    
    return result;
}

// Host wrapper for CUDA DIR coupler simulation
Image3D apply_dir_couplers_cuda(const Image3D& density, const DIRCouplerParams& params) {
    if (!params.enable_dir_couplers) {
        return density;
    }
    
    Image3D result = density;
    
    // Get fixed random values
    std::vector<float> fixed_rands = AgXEmulsion::get_fixed_random_values(density.size());
    
    // DIR coupler matrix
    std::array<float, 9> dir_matrix = {
        1.0f, -0.1f, -0.1f,
        -0.1f, 1.0f, -0.1f,
        -0.1f, -0.1f, 1.0f
    };
    
    // Allocate device memory
    thrust::device_vector<float> d_density(density.data);
    thrust::device_vector<float> d_output(density.data);
    thrust::device_vector<float> d_fixed_rands(fixed_rands);
    thrust::device_vector<float> d_dir_matrix(dir_matrix.begin(), dir_matrix.end());
    
    // Launch kernel
    dim3 block_size(16, 16, 4);
    dim3 grid_size((density.width + block_size.x - 1) / block_size.x,
                   (density.height + block_size.y - 1) / block_size.y,
                   (density.depth + block_size.z - 1) / block_size.z);
    
    dir_coupler_kernel<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_density.data()),
        thrust::raw_pointer_cast(d_output.data()),
        density.width, density.height, density.depth, density.channels,
        thrust::raw_pointer_cast(d_fixed_rands.data()), 0,
        thrust::raw_pointer_cast(d_dir_matrix.data()), params.dir_coupler_scale
    );
    
    // Copy result back to host
    thrust::copy(d_output.begin(), d_output.end(), result.data.begin());
    
    // Apply Gaussian blur if specified
    if (params.dir_coupler_blur > 0.0f) {
        Image3D blurred(result.width, result.height, result.depth, result.channels);
        AgXEmulsion::gaussian_blur_3d(result, blurred, params.dir_coupler_blur, params.dir_coupler_blur, 0.0f);
        result = blurred;
    }
    
    return result;
}

} // namespace agx_emulsion
