#include "fast_interp_lut.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <cmath>
#include <vector>

//================================================================================
// HOST & DEVICE HELPER FUNCTIONS
//================================================================================

// Mitchell-Netravali cubic kernel (matches Python's cubic_interp_lut_at_2d)
__host__ __device__ inline double mitchell_weight_d(double t) {
    const double B = 1.0 / 3.0;
    const double C = 1.0 / 3.0;
    double x = fabs(t);
    double x2 = x * x;
    double x3 = x2 * x;
    
    if (x < 1.0) {
        return (1.0 / 6.0) * ((12.0 - 9.0 * B - 6.0 * C) * x3 +
                              (-18.0 + 12.0 * B + 6.0 * C) * x2 +
                              (6.0 - 2.0 * B));
    } else if (x < 2.0) {
        return (1.0 / 6.0) * ((-B - 6.0 * C) * x3 +
                              (6.0 * B + 30.0 * C) * x2 +
                              (-12.0 * B - 48.0 * C) * x +
                              (8.0 * B + 24.0 * C));
    }
    return 0.0;
}

// Reflection boundary handling (matches Python's safe_index)
__host__ __device__ inline int reflect_index(int idx, int L) {
    if (idx < 0) return -idx;
    if (idx >= L) return 2 * (L - 1) - idx;
    return idx;
}

//================================================================================
// CUDA KERNELS
//================================================================================

__global__ void apply_lut_cubic_2d_kernel(
    float* output,
    const float* image,
    const float* lut,
    int height, int width,
    int lut_size, int lut_channels)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x; // width index
    const int i = blockIdx.y * blockDim.y + threadIdx.y; // height index

    if (i >= height || j >= width) return;

    const int image_pixel_idx = (i * width + j);
    const float x_in = image[image_pixel_idx * 2 + 0] * (lut_size - 1);
    const float y_in = image[image_pixel_idx * 2 + 1] * (lut_size - 1);

    const int x_base = floorf(x_in), y_base = floorf(y_in);
    const float x_frac = x_in - x_base, y_frac = y_in - y_base;

    double wx[4], wy[4];
    wx[0] = mitchell_weight_d((double)x_frac + 1.0); wx[1] = mitchell_weight_d((double)x_frac);
    wx[2] = mitchell_weight_d((double)x_frac - 1.0); wx[3] = mitchell_weight_d((double)x_frac - 2.0);
    wy[0] = mitchell_weight_d((double)y_frac + 1.0); wy[1] = mitchell_weight_d((double)y_frac);
    wy[2] = mitchell_weight_d((double)y_frac - 1.0); wy[3] = mitchell_weight_d((double)y_frac - 2.0);

    double weight_sum = 0.0;
    double out_val[16];
    #pragma unroll
    for (int c=0;c<16;++c) out_val[c]=0.0;

    for (int m = 0; m < 4; ++m) {
        int y_idx = reflect_index(y_base - 1 + m, lut_size);
        for (int n = 0; n < 4; ++n) {
            int x_idx = reflect_index(x_base - 1 + n, lut_size);
            double weight = wx[n] * wy[m];
            weight_sum += weight;
            // Index into flattened LUT: [x*L + y, channel] for row-major order
            int lut_pixel_idx = x_idx * lut_size + y_idx;
            for (int c = 0; c < lut_channels; ++c) {
                out_val[c] += weight * (double)lut[lut_pixel_idx * lut_channels + c];
            }
        }
    }

    const int output_pixel_idx = (i * width + j);
    if (weight_sum != 0.0) {
        for (int c = 0; c < lut_channels; ++c) {
            output[output_pixel_idx * lut_channels + c] = (float)(out_val[c] / weight_sum);
        }
    }
}

__global__ void apply_lut_cubic_3d_kernel(
    float* output,
    const float* image,
    const float* lut,
    int height, int width,
    int lut_size, int lut_channels)
{
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;

    if (i >= height || j >= width) return;

    const int image_pixel_idx = (i * width + j);
    const float r_in = image[image_pixel_idx * 3 + 0] * (lut_size - 1);
    const float g_in = image[image_pixel_idx * 3 + 1] * (lut_size - 1);
    const float b_in = image[image_pixel_idx * 3 + 2] * (lut_size - 1);

    const int r_base = floorf(r_in), g_base = floorf(g_in), b_base = floorf(b_in);
    const float r_frac = r_in - r_base, g_frac = g_in - g_base, b_frac = b_in - b_base;

    double wr[4], wg[4], wb[4];
    wr[0] = mitchell_weight_d((double)r_frac + 1.0); wr[1] = mitchell_weight_d((double)r_frac);
    wr[2] = mitchell_weight_d((double)r_frac - 1.0); wr[3] = mitchell_weight_d((double)r_frac - 2.0);
    wg[0] = mitchell_weight_d((double)g_frac + 1.0); wg[1] = mitchell_weight_d((double)g_frac);
    wg[2] = mitchell_weight_d((double)g_frac - 1.0); wg[3] = mitchell_weight_d((double)g_frac - 2.0);
    wb[0] = mitchell_weight_d((double)b_frac + 1.0); wb[1] = mitchell_weight_d((double)b_frac);
    wb[2] = mitchell_weight_d((double)b_frac - 1.0); wb[3] = mitchell_weight_d((double)b_frac - 2.0);

    double weight_sum = 0.0;
    double out_val[16];
    #pragma unroll
    for (int c=0;c<16;++c) out_val[c]=0.0;

    for (int m = 0; m < 4; ++m) {
        int r_idx = reflect_index(r_base - 1 + m, lut_size);
        for (int n = 0; n < 4; ++n) {
            int g_idx = reflect_index(g_base - 1 + n, lut_size);
            for (int p = 0; p < 4; ++p) {
                int b_idx = reflect_index(b_base - 1 + p, lut_size);
                double weight = wr[m] * wg[n] * wb[p];
                weight_sum += weight;
                int lut_pixel_idx = (r_idx * lut_size + g_idx) * lut_size + b_idx;
                for (int c = 0; c < lut_channels; ++c) {
                    out_val[c] += weight * (double)lut[lut_pixel_idx * lut_channels + c];
                }
            }
        }
    }

    const int output_pixel_idx = (i * width + j);
    if (weight_sum != 0.0) {
        for (int c = 0; c < lut_channels; ++c) {
            output[output_pixel_idx * lut_channels + c] = (float)(out_val[c] / weight_sum);
        }
    }
}

// Linear-index versions to avoid oversized 2D grids
__global__ void apply_lut_cubic_2d_kernel_linear(
    float* output,
    const float* image,
    const float* lut,
    int N, int width,
    int lut_size, int lut_channels)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float x_in = image[idx * 2 + 0] * (lut_size - 1);
    const float y_in = image[idx * 2 + 1] * (lut_size - 1);

    const int x_base = floorf(x_in), y_base = floorf(y_in);
    const float x_frac = x_in - x_base, y_frac = y_in - y_base;

    double wx[4], wy[4];
    wx[0] = mitchell_weight_d((double)x_frac + 1.0); wx[1] = mitchell_weight_d((double)x_frac);
    wx[2] = mitchell_weight_d((double)x_frac - 1.0); wx[3] = mitchell_weight_d((double)x_frac - 2.0);
    wy[0] = mitchell_weight_d((double)y_frac + 1.0); wy[1] = mitchell_weight_d((double)y_frac);
    wy[2] = mitchell_weight_d((double)y_frac - 1.0); wy[3] = mitchell_weight_d((double)y_frac - 2.0);

    double weight_sum = 0.0;
    double out_val[16];
    #pragma unroll
    for (int c=0;c<16;++c) out_val[c]=0.0;

    for (int m = 0; m < 4; ++m) {
        int y_idx = reflect_index(y_base - 1 + m, lut_size);
        for (int n = 0; n < 4; ++n) {
            int x_idx = reflect_index(x_base - 1 + n, lut_size);
            double weight = wx[n] * wy[m];
            weight_sum += weight;
            int lut_pixel_idx = x_idx * lut_size + y_idx;
            for (int c = 0; c < lut_channels; ++c) {
                out_val[c] += weight * (double)lut[lut_pixel_idx * lut_channels + c];
            }
        }
    }

    if (weight_sum != 0.0) {
        for (int c = 0; c < lut_channels; ++c) {
            output[idx * lut_channels + c] = (float)(out_val[c] / weight_sum);
        }
    }
}

__global__ void apply_lut_cubic_3d_kernel_linear(
    float* output,
    const float* image,
    const float* lut,
    int N, int width,
    int lut_size, int lut_channels)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    const float r_in = image[idx * 3 + 0] * (lut_size - 1);
    const float g_in = image[idx * 3 + 1] * (lut_size - 1);
    const float b_in = image[idx * 3 + 2] * (lut_size - 1);

    const int r_base = floorf(r_in), g_base = floorf(g_in), b_base = floorf(b_in);
    const float r_frac = r_in - r_base, g_frac = g_in - g_base, b_frac = b_in - b_base;

    double wr[4], wg[4], wb[4];
    wr[0] = mitchell_weight_d((double)r_frac + 1.0); wr[1] = mitchell_weight_d((double)r_frac);
    wr[2] = mitchell_weight_d((double)r_frac - 1.0); wr[3] = mitchell_weight_d((double)r_frac - 2.0);
    wg[0] = mitchell_weight_d((double)g_frac + 1.0); wg[1] = mitchell_weight_d((double)g_frac);
    wg[2] = mitchell_weight_d((double)g_frac - 1.0); wg[3] = mitchell_weight_d((double)g_frac - 2.0);
    wb[0] = mitchell_weight_d((double)b_frac + 1.0); wb[1] = mitchell_weight_d((double)b_frac);
    wb[2] = mitchell_weight_d((double)b_frac - 1.0); wb[3] = mitchell_weight_d((double)b_frac - 2.0);

    double weight_sum = 0.0;
    double out_val[16];
    #pragma unroll
    for (int c=0;c<16;++c) out_val[c]=0.0;

    for (int m = 0; m < 4; ++m) {
        int r_idx = reflect_index(r_base - 1 + m, lut_size);
        for (int n = 0; n < 4; ++n) {
            int g_idx = reflect_index(g_base - 1 + n, lut_size);
            for (int p = 0; p < 4; ++p) {
                int b_idx = reflect_index(b_base - 1 + p, lut_size);
                double weight = wr[m] * wg[n] * wb[p];
                weight_sum += weight;
                int lut_pixel_idx = (r_idx * lut_size + g_idx) * lut_size + b_idx;
                for (int c = 0; c < lut_channels; ++c) {
                    out_val[c] += weight * (double)lut[lut_pixel_idx * lut_channels + c];
                }
            }
        }
    }

    if (weight_sum != 0.0) {
        for (int c = 0; c < lut_channels; ++c) {
            output[idx * lut_channels + c] = (float)(out_val[c] / weight_sum);
        }
    }
}

//================================================================================
// HOST-FACING FUNCTIONS
//================================================================================

namespace agx {

std::vector<float> cubic_interp_lut_at_2d(const nc::NdArray<float>& lut, float x, float y) {
    // The LUT is flattened as [L*L, C] where L is the size of each dimension
    const int L = static_cast<int>(round(sqrt(lut.shape().rows)));
    const int channels = lut.shape().cols;
    const int x_base = floorf(x), y_base = floorf(y);
    const float x_frac = x - x_base, y_frac = y - y_base;

    double wx[4], wy[4];
    wx[0] = mitchell_weight_d((double)x_frac + 1.0); wx[1] = mitchell_weight_d((double)x_frac);
    wx[2] = mitchell_weight_d((double)x_frac - 1.0); wx[3] = mitchell_weight_d((double)x_frac - 2.0);
    wy[0] = mitchell_weight_d((double)y_frac + 1.0); wy[1] = mitchell_weight_d((double)y_frac);
    wy[2] = mitchell_weight_d((double)y_frac - 1.0); wy[3] = mitchell_weight_d((double)y_frac - 2.0);

    double weight_sum = 0.0;
    std::vector<double> out(channels, 0.0);

    for (int i = 0; i < 4; ++i) {
        int xi = reflect_index(x_base - 1 + i, L);
        for (int j = 0; j < 4; ++j) {
            int yj = reflect_index(y_base - 1 + j, L);
            double weight = wx[i] * wy[j];
            weight_sum += weight;
            // Index into flattened LUT: [x*L + y, channel] for row-major order
            int lut_idx = xi * L + yj;
            for (int c = 0; c < channels; ++c) {
                out[c] += weight * (double)lut(lut_idx, c);
            }
        }
    }

    if (weight_sum != 0.0) {
        for (int c = 0; c < channels; ++c) out[c] /= weight_sum;
    }
    std::vector<float> out_f(channels);
    for (int c=0;c<channels;++c) out_f[c] = (float)out[c];
    return out_f;
}

std::vector<float> cubic_interp_lut_at_3d(const nc::NdArray<float>& lut, float r, float g, float b) {
    // The LUT is flattened as [L*L*L, C] where L is the size of each dimension
    const int L = static_cast<int>(round(cbrt(lut.shape().rows)));
    const int channels = lut.shape().cols;
    const int r_base = floorf(r), g_base = floorf(g), b_base = floorf(b);
    const float r_frac = r - r_base, g_frac = g - g_base, b_frac = b - b_base;

    double wr[4], wg[4], wb[4];
    wr[0] = mitchell_weight_d((double)r_frac + 1.0); wr[1] = mitchell_weight_d((double)r_frac);
    wr[2] = mitchell_weight_d((double)r_frac - 1.0); wr[3] = mitchell_weight_d((double)r_frac - 2.0);
    wg[0] = mitchell_weight_d((double)g_frac + 1.0); wg[1] = mitchell_weight_d((double)g_frac);
    wg[2] = mitchell_weight_d((double)g_frac - 1.0); wg[3] = mitchell_weight_d((double)g_frac - 2.0);
    wb[0] = mitchell_weight_d((double)b_frac + 1.0); wb[1] = mitchell_weight_d((double)b_frac);
    wb[2] = mitchell_weight_d((double)b_frac - 1.0); wb[3] = mitchell_weight_d((double)b_frac - 2.0);

    double weight_sum = 0.0;
    std::vector<double> out(channels, 0.0);

    for (int i = 0; i < 4; ++i) {
        int ri = reflect_index(r_base - 1 + i, L);
        for (int j = 0; j < 4; ++j) {
            int gj = reflect_index(g_base - 1 + j, L);
            for (int k = 0; k < 4; ++k) {
                int bk = reflect_index(b_base - 1 + k, L);
                double weight = wr[i] * wg[j] * wb[k];
                weight_sum += weight;
                // Index into flattened LUT: [(r*L + g)*L + b, channel]
                int lut_idx = (ri * L + gj) * L + bk;
                for (int c = 0; c < channels; ++c) {
                    out[c] += weight * (double)lut(lut_idx, c);
                }
            }
        }
    }

    if (weight_sum != 0.0) {
        for (int c = 0; c < channels; ++c) out[c] /= weight_sum;
    }
    std::vector<float> out_f(channels);
    for (int c=0;c<channels;++c) out_f[c] = (float)out[c];
    return out_f;
}

nc::NdArray<float> apply_lut_cubic_2d(const nc::NdArray<float>& lut, const nc::NdArray<float>& image, int height, int width) {
    const int lut_size = static_cast<int>(round(sqrt(lut.shape().rows)));
    const int lut_channels = lut.shape().cols;
    auto output = nc::NdArray<float>(height * width, lut_channels);

    // Validate dimensions to prevent CUDA launch errors
    if (width <= 0 || height <= 0 || width > 65535 || height > 65535) {
        throw std::runtime_error("Invalid image dimensions for CUDA kernel: " + std::to_string(width) + "x" + std::to_string(height));
    }

    float *dev_lut, *dev_image, *dev_output;
    cudaMalloc(&dev_lut, lut.nbytes());
    cudaMemcpy(dev_lut, lut.data(), lut.nbytes(), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_image, image.nbytes());
    cudaMemcpy(dev_image, image.data(), image.nbytes(), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_output, output.nbytes());

    // Use linear kernel to avoid oversized grid.y
    const int N = height * width;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) {
        cudaFree(dev_lut);
        cudaFree(dev_image);
        cudaFree(dev_output);
        throw std::runtime_error("Grid.x too large for CUDA: blocks=" + std::to_string(blocks));
    }
    apply_lut_cubic_2d_kernel_linear<<<blocks, threads>>>(dev_output, dev_image, dev_lut, N, width, lut_size, lut_channels);
    
    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(dev_lut);
        cudaFree(dev_image);
        cudaFree(dev_output);
        throw std::runtime_error("CUDA Kernel Launch Error");
    }

    cudaMemcpy(output.data(), dev_output, output.nbytes(), cudaMemcpyDeviceToHost);

    cudaFree(dev_lut); 
    cudaFree(dev_image); 
    cudaFree(dev_output);
    return output;
}

nc::NdArray<float> apply_lut_cubic_3d(const nc::NdArray<float>& lut, const nc::NdArray<float>& image, int height, int width) {
    const int lut_size = static_cast<int>(round(cbrt(lut.shape().rows)));
    const int lut_channels = lut.shape().cols;
    auto output = nc::NdArray<float>(height * width, lut_channels);

    // Validate dimensions to prevent CUDA launch errors
    if (width <= 0 || height <= 0 || width > 65535 || height > 65535) {
        throw std::runtime_error("Invalid image dimensions for CUDA kernel: " + std::to_string(width) + "x" + std::to_string(height));
    }

    float *dev_lut, *dev_image, *dev_output;
    cudaMalloc(&dev_lut, lut.nbytes());
    cudaMemcpy(dev_lut, lut.data(), lut.nbytes(), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_image, image.nbytes());
    cudaMemcpy(dev_image, image.data(), image.nbytes(), cudaMemcpyHostToDevice);
    cudaMalloc(&dev_output, output.nbytes());

    // Use linear kernel to avoid oversized grid.y
    const int N = height * width;
    const int threads = 256;
    const int blocks = (N + threads - 1) / threads;
    if (blocks > 65535) {
        cudaFree(dev_lut);
        cudaFree(dev_image);
        cudaFree(dev_output);
        throw std::runtime_error("Grid.x too large for CUDA: blocks=" + std::to_string(blocks));
    }
    apply_lut_cubic_3d_kernel_linear<<<blocks, threads>>>(dev_output, dev_image, dev_lut, N, width, lut_size, lut_channels);

    cudaDeviceSynchronize();
    if (cudaGetLastError() != cudaSuccess) {
        cudaFree(dev_lut);
        cudaFree(dev_image);
        cudaFree(dev_output);
        throw std::runtime_error("CUDA Kernel Launch Error");
    }

    cudaMemcpy(output.data(), dev_output, output.nbytes(), cudaMemcpyDeviceToHost);

    cudaFree(dev_lut); 
    cudaFree(dev_image); 
    cudaFree(dev_output);
    return output;
}

} // namespace agx
