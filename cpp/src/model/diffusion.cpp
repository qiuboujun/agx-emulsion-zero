// diffusion.cpp
#include "diffusion.hpp"
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <iostream>

namespace agx_emulsion {

static inline size_t idx_rgb(int y, int x, int w, int c) {
    return static_cast<size_t>((y * w + x) * 3 + c);
}

int Diffusion::reflect(int x, int n) {
    if (n <= 1) return 0;
    // SciPy's reflect: ... 3 2 1 0 0 1 2 3 | 3 2 1 0 0 1 ...
    while (x < 0 || x >= n) {
        if (x < 0) x = -x - 1;
        else       x = 2 * n - x - 1;
    }
    return x;
}

void Diffusion::gaussian_kernel_1d(float sigma, float truncate, std::vector<float>& k) {
    if (sigma <= 0.f) { k = {1.f}; return; }
    int radius = static_cast<int>(std::ceil(truncate * sigma));
    int size = 2 * radius + 1;
    k.resize(size);
    const float s2 = 2.0f * sigma * sigma;
    float sum = 0.f;
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-(i*i) / s2);
        k[i + radius] = v;
        sum += v;
    }
    for (float& v : k) v /= sum;
}

void Diffusion::convolve_separable_channel(const float* in, float* tmp, float* out,
                                           int height, int width,
                                           const std::vector<float>& k) {
    const int radius = static_cast<int>(k.size() / 2);

    // Horizontal
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.f;
            for (int t = -radius; t <= radius; ++t) {
                int xr = reflect(x + t, width);
                acc += in[y * width + xr] * k[t + radius];
            }
            tmp[y * width + x] = acc;
        }
    }

    // Vertical
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.f;
            for (int t = -radius; t <= radius; ++t) {
                int yr = reflect(y + t, height);
                acc += tmp[yr * width + x] * k[t + radius];
            }
            out[y * width + x] = acc;
        }
    }
}

void Diffusion::blur_all_channels_cpu(const std::vector<float>& image,
                                      int height, int width,
                                      float sigma, float truncate,
                                      std::vector<float>& output) {
    if (sigma <= 0.f) { output = image; return; }

    std::vector<float> k;
    gaussian_kernel_1d(sigma, truncate, k);

    output.resize(image.size());
    std::vector<float> chan_in(width * height);
    std::vector<float> chan_tmp(width * height);
    std::vector<float> chan_out(width * height);

    for (int c = 0; c < 3; ++c) {
        // Extract channel
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                chan_in[y * width + x] = image[idx_rgb(y,x,width,c)];

        convolve_separable_channel(chan_in.data(), chan_tmp.data(), chan_out.data(),
                                   height, width, k);

        // Write back
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                output[idx_rgb(y,x,width,c)] = chan_out[y * width + x];
    }
}

void Diffusion::apply_gaussian_blur(const std::vector<float>& image,
                                    int height,
                                    int width,
                                    float sigma,
                                    std::vector<float>& output,
                                    float truncate,
                                    bool try_cuda) {
    if (sigma <= 0.f) { output = image; return; }

    // Build kernel
    std::vector<float> k;
    gaussian_kernel_1d(sigma, truncate, k);
    if (try_cuda) {
        output.resize(image.size());
        if (diffusion_cuda::gaussian_blur_rgb(image.data(), output.data(),
                                              height, width,
                                              k.data(), static_cast<int>(k.size()))) {
            return; // CUDA path succeeded
        }
        // else: fall back to CPU
    }
    blur_all_channels_cpu(image, height, width, sigma, truncate, output);
}

void Diffusion::apply_gaussian_blur_planar(const std::vector<float>& plane,
                                           int height,
                                           int width,
                                           float sigma,
                                           std::vector<float>& output,
                                           float truncate,
                                           bool try_cuda) {
    if (sigma <= 0.f) { output = plane; return; }
    std::vector<float> k; gaussian_kernel_1d(sigma, truncate, k);
    output.resize(height * width);
    if (try_cuda) {
        if (diffusion_cuda::gaussian_blur_planar(plane.data(), output.data(), height, width, k.data(), (int)k.size())) {
            return;
        }
    }
    std::vector<float> tmp(height * width);
    const int radius = (int)k.size() / 2;
    // Horizontal
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.f;
            for (int t = -radius; t <= radius; ++t) {
                int xr = reflect(x + t, width);
                acc += plane[y * width + xr] * k[t + radius];
            }
            tmp[y * width + x] = acc;
        }
    }
    // Vertical
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            float acc = 0.f;
            for (int t = -radius; t <= radius; ++t) {
                int yr = reflect(y + t, height);
                acc += tmp[yr * width + x] * k[t + radius];
            }
            output[y * width + x] = acc;
        }
    }
}

void Diffusion::apply_gaussian_blur_um(const std::vector<float>& image,
                                       int height,
                                       int width,
                                       float sigma_um,
                                       float pixel_size_um,
                                       std::vector<float>& output,
                                       float truncate,
                                       bool try_cuda) {
    float sigma_px = (pixel_size_um > 0.f) ? (sigma_um / pixel_size_um) : 0.f;
    apply_gaussian_blur(image, height, width, sigma_px, output, truncate, try_cuda);
}

void Diffusion::apply_unsharp_mask(const std::vector<float>& image,
                                   int height,
                                   int width,
                                   float sigma,
                                   float amount,
                                   std::vector<float>& output) {
    std::vector<float> blur;
    apply_gaussian_blur(image, height, width, sigma, blur, /*truncate=*/4.0f);
    output.resize(image.size());
    for (size_t i = 0; i < image.size(); ++i) {
        output[i] = image[i] + amount * (image[i] - blur[i]);
    }
}

void Diffusion::apply_halation_um(std::vector<float>& raw,
                                  int height,
                                  int width,
                                  const HalationParams& halation,
                                  float pixel_size_um) {
    if (!halation.active) return;

    // Work buffers
    std::vector<float> chan_in(width * height);
    std::vector<float> chan_blur(width * height);

    // Halation (truncate=7). Use efficient planar blur per channel.
    for (int c = 0; c < 3; ++c) {
        float strength = halation.strength[c];
        float sigma_um = halation.size_um[c];
        if (strength <= 0.f || sigma_um <= 0.f) continue;

        float sigma_px = sigma_um / pixel_size_um;
        
        // Clamp extremely large kernels to prevent hangs
        const float max_sigma_px = 50.0f; // Reasonable maximum
        if (sigma_px > max_sigma_px) {
            std::cerr << "Warning: Halation sigma_px=" << sigma_px << " clamped to " << max_sigma_px << std::endl;
            sigma_px = max_sigma_px;
        }

        // Extract channel to planar buffer
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                chan_in[y * width + x] = raw[idx_rgb(y,x,width,c)];

        // Use efficient planar blur (CUDA if available, CPU fallback)
        apply_gaussian_blur_planar(chan_in, height, width, sigma_px, chan_blur, /*truncate=*/7.0f, /*try_cuda=*/true);

        // Apply: raw[:,:,c] = (raw[:,:,c] + s * blur) / (1 + s)
        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                size_t i = idx_rgb(y,x,width,c);
                float v = raw[i];
                v += strength * chan_blur[y * width + x];
                v /= (1.f + strength);
                raw[i] = v;
            }
    }

    // Scattering (truncate=7)
    for (int c = 0; c < 3; ++c) {
        float strength = halation.scattering_strength[c];
        float sigma_um = halation.scattering_size_um[c];
        if (strength <= 0.f || sigma_um <= 0.f) continue;

        float sigma_px = sigma_um / pixel_size_um;
        
        // Clamp scattering kernels too
        const float max_sigma_px = 20.0f; // Smaller limit for scattering
        if (sigma_px > max_sigma_px) {
            std::cerr << "Warning: Scattering sigma_px=" << sigma_px << " clamped to " << max_sigma_px << std::endl;
            sigma_px = max_sigma_px;
        }

        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x)
                chan_in[y * width + x] = raw[idx_rgb(y,x,width,c)];

        apply_gaussian_blur_planar(chan_in, height, width, sigma_px, chan_blur, /*truncate=*/7.0f, /*try_cuda=*/true);

        for (int y = 0; y < height; ++y)
            for (int x = 0; x < width; ++x) {
                size_t i = idx_rgb(y,x,width,c);
                float v = raw[i];
                v += strength * chan_blur[y * width + x];
                v /= (1.f + strength);
                raw[i] = v;
            }
    }
}

void Diffusion::apply_gaussian_blur_cuda_only(const std::vector<float>& image,
                                              int height,
                                              int width,
                                              float sigma,
                                              std::vector<float>& output,
                                              float truncate) {
    if (sigma <= 0.f) { output = image; return; }
    std::vector<float> k; gaussian_kernel_1d(sigma, truncate, k);
    output.resize(image.size());
    if (!diffusion_cuda::gaussian_blur_rgb(image.data(), output.data(), height, width, k.data(), (int)k.size())) {
        throw std::runtime_error("CUDA gaussian_blur_rgb failed");
    }
}

void Diffusion::apply_gaussian_blur_um_cuda_only(const std::vector<float>& image,
                                                 int height,
                                                 int width,
                                                 float sigma_um,
                                                 float pixel_size_um,
                                                 std::vector<float>& output,
                                                 float truncate) {
    float sigma_px = (pixel_size_um > 0.f) ? (sigma_um / pixel_size_um) : 0.f;
    apply_gaussian_blur_cuda_only(image, height, width, sigma_px, output, truncate);
}

} // namespace agx_emulsion 