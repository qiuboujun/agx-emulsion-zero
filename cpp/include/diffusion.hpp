#pragma once
// diffusion.hpp
// C++/CUDA port of agx_emulsion/model/diffusion.py core blur/halation logic.
// Images are interleaved RGB float32, row-major: [r,g,b, r,g,b, ...].

#include <vector>
#include <array>

namespace agx_emulsion {

struct HalationParams {
    bool active = false;
    std::array<float,3> size_um{{0.f,0.f,0.f}};
    std::array<float,3> strength{{0.f,0.f,0.f}};
    std::array<float,3> scattering_size_um{{0.f,0.f,0.f}};
    std::array<float,3> scattering_strength{{0.f,0.f,0.f}};
};

class Diffusion {
public:
    // Gaussian blur with sigma (in pixels). truncate controls the kernel radius
    // as radius = ceil(truncate * sigma), matching SciPy semantics.
    static void apply_gaussian_blur(const std::vector<float>& image,
                                    int height,
                                    int width,
                                    float sigma,
                                    std::vector<float>& output,
                                    float truncate = 4.0f,
                                    bool try_cuda = true);

    // Gaussian blur with sigma expressed in micrometres -> converted using pixel_size_um.
    static void apply_gaussian_blur_um(const std::vector<float>& image,
                                       int height,
                                       int width,
                                       float sigma_um,
                                       float pixel_size_um,
                                       std::vector<float>& output,
                                       float truncate = 4.0f,
                                       bool try_cuda = true);

    // Unsharp mask: image + amount * (image - gaussian_blur(image, sigma))
    static void apply_unsharp_mask(const std::vector<float>& image,
                                   int height,
                                   int width,
                                   float sigma,
                                   float amount,
                                   std::vector<float>& output);

    // CUDA-only Gaussian blur (pixel sigma). Throws on failure; no CPU fallback.
    static void apply_gaussian_blur_cuda_only(const std::vector<float>& image,
                                              int height,
                                              int width,
                                              float sigma,
                                              std::vector<float>& output,
                                              float truncate = 4.0f);

    // CUDA-only Gaussian blur (micrometre sigma). Throws on failure; no CPU fallback.
    static void apply_gaussian_blur_um_cuda_only(const std::vector<float>& image,
                                                 int height,
                                                 int width,
                                                 float sigma_um,
                                                 float pixel_size_um,
                                                 std::vector<float>& output,
                                                 float truncate = 4.0f);

    // Planar single-channel Gaussian blur helper (used to accelerate halation).
    static void apply_gaussian_blur_planar(const std::vector<float>& plane,
                                           int height,
                                           int width,
                                           float sigma,
                                           std::vector<float>& output,
                                           float truncate = 7.0f,
                                           bool try_cuda = true);

    // In-place halation/scattering in micrometres (matches diffusion.py):
    //   raw[:,:,i] += strength[i] * G(raw[:,:,i], size_pixel[i]); raw[:,:,i] /= (1+strength[i])
    // Then repeat for scattering_* with truncate=7 in both cases.
    static void apply_halation_um(std::vector<float>& raw,
                                  int height,
                                  int width,
                                  const HalationParams& halation,
                                  float pixel_size_um);

private:
    // CPU helpers
    static void gaussian_kernel_1d(float sigma, float truncate, std::vector<float>& k);
    static inline int reflect(int x, int n);
    static void convolve_separable_channel(const float* in, float* tmp, float* out,
                                           int height, int width,
                                           const std::vector<float>& k);

    static void blur_all_channels_cpu(const std::vector<float>& image,
                                      int height, int width,
                                      float sigma, float truncate,
                                      std::vector<float>& output);
};

// CUDA hooks (implemented in diffusion.cu)
namespace diffusion_cuda {
    // Returns true if CUDA executed successfully; false if not available.
    bool gaussian_blur_rgb(const float* in, float* out,
                           int height, int width,
                           const float* k1d, int ksize);
    bool gaussian_blur_planar(const float* in, float* out,
                              int height, int width,
                              const float* k1d, int ksize);
} // namespace diffusion_cuda

} // namespace agx_emulsion 