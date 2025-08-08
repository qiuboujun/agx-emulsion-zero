#pragma once
#include <vector>
#include <array>
#include <cstdint>
#include <cmath>

namespace agx_emulsion {

struct Image2D {
    int width = 0;
    int height = 0;
    int channels = 1;                // 1 or 3
    std::vector<float> data;         // size = width * height * channels

    Image2D() = default;
    Image2D(int w, int h, int c = 1) : width(w), height(h), channels(c), data(w*h*c, 0.0f) {}

    inline float& at(int x, int y, int c = 0) {
        return data[(y * width + x) * channels + c];
    }
    inline const float& at(int x, int y, int c = 0) const {
        return data[(y * width + x) * channels + c];
    }
};

namespace Grain {

// --------------------------- CPU API --------------------------- //

/**
 * CPU version of the particle model:
 *  - probability = density / density_max  (clamped to [1e-6, 1-1e-6])
 *  - saturation  = 1 - probability * grain_uniformity * (1-1e-6)
 *  - seeds ~ Poisson(n_particles_per_pixel / saturation)
 *  - grain  ~ Binomial(seeds, probability) * (density_max / n_particles_per_pixel) * saturation
 *  - optional Gaussian blur with sigma = blur_particle * sqrt(od_particle)
 */
Image2D layer_particle_model(const Image2D& density,
                             float density_max = 2.2f,
                             float n_particles_per_pixel = 10.0f,
                             float grain_uniformity = 0.98f,
                             uint64_t seed = 0,
                             float blur_particle = 0.0f);

/**
 * Apply grain to 3-channel density (RGB) similarly to the Python function.
 *   - Adds density_min, runs sublayers with fixed seeds [0,1,2] + sl*10,
 *     averages, then subtracts density_min.
 *   - Optional Gaussian blur across X,Y only.
 */
Image2D apply_grain_to_density(const Image2D& density_cmy,
                               float pixel_size_um = 10.0f,
                               float agx_particle_area_um2 = 0.2f,
                               std::array<float,3> agx_particle_scale = {1.0f, 0.8f, 3.0f},
                               std::array<float,3> density_min = {0.03f, 0.06f, 0.04f},
                               std::array<float,3> density_max_curves = {2.2f, 2.2f, 2.2f},
                               std::array<float,3> grain_uniformity = {0.98f, 0.98f, 0.98f},
                               float grain_blur = 1.0f,
                               int n_sub_layers = 1,
                               bool fixed_seed = false);

/**
 * Experimental layered model (x,y,sublayers,rgb) with micro-structure.
 */
Image2D apply_grain_to_density_layers(const Image2D& density_cmy_layers, // shape: W x H x (3*3) flattened as [sl + 3*ch]
                                      const std::array<std::array<float,3>,3>& density_max_layers, // [sublayer][rgb]
                                      float pixel_size_um = 10.0f,
                                      float agx_particle_area_um2 = 0.2f,
                                      std::array<float,3> agx_particle_scale = {1.0f, 0.8f, 3.0f},
                                      std::array<float,3> agx_particle_scale_layers = {3.0f, 1.0f, 0.3f},
                                      std::array<float,3> density_min = {0.03f, 0.06f, 0.04f},
                                      std::array<float,3> grain_uniformity = {0.98f, 0.98f, 0.98f},
                                      float grain_blur = 1.0f,
                                      float grain_blur_dye_clouds_um = 1.0f,
                                      std::pair<float, float> grain_micro_structure = {0.1f, 30.0f}, // (blur_um, sigma_nm)
                                      bool fixed_seed = false,
                                      bool use_fast_stats = false);

// ----------------------- Utility (CPU) ------------------------- //

// Separable Gaussian blur (reflect boundary). Applies to each channel independently.
void gaussian_filter_2d(const Image2D& src, Image2D& dst, float sigma);

// Draw a lognormal "clumping" field with given sigma (in normalized pixel units).
// If blur_sigma > 0, blur the field before applying.
void apply_micro_structure(Image2D& rgb, float pixel_size_um,
                           std::pair<float,float> micro,  // (blur_um, sigma_nm)
                           uint64_t seed);

// --------------------------- CUDA API -------------------------- //
// GPU-accelerated particle model (blur is ignored on GPU for simplicity).
Image2D layer_particle_model_cuda(const Image2D& density,
                                  float density_max = 2.2f,
                                  float n_particles_per_pixel = 10.0f,
                                  float grain_uniformity = 0.98f,
                                  uint64_t seed = 0,
                                  float blur_particle = 0.0f);

} // namespace Grain

} // namespace agx_emulsion 