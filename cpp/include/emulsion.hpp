// emulsion.hpp
//
// Header definitions for the AgX emulsion model in C++/CUDA.
//
// This file declares the core data structures and functions used to
// implement a photographic emulsion simulation.  The implementation in
// emulsion.cpp and emulsion.cu follow the logic of the reference
// Python implementation located in agx_emulsion/model/emulsion.py.  The
// C++ version is organised into a pair of classes (AgXEmulsion and
// Film) together with a number of free functions for interpolation,
// Gaussian filtering, coupler modelling and grain simulation.  All
// arrays are stored in contiguous vectors and accessed via simple
// helper classes.

#pragma once

#include <vector>
#include <array>
#include <memory>
#include <cstdint>
#include <cmath>
#include <random>

namespace agx_emulsion {

// Forward declarations
struct Image3D;
struct GrainParams;
struct DIRCouplerParams;

/**
 * @brief 3D image data structure for emulsion processing
 */
struct Image3D {
    int width = 0;
    int height = 0;
    int depth = 0;
    int channels = 1;
    std::vector<float> data;

    Image3D() = default;
    Image3D(int w, int h, int d, int c = 1) 
        : width(w), height(h), depth(d), channels(c), data(w * h * d * c, 0.0f) {}

    inline float& at(int x, int y, int z, int c = 0) {
        return data[((z * height + y) * width + x) * channels + c];
    }

    inline const float& at(int x, int y, int z, int c = 0) const {
        return data[((z * height + y) * width + x) * channels + c];
    }

    inline size_t size() const { return data.size(); }
    inline bool empty() const { return data.empty(); }
};

/**
 * @brief Parameters for grain simulation
 */
struct GrainParams {
    float pixel_size_um = 10.0f;
    float agx_particle_area_um2 = 0.2f;
    std::array<float, 3> agx_particle_scale = {1.0f, 0.8f, 3.0f};
    std::array<float, 3> density_min = {0.03f, 0.06f, 0.04f};
    std::array<float, 3> density_max_curves = {2.2f, 2.2f, 2.2f};
    std::array<float, 3> grain_uniformity = {0.98f, 0.98f, 0.98f};
    float grain_blur = 1.0f;
    int n_sub_layers = 1;
    bool fixed_seed = false;
    uint64_t seed = 42;
};

/**
 * @brief Parameters for DIR coupler simulation
 */
struct DIRCouplerParams {
    float dir_coupler_scale = 1.0f;
    float dir_coupler_blur = 0.0f;
    bool enable_dir_couplers = true;
    uint64_t seed = 123;
};

/**
 * @brief Main emulsion development class
 */
class AgXEmulsion {
public:
    AgXEmulsion();
    ~AgXEmulsion() = default;

    /**
     * @brief Develop film with grain and DIR couplers
     */
    Image3D develop_film(const Image3D& exposure,
                        const GrainParams& grain_params = GrainParams{},
                        const DIRCouplerParams& dir_params = DIRCouplerParams{});

    /**
     * @brief Apply grain to density image
     */
    Image3D apply_grain(const Image3D& density, const GrainParams& params);

    /**
     * @brief Apply DIR couplers to density image
     */
    Image3D apply_dir_couplers(const Image3D& density, const DIRCouplerParams& params);

    /**
     * @brief Convert exposure to density using characteristic curves
     */
    Image3D exposure_to_density(const Image3D& exposure);

    /**
     * @brief Apply Gaussian blur to 3D image
     */
    static void gaussian_blur_3d(const Image3D& src, Image3D& dst, float sigma_x, float sigma_y, float sigma_z);

    /**
     * @brief Linear interpolation between two values
     */
    static float lerp(float a, float b, float t);

    /**
     * @brief Clamp value between min and max
     */
    static float clamp(float value, float min_val, float max_val);

    /**
     * @brief Generate fixed random values for deterministic testing
     */
    static std::vector<float> get_fixed_random_values(size_t count);

private:
    std::mt19937_64 rng_;
    
    /**
     * @brief Apply grain using fixed arrays (deterministic)
     */
    Image3D apply_grain_fixed(const Image3D& density, const GrainParams& params);
    
    /**
     * @brief Apply DIR couplers using fixed arrays (deterministic)
     */
    Image3D apply_dir_couplers_fixed(const Image3D& density, const DIRCouplerParams& params);
};

/**
 * @brief Film class for managing film characteristics
 */
class Film {
public:
    Film();
    ~Film() = default;

    /**
     * @brief Set characteristic curves for RGB channels
     */
    void set_characteristic_curves(const std::vector<float>& log_exposure,
                                  const std::vector<std::vector<float>>& density_curves);

    /**
     * @brief Convert log exposure to density for a channel
     */
    float exposure_to_density(float log_exposure, int channel) const;

    /**
     * @brief Convert density to log exposure for a channel
     */
    float density_to_exposure(float density, int channel) const;

private:
    std::vector<float> log_exposure_;
    std::vector<std::vector<float>> density_curves_;
    bool curves_loaded_ = false;
};

// Utility functions
namespace utils {

/**
 * @brief Create Gaussian kernel for 1D convolution
 */
std::vector<float> create_gaussian_kernel(float sigma, int radius);

/**
 * @brief Apply 1D Gaussian filter
 */
void apply_gaussian_filter_1d(const std::vector<float>& src, std::vector<float>& dst,
                             const std::vector<float>& kernel, int axis, int width, int height, int depth);

/**
 * @brief Reflect index for boundary handling
 */
int reflect_index(int i, int n);

/**
 * @brief Print image statistics
 */
void print_image_stats(const Image3D& img, const std::string& name);

/**
 * @brief Print image data (first few elements)
 */
void print_image_data(const Image3D& img, const std::string& name, int max_elements = 20);

/**
 * @brief Calculate maximum absolute difference between two images
 */
float max_abs_diff(const Image3D& a, const Image3D& b);

} // namespace utils

} // namespace agx_emulsion

// C linkage wrapper for external calling
extern "C" {
    /**
     * @brief Simple C interface for film development
     */
    int agx_film_develop_simple(const float* exposure_data, int width, int height, int depth,
                               float* density_data, const agx_emulsion::GrainParams* grain_params,
                               const agx_emulsion::DIRCouplerParams* dir_params);
}
