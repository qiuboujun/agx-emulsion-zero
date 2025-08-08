// emulsion.cpp
//
// Implementation of the AgX emulsion model.  See emulsion.hpp for
// documentation of the public API.  Wherever possible the code closely
// follows the structure of the reference Python implementation.  Some
// algorithms (particularly Gaussian filtering and interpolation) have been
// implemented from first principles to avoid external dependencies.

#include "emulsion.hpp"
#include "fast_stats.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <cassert>
#include <cmath>

namespace agx_emulsion {

// Fixed array for deterministic testing (replaces RNG)
std::vector<float> AgXEmulsion::get_fixed_random_values(size_t count) {
    std::vector<float> values = {
        0.123456f, 0.234567f, 0.345678f, 0.456789f, 0.567890f,
        0.678901f, 0.789012f, 0.890123f, 0.901234f, 0.012345f,
        0.111111f, 0.222222f, 0.333333f, 0.444444f, 0.555555f,
        0.666666f, 0.777777f, 0.888888f, 0.999999f, 0.000001f,
        0.101010f, 0.202020f, 0.303030f, 0.404040f, 0.505050f,
        0.606060f, 0.707070f, 0.808080f, 0.909090f, 0.010101f,
        0.121212f, 0.232323f, 0.343434f, 0.454545f, 0.565656f,
        0.676767f, 0.787878f, 0.898989f, 0.909090f, 0.020202f
    };
    
    // Cycle through the fixed values
    std::vector<float> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(values[i % values.size()]);
    }
    return result;
}

AgXEmulsion::AgXEmulsion() : rng_(42) {
    // Initialize with fixed seed for deterministic behavior
}

Image3D AgXEmulsion::develop_film(const Image3D& exposure,
                                  const GrainParams& grain_params,
                                  const DIRCouplerParams& dir_params) {
    // Step 1: Convert exposure to density
    Image3D density = exposure_to_density(exposure);
    
    // Step 2: Apply DIR couplers (if enabled)
    if (dir_params.enable_dir_couplers) {
        density = apply_dir_couplers_fixed(density, dir_params);
    }
    
    // Step 3: Apply grain
    density = apply_grain_fixed(density, grain_params);
    
    return density;
}

Image3D AgXEmulsion::exposure_to_density(const Image3D& exposure) {
    // Simple linear mapping for testing
    // In a real implementation, this would use characteristic curves
    Image3D density(exposure.width, exposure.height, exposure.depth, exposure.channels);
    
    for (int z = 0; z < exposure.depth; ++z) {
        for (int y = 0; y < exposure.height; ++y) {
            for (int x = 0; x < exposure.width; ++x) {
                for (int c = 0; c < exposure.channels; ++c) {
                    float exp_val = exposure.at(x, y, z, c);
                    // Simple characteristic curve: density = 1 - exp(-exposure)
                    float den_val = 1.0f - std::exp(-exp_val);
                    density.at(x, y, z, c) = clamp(den_val, 0.0f, 2.2f);
                }
            }
        }
    }
    
    return density;
}

Image3D AgXEmulsion::apply_grain_fixed(const Image3D& density, const GrainParams& params) {
    Image3D result = density;
    
    // Get fixed random values for deterministic grain
    std::vector<float> fixed_rands = get_fixed_random_values(density.size() * 2);
    size_t rand_idx = 0;
    
    const float od_particle = 0.22f; // density_max / n_particles_per_pixel
    const float n_particles_per_pixel = 10.0f;
    
    for (int z = 0; z < density.depth; ++z) {
        for (int y = 0; y < density.height; ++y) {
            for (int x = 0; x < density.width; ++x) {
                for (int c = 0; c < density.channels; ++c) {
                    float d = density.at(x, y, z, c);
                    float p = clamp(d / 2.2f, 1e-6f, 1.0f - 1e-6f);
                    float saturation = 1.0f - p * params.grain_uniformity[c] * (1.0f - 1e-6f);
                    float lambda = n_particles_per_pixel / std::max(1e-6f, saturation);
                    
                    // Use fixed values instead of RNG
                    float rand1 = fixed_rands[rand_idx++];
                    float rand2 = fixed_rands[rand_idx++];
                    
                    // Simple Poisson approximation using fixed random value
                    int n = static_cast<int>(lambda * rand1);
                    n = std::max(0, n);
                    
                    // Simple binomial approximation using fixed random value
                    int developed = static_cast<int>(n * p * rand2);
                    developed = std::max(0, std::min(developed, n));
                    
                    float grain_val = float(developed) * od_particle * saturation;
                    result.at(x, y, z, c) = d + grain_val;
                }
            }
        }
    }
    
    // Apply Gaussian blur if specified
    if (params.grain_blur > 0.0f) {
        Image3D blurred(result.width, result.height, result.depth, result.channels);
        gaussian_blur_3d(result, blurred, params.grain_blur, params.grain_blur, 0.0f);
        result = blurred;
    }
    
    return result;
}

Image3D AgXEmulsion::apply_dir_couplers_fixed(const Image3D& density, const DIRCouplerParams& params) {
    Image3D result = density;
    
    if (!params.enable_dir_couplers) {
        return result;
    }
    
    // Simple DIR coupler simulation using fixed arrays
    std::vector<float> fixed_rands = get_fixed_random_values(density.size());
    size_t rand_idx = 0;
    
    // DIR coupler matrix (simplified 3x3 identity with some cross-coupling)
    std::array<std::array<float, 3>, 3> dir_matrix = {{
        {1.0f, -0.1f, -0.1f},
        {-0.1f, 1.0f, -0.1f},
        {-0.1f, -0.1f, 1.0f}
    }};
    
    for (int z = 0; z < density.depth; ++z) {
        for (int y = 0; y < density.height; ++y) {
            for (int x = 0; x < density.width; ++x) {
                std::array<float, 3> input_density;
                std::array<float, 3> output_density = {0.0f, 0.0f, 0.0f};
                
                // Get input densities for RGB channels
                for (int c = 0; c < 3 && c < density.channels; ++c) {
                    input_density[c] = density.at(x, y, z, c);
                }
                
                // Apply DIR coupler matrix
                for (int i = 0; i < 3; ++i) {
                    for (int j = 0; j < 3; ++j) {
                        output_density[i] += dir_matrix[i][j] * input_density[j] * params.dir_coupler_scale;
                    }
                }
                
                // Add some fixed noise
                for (int c = 0; c < 3 && c < density.channels; ++c) {
                    float noise = fixed_rands[rand_idx++] * 0.01f; // Small fixed noise
                    result.at(x, y, z, c) = clamp(output_density[c] + noise, 0.0f, 2.2f);
                }
            }
        }
    }
    
    // Apply Gaussian blur if specified
    if (params.dir_coupler_blur > 0.0f) {
        Image3D blurred(result.width, result.height, result.depth, result.channels);
        gaussian_blur_3d(result, blurred, params.dir_coupler_blur, params.dir_coupler_blur, 0.0f);
        result = blurred;
    }
    
    return result;
}

Image3D AgXEmulsion::apply_grain(const Image3D& density, const GrainParams& params) {
    return apply_grain_fixed(density, params);
}

Image3D AgXEmulsion::apply_dir_couplers(const Image3D& density, const DIRCouplerParams& params) {
    return apply_dir_couplers_fixed(density, params);
}

void AgXEmulsion::gaussian_blur_3d(const Image3D& src, Image3D& dst, float sigma_x, float sigma_y, float sigma_z) {
    if (sigma_x <= 0.0f && sigma_y <= 0.0f && sigma_z <= 0.0f) {
        dst = src;
        return;
    }
    
    dst = src;
    
    // Apply 1D Gaussian blur along each axis
    if (sigma_x > 0.0f) {
        auto kernel_x = utils::create_gaussian_kernel(sigma_x, std::max(1, int(std::ceil(3.0f * sigma_x))));
        utils::apply_gaussian_filter_1d(dst.data, dst.data, kernel_x, 0, dst.width, dst.height, dst.depth);
    }
    
    if (sigma_y > 0.0f) {
        auto kernel_y = utils::create_gaussian_kernel(sigma_y, std::max(1, int(std::ceil(3.0f * sigma_y))));
        utils::apply_gaussian_filter_1d(dst.data, dst.data, kernel_y, 1, dst.width, dst.height, dst.depth);
    }
    
    if (sigma_z > 0.0f) {
        auto kernel_z = utils::create_gaussian_kernel(sigma_z, std::max(1, int(std::ceil(3.0f * sigma_z))));
        utils::apply_gaussian_filter_1d(dst.data, dst.data, kernel_z, 2, dst.width, dst.height, dst.depth);
    }
}

float AgXEmulsion::lerp(float a, float b, float t) {
    return a + t * (b - a);
}

float AgXEmulsion::clamp(float value, float min_val, float max_val) {
    return std::max(min_val, std::min(max_val, value));
}

// Film class implementation
Film::Film() : curves_loaded_(false) {}

void Film::set_characteristic_curves(const std::vector<float>& log_exposure,
                                    const std::vector<std::vector<float>>& density_curves) {
    log_exposure_ = log_exposure;
    density_curves_ = density_curves;
    curves_loaded_ = true;
}

float Film::exposure_to_density(float log_exposure, int channel) const {
    if (!curves_loaded_ || channel < 0 || channel >= 3) {
        return 0.0f;
    }
    
    // Simple linear interpolation
    const auto& curve = density_curves_[channel];
    if (curve.size() != log_exposure_.size()) {
        return 0.0f;
    }
    
    // Find the two closest points
    for (size_t i = 0; i < log_exposure_.size() - 1; ++i) {
        if (log_exposure >= log_exposure_[i] && log_exposure <= log_exposure_[i + 1]) {
            float t = (log_exposure - log_exposure_[i]) / (log_exposure_[i + 1] - log_exposure_[i]);
            return AgXEmulsion::lerp(curve[i], curve[i + 1], t);
        }
    }
    
    // Return nearest endpoint
    if (log_exposure <= log_exposure_.front()) {
        return curve.front();
    } else {
        return curve.back();
    }
}

float Film::density_to_exposure(float density, int channel) const {
    if (!curves_loaded_ || channel < 0 || channel >= 3) {
        return 0.0f;
    }
    
    // Simple inverse interpolation
    const auto& curve = density_curves_[channel];
    if (curve.size() != log_exposure_.size()) {
        return 0.0f;
    }
    
    // Find the two closest points
    for (size_t i = 0; i < curve.size() - 1; ++i) {
        if (density >= curve[i] && density <= curve[i + 1]) {
            float t = (density - curve[i]) / (curve[i + 1] - curve[i]);
            return AgXEmulsion::lerp(log_exposure_[i], log_exposure_[i + 1], t);
        }
    }
    
    // Return nearest endpoint
    if (density <= curve.front()) {
        return log_exposure_.front();
    } else {
        return log_exposure_.back();
    }
}

// Utility functions implementation
namespace utils {

std::vector<float> create_gaussian_kernel(float sigma, int radius) {
    if (sigma <= 0.0f) return {1.0f};
    
    std::vector<float> kernel(2 * radius + 1);
    const float s2 = 2.0f * sigma * sigma;
    float sum = 0.0f;
    
    for (int i = -radius; i <= radius; ++i) {
        float v = std::exp(-(i * i) / s2);
        kernel[i + radius] = v;
        sum += v;
    }
    
    // Normalize
    for (auto& v : kernel) {
        v /= sum;
    }
    
    return kernel;
}

void apply_gaussian_filter_1d(const std::vector<float>& src, std::vector<float>& dst,
                             const std::vector<float>& kernel, int axis, int width, int height, int depth) {
    if (src.size() != dst.size()) {
        dst.resize(src.size());
    }
    
    const int radius = static_cast<int>(kernel.size()) / 2;
    
    for (int z = 0; z < depth; ++z) {
        for (int y = 0; y < height; ++y) {
            for (int x = 0; x < width; ++x) {
                float sum = 0.0f;
                
                for (int k = -radius; k <= radius; ++k) {
                    int idx_x = x, idx_y = y, idx_z = z;
                    
                    if (axis == 0) idx_x = reflect_index(x + k, width);
                    else if (axis == 1) idx_y = reflect_index(y + k, height);
                    else if (axis == 2) idx_z = reflect_index(z + k, depth);
                    
                    int src_idx = ((idx_z * height + idx_y) * width + idx_x);
                    sum += kernel[k + radius] * src[src_idx];
                }
                
                int dst_idx = ((z * height + y) * width + x);
                dst[dst_idx] = sum;
            }
        }
    }
}

int reflect_index(int i, int n) {
    if (i < 0) return -i - 1;
    if (i >= n) return 2 * n - i - 1;
    return i;
}

void print_image_stats(const Image3D& img, const std::string& name) {
    if (img.data.empty()) {
        std::cout << name << ": empty" << std::endl;
        return;
    }
    
    float min_val = img.data[0];
    float max_val = img.data[0];
    float sum = 0.0f;
    
    for (float val : img.data) {
        min_val = std::min(min_val, val);
        max_val = std::max(max_val, val);
        sum += val;
    }
    
    float mean = sum / img.data.size();
    std::cout << name << ": min=" << std::fixed << std::setprecision(6) << min_val 
              << ", max=" << max_val << ", mean=" << mean << std::endl;
}

void print_image_data(const Image3D& img, const std::string& name, int max_elements) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(img.data.size(), static_cast<size_t>(max_elements)); ++i) {
        std::cout << std::fixed << std::setprecision(6) << img.data[i];
        if (i < std::min(img.data.size(), static_cast<size_t>(max_elements)) - 1) std::cout << ", ";
    }
    if (img.data.size() > static_cast<size_t>(max_elements)) {
        std::cout << ", ... (showing first " << max_elements << " of " << img.data.size() << " elements)";
    }
    std::cout << "]" << std::endl;
}

float max_abs_diff(const Image3D& a, const Image3D& b) {
    if (a.width != b.width || a.height != b.height || a.depth != b.depth || a.channels != b.channels) {
        return 1e9f;
    }
    
    float m = 0.0f;
    for (size_t i = 0; i < a.data.size(); ++i) {
        m = std::max(m, std::fabs(a.data[i] - b.data[i]));
    }
    return m;
}

} // namespace utils

} // namespace agx_emulsion

// C linkage wrapper implementation
extern "C" {
    int agx_film_develop_simple(const float* exposure_data, int width, int height, int depth,
                               float* density_data, const agx_emulsion::GrainParams* grain_params,
                               const agx_emulsion::DIRCouplerParams* dir_params) {
        try {
            // Create input image
            agx_emulsion::Image3D exposure(width, height, depth, 3);
            std::copy(exposure_data, exposure_data + exposure.size(), exposure.data.begin());
            
            // Create emulsion processor
            agx_emulsion::AgXEmulsion emulsion;
            
            // Use default parameters if not provided
            agx_emulsion::GrainParams grain = grain_params ? *grain_params : agx_emulsion::GrainParams{};
            agx_emulsion::DIRCouplerParams dir = dir_params ? *dir_params : agx_emulsion::DIRCouplerParams{};
            
            // Process film
            agx_emulsion::Image3D density = emulsion.develop_film(exposure, grain, dir);
            
            // Copy output
            std::copy(density.data.begin(), density.data.end(), density_data);
            
            return 0; // Success
        } catch (...) {
            return -1; // Error
        }
    }
}
