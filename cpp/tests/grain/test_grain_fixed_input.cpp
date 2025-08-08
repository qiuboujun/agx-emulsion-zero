#include "grain.hpp"
#include "fast_stats.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>

using namespace agx_emulsion;

void print_image_stats(const Image2D& img, const std::string& name) {
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

void print_image_data(const Image2D& img, const std::string& name, int max_elements = 20) {
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

float max_abs_diff(const Image2D& a, const Image2D& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels) return 1e9f;
    float m = 0.0f;
    for (size_t i = 0; i < a.data.size(); ++i)
        m = std::max(m, std::fabs(a.data[i] - b.data[i]));
    return m;
}

// Fixed array for deterministic testing (replaces RNG)
std::vector<float> get_fixed_random_values(size_t count) {
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

// Modified grain function that uses fixed arrays instead of RNG
Image2D layer_particle_model_fixed(const Image2D& density,
                                   float density_max,
                                   float n_particles_per_pixel,
                                   float grain_uniformity,
                                   uint64_t seed,
                                   float blur_particle) {
    assert(density.channels == 1 && "layer_particle_model expects 1-channel image.");
    const int W = density.width, H = density.height;

    const float od_particle = density_max / std::max(1.0f, n_particles_per_pixel);

    // Get fixed random values instead of using RNG
    std::vector<float> fixed_rands = get_fixed_random_values(W * H * 2); // Need 2 values per pixel
    size_t rand_idx = 0;

    Image2D out(W, H, 1);

    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float d = density.at(x, y, 0);
            float p = std::max(1e-6f, std::min(1.0f - 1e-6f, d / density_max));
            float saturation = 1.0f - p * grain_uniformity * (1.0f - 1e-6f);
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

            float val = float(developed) * od_particle * saturation;
            out.at(x, y, 0) = val;
        }
    }

    if (blur_particle > 0.0f) {
        const float sigma = blur_particle * std::sqrt(od_particle);
        Image2D tmp(W,H,1);
        Grain::gaussian_filter_2d(out, tmp, sigma);
        return tmp;
    }
    return out;
}

void test_fixed_input_comparison() {
    std::cout << "=== Grain Model: Fixed Input Comparison (No RNG) ===" << std::endl;
    std::cout << "=" << std::string(55, '=') << std::endl;
    
    // Test 1: Simple 2x2x1 fixed input
    std::cout << "\n1. Test Case: Simple 2x2x1 Fixed Input" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    Image2D fixed_input(2, 2, 1);
    // Fixed values: [0.5, 1.0, 1.5, 2.0]
    fixed_input.at(0, 0, 0) = 0.5f;
    fixed_input.at(1, 0, 0) = 1.0f;
    fixed_input.at(0, 1, 0) = 1.5f;
    fixed_input.at(1, 1, 0) = 2.0f;
    
    std::cout << "Input data: [0.5, 1.0, 1.5, 2.0]" << std::endl;
    print_image_stats(fixed_input, "Input image");
    
    // CPU computation with fixed arrays
    Image2D cpu_result = layer_particle_model_fixed(fixed_input,
                                                    /*density_max*/2.2f,
                                                    /*n_particles_per_pixel*/10.0f,
                                                    /*grain_uniformity*/0.98f,
                                                    /*seed*/42,  // Fixed seed
                                                    /*blur_particle*/0.0f);
    
    print_image_stats(cpu_result, "CPU result (fixed arrays)");
    print_image_data(cpu_result, "CPU result (all values)");
    
    // Test 2: Larger fixed pattern (4x4x1)
    std::cout << "\n\n2. Test Case: Larger Fixed Pattern (4x4x1)" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    Image2D larger_input(4, 4, 1);
    // Create a fixed pattern using mathematical functions
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            float val = 0.5f + 0.3f * std::sin(2.0f * M_PI * x / 4.0f) + 0.2f * std::cos(2.0f * M_PI * y / 4.0f);
            larger_input.at(x, y, 0) = val;
        }
    }
    
    std::cout << "Input pattern (4x4):" << std::endl;
    for (int y = 0; y < 4; ++y) {
        std::cout << "  ";
        for (int x = 0; x < 4; ++x) {
            std::cout << std::fixed << std::setprecision(3) << larger_input.at(x, y, 0) << " ";
        }
        std::cout << std::endl;
    }
    
    print_image_stats(larger_input, "Larger input");
    
    // CPU computation with fixed arrays
    Image2D cpu_larger = layer_particle_model_fixed(larger_input,
                                                    /*density_max*/2.2f,
                                                    /*n_particles_per_pixel*/10.0f,
                                                    /*grain_uniformity*/0.98f,
                                                    /*seed*/123,  // Fixed seed
                                                    /*blur_particle*/0.0f);
    
    print_image_stats(cpu_larger, "CPU larger result (fixed arrays)");
    print_image_data(cpu_larger, "CPU larger result (all values)");
    
    // Test 3: 3-channel fixed input
    std::cout << "\n\n3. Test Case: 3-Channel Fixed Input" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    Image2D rgb_input(2, 2, 3);
    // Fixed RGB values
    rgb_input.at(0, 0, 0) = 0.5f; rgb_input.at(0, 0, 1) = 0.6f; rgb_input.at(0, 0, 2) = 0.7f;
    rgb_input.at(1, 0, 0) = 1.0f; rgb_input.at(1, 0, 1) = 1.1f; rgb_input.at(1, 0, 2) = 1.2f;
    rgb_input.at(0, 1, 0) = 1.5f; rgb_input.at(0, 1, 1) = 1.6f; rgb_input.at(0, 1, 2) = 1.7f;
    rgb_input.at(1, 1, 0) = 2.0f; rgb_input.at(1, 1, 1) = 2.1f; rgb_input.at(1, 1, 2) = 2.2f;
    
    std::cout << "RGB input data:" << std::endl;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            std::cout << "  (" << x << "," << y << "): [" 
                      << std::fixed << std::setprecision(1) << rgb_input.at(x, y, 0) << ", "
                      << rgb_input.at(x, y, 1) << ", " << rgb_input.at(x, y, 2) << "]" << std::endl;
        }
    }
    
    print_image_stats(rgb_input, "RGB input");
    
    // CPU computation for RGB (using original function but with fixed seed)
    Image2D cpu_rgb = Grain::apply_grain_to_density(rgb_input,
                                                    /*pixel_size_um*/10.0f,
                                                    /*agx_particle_area_um2*/0.2f,
                                                    /*agx_particle_scale*/{1.0f, 0.8f, 3.0f},
                                                    /*density_min*/{0.03f, 0.06f, 0.04f},
                                                    /*density_max_curves*/{2.2f, 2.2f, 2.2f},
                                                    /*grain_uniformity*/{0.98f, 0.98f, 0.98f},
                                                    /*grain_blur*/0.0f,
                                                    /*n_sub_layers*/1,
                                                    /*fixed_seed*/true);
    
    print_image_stats(cpu_rgb, "CPU RGB result");
    print_image_data(cpu_rgb, "CPU RGB result (all values)");
    
    // Test 4: FastStats integration
    std::cout << "\n\n4. Test Case: FastStats Integration" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    // Create a larger test image for statistics
    Image2D stats_input(8, 8, 1);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            float val = 0.5f + 0.3f * std::sin(2.0f * M_PI * x / 8.0f) + 0.2f * std::cos(2.0f * M_PI * y / 8.0f);
            stats_input.at(x, y, 0) = val;
        }
    }
    
    // Compute grain with fixed arrays
    Image2D grain_result = layer_particle_model_fixed(stats_input,
                                                      /*density_max*/2.2f,
                                                      /*n_particles_per_pixel*/10.0f,
                                                      /*grain_uniformity*/0.98f,
                                                      /*seed*/42,
                                                      /*blur_particle*/0.0f);
    
    // Use FastStats to compute statistics
    auto [mean, stddev] = FastStats::mean_stddev(grain_result.data);
    
    std::cout << "Grain result statistics (using FastStats):" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(15) << mean << std::endl;
    std::cout << "  Std:  " << std::fixed << std::setprecision(15) << stddev << std::endl;
    
    print_image_stats(grain_result, "Grain result");
    
    std::cout << "\n" << std::string(55, '=') << std::endl;
    std::cout << "Fixed input comparison complete (No RNG)!" << std::endl;
}

int main() {
    test_fixed_input_comparison();
    return 0;
} 