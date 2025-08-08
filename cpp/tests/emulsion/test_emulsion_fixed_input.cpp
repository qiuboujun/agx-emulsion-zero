#include "emulsion.hpp"
#include "fast_stats.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <vector>
#include <random>
#include <cassert>
#include <algorithm>

using namespace agx_emulsion;

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

void print_image_data(const Image3D& img, const std::string& name, int max_elements = 20) {
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

void test_fixed_input_comparison() {
    std::cout << "=== Emulsion Model: Fixed Input Comparison (No RNG) ===" << std::endl;
    std::cout << "=" << std::string(55, '=') << std::endl;
    
    // Test 1: Simple 2x2x1x3 fixed input
    std::cout << "\n1. Test Case: Simple 2x2x1x3 Fixed Input" << std::endl;
    std::cout << std::string(45, '-') << std::endl;
    
    Image3D fixed_input(2, 2, 1, 3);
    // Fixed RGB values: [0.5,0.6,0.7], [1.0,1.1,1.2], [1.5,1.6,1.7], [2.0,2.1,2.2]
    fixed_input.at(0, 0, 0, 0) = 0.5f; fixed_input.at(0, 0, 0, 1) = 0.6f; fixed_input.at(0, 0, 0, 2) = 0.7f;
    fixed_input.at(1, 0, 0, 0) = 1.0f; fixed_input.at(1, 0, 0, 1) = 1.1f; fixed_input.at(1, 0, 0, 2) = 1.2f;
    fixed_input.at(0, 1, 0, 0) = 1.5f; fixed_input.at(0, 1, 0, 1) = 1.6f; fixed_input.at(0, 1, 0, 2) = 1.7f;
    fixed_input.at(1, 1, 0, 0) = 2.0f; fixed_input.at(1, 1, 0, 1) = 2.1f; fixed_input.at(1, 1, 0, 2) = 2.2f;
    
    std::cout << "Input data (RGB):" << std::endl;
    for (int y = 0; y < 2; ++y) {
        for (int x = 0; x < 2; ++x) {
            std::cout << "  (" << x << "," << y << "): [" 
                      << std::fixed << std::setprecision(1) << fixed_input.at(x, y, 0, 0) << ", "
                      << fixed_input.at(x, y, 0, 1) << ", " << fixed_input.at(x, y, 0, 2) << "]" << std::endl;
        }
    }
    
    print_image_stats(fixed_input, "Input image");
    
    // Create emulsion processor
    AgXEmulsion emulsion;
    
    // Test parameters
    GrainParams grain_params;
    grain_params.pixel_size_um = 10.0f;
    grain_params.agx_particle_area_um2 = 0.2f;
    grain_params.agx_particle_scale = {1.0f, 0.8f, 3.0f};
    grain_params.density_min = {0.03f, 0.06f, 0.04f};
    grain_params.density_max_curves = {2.2f, 2.2f, 2.2f};
    grain_params.grain_uniformity = {0.98f, 0.98f, 0.98f};
    grain_params.grain_blur = 0.0f; // No blur for exact comparison
    grain_params.n_sub_layers = 1;
    grain_params.fixed_seed = true;
    grain_params.seed = 42;
    
    DIRCouplerParams dir_params;
    dir_params.dir_coupler_scale = 1.0f;
    dir_params.dir_coupler_blur = 0.0f; // No blur for exact comparison
    dir_params.enable_dir_couplers = true;
    dir_params.seed = 123;
    
    // CPU computation with fixed arrays
    Image3D cpu_result = emulsion.develop_film(fixed_input, grain_params, dir_params);
    
    print_image_stats(cpu_result, "CPU result (fixed arrays)");
    print_image_data(cpu_result, "CPU result (all values)");
    
    // Test 2: Larger fixed pattern (4x4x1x3)
    std::cout << "\n\n2. Test Case: Larger Fixed Pattern (4x4x1x3)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    Image3D larger_input(4, 4, 1, 3);
    // Create a fixed pattern using mathematical functions
    for (int y = 0; y < 4; ++y) {
        for (int x = 0; x < 4; ++x) {
            for (int c = 0; c < 3; ++c) {
                float val = 0.5f + 0.3f * std::sin(2.0f * M_PI * x / 4.0f) + 
                           0.2f * std::cos(2.0f * M_PI * y / 4.0f) + 
                           0.1f * (c + 1); // Different per channel
                larger_input.at(x, y, 0, c) = val;
            }
        }
    }
    
    std::cout << "Input pattern (4x4x3) - showing first channel:" << std::endl;
    for (int y = 0; y < 4; ++y) {
        std::cout << "  ";
        for (int x = 0; x < 4; ++x) {
            std::cout << std::fixed << std::setprecision(3) << larger_input.at(x, y, 0, 0) << " ";
        }
        std::cout << std::endl;
    }
    
    print_image_stats(larger_input, "Larger input");
    
    // CPU computation with fixed arrays
    Image3D cpu_larger = emulsion.develop_film(larger_input, grain_params, dir_params);
    
    print_image_stats(cpu_larger, "CPU larger result (fixed arrays)");
    print_image_data(cpu_larger, "CPU larger result (all values)");
    
    // Test 3: Individual component testing
    std::cout << "\n\n3. Test Case: Individual Component Testing" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    // Test exposure to density conversion
    Image3D density_only = emulsion.exposure_to_density(fixed_input);
    print_image_stats(density_only, "Exposure to density only");
    print_image_data(density_only, "Density values (all)");
    
    // Test grain only
    Image3D grain_only = emulsion.apply_grain(density_only, grain_params);
    print_image_stats(grain_only, "Grain only");
    print_image_data(grain_only, "Grain values (all)");
    
    // Test DIR couplers only
    Image3D dir_only = emulsion.apply_dir_couplers(density_only, dir_params);
    print_image_stats(dir_only, "DIR couplers only");
    print_image_data(dir_only, "DIR coupler values (all)");
    
    // Test 4: FastStats integration
    std::cout << "\n\n4. Test Case: FastStats Integration" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    // Create a larger test image for statistics
    Image3D stats_input(8, 8, 1, 3);
    for (int y = 0; y < 8; ++y) {
        for (int x = 0; x < 8; ++x) {
            for (int c = 0; c < 3; ++c) {
                float val = 0.5f + 0.3f * std::sin(2.0f * M_PI * x / 8.0f) + 
                           0.2f * std::cos(2.0f * M_PI * y / 8.0f) + 
                           0.1f * (c + 1);
                stats_input.at(x, y, 0, c) = val;
            }
        }
    }
    
    // Compute full emulsion development
    Image3D emulsion_result = emulsion.develop_film(stats_input, grain_params, dir_params);
    
    // Use FastStats to compute statistics for each channel
    for (int c = 0; c < 3; ++c) {
        std::vector<float> channel_data;
        channel_data.reserve(8 * 8);
        for (int y = 0; y < 8; ++y) {
            for (int x = 0; x < 8; ++x) {
                channel_data.push_back(emulsion_result.at(x, y, 0, c));
            }
        }
        
        auto [mean, stddev] = FastStats::mean_stddev(channel_data);
        
        std::cout << "Channel " << c << " statistics (using FastStats):" << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(15) << mean << std::endl;
        std::cout << "  Std:  " << std::fixed << std::setprecision(15) << stddev << std::endl;
    }
    
    print_image_stats(emulsion_result, "Full emulsion result");
    
    // Test 5: Film characteristic curves
    std::cout << "\n\n5. Test Case: Film Characteristic Curves" << std::endl;
    std::cout << std::string(35, '-') << std::endl;
    
    Film film;
    
    // Create simple characteristic curves
    std::vector<float> log_exposure = {-2.0f, -1.0f, 0.0f, 1.0f, 2.0f};
    std::vector<std::vector<float>> density_curves = {
        {0.1f, 0.3f, 0.8f, 1.5f, 2.0f}, // Red channel
        {0.1f, 0.4f, 1.0f, 1.8f, 2.1f}, // Green channel
        {0.1f, 0.2f, 0.6f, 1.2f, 1.9f}  // Blue channel
    };
    
    film.set_characteristic_curves(log_exposure, density_curves);
    
    // Test exposure to density conversion
    for (int c = 0; c < 3; ++c) {
        float exp_val = 0.5f;
        float den_val = film.exposure_to_density(exp_val, c);
        std::cout << "Channel " << c << ": exposure=" << exp_val 
                  << " -> density=" << std::fixed << std::setprecision(6) << den_val << std::endl;
    }
    
    std::cout << "\n" << std::string(55, '=') << std::endl;
    std::cout << "Fixed input comparison complete (No RNG)!" << std::endl;
}

int main() {
    test_fixed_input_comparison();
    return 0;
} 