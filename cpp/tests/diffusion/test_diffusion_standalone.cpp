#include "diffusion.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>

using namespace agx_emulsion;

void print_vector(const std::vector<float>& vec, const std::string& name, int max_elements = 20) {
    std::cout << name << ": [";
    for (size_t i = 0; i < std::min(vec.size(), static_cast<size_t>(max_elements)); ++i) {
        std::cout << std::fixed << std::setprecision(6) << vec[i];
        if (i < std::min(vec.size(), static_cast<size_t>(max_elements)) - 1) std::cout << ", ";
    }
    if (vec.size() > static_cast<size_t>(max_elements)) {
        std::cout << ", ... (showing first " << max_elements << " of " << vec.size() << " elements)";
    }
    std::cout << "]" << std::endl;
}

void print_image_stats(const std::vector<float>& img, const std::string& name) {
    if (img.empty()) {
        std::cout << name << ": empty" << std::endl;
        return;
    }
    
    float min_val = img[0], max_val = img[0], sum = 0.0f;
    for (float v : img) {
        min_val = std::min(min_val, v);
        max_val = std::max(max_val, v);
        sum += v;
    }
    float mean = sum / img.size();
    
    std::cout << name << " stats: min=" << std::fixed << std::setprecision(6) << min_val 
              << ", max=" << max_val << ", mean=" << mean << ", size=" << img.size() << std::endl;
}

int main() {
    std::cout << "=== C++ Diffusion Test Results ===" << std::endl << std::endl;

    // Fixed test image (64x80x3 = 15360 elements)
    const int height = 64;
    const int width = 80;
    const int total_elements = height * width * 3;
    
    // Create fixed, predictable input data (no random numbers)
    std::vector<float> image(total_elements);
    for (int i = 0; i < total_elements; ++i) {
        // Create a predictable pattern: alternating values with some variation
        int pixel = i / 3;  // pixel index
        int channel = i % 3;  // RGB channel (0,1,2)
        
        // Base pattern: sine wave with different frequencies per channel
        float base_val = 0.5f + 0.3f * std::sin(pixel * 0.1f + channel * 0.5f);
        
        // Add some variation based on position
        float variation = 0.1f * std::sin(pixel * 0.05f) * std::cos(channel * 0.3f);
        
        // Ensure values are in [0, 1] range
        image[i] = std::max(0.0f, std::min(1.0f, base_val + variation));
    }
    
    print_image_stats(image, "Input image");
    print_vector(image, "Input image (first 20 elements)");
    
    std::cout << std::endl;

    // Test 1: Gaussian blur
    std::cout << "Test 1: apply_gaussian_blur" << std::endl;
    std::cout << "==========================" << std::endl;
    
    float sigma = 2.0f;
    std::vector<float> blurred;
    Diffusion::apply_gaussian_blur(image, height, width, sigma, blurred);
    
    print_image_stats(blurred, "Gaussian blurred image");
    print_vector(blurred, "Gaussian blurred image (first 20 elements)");
    
    std::cout << std::endl;

    // Test 2: Gaussian blur with micrometres
    std::cout << "Test 2: apply_gaussian_blur_um" << std::endl;
    std::cout << "=============================" << std::endl;
    
    float sigma_um = 3.25f;
    float pixel_um = 2.5f;  // => sigma_px = 1.3
    std::vector<float> blurred_um;
    Diffusion::apply_gaussian_blur_um(image, height, width, sigma_um, pixel_um, blurred_um);
    
    print_image_stats(blurred_um, "Gaussian blurred (um) image");
    print_vector(blurred_um, "Gaussian blurred (um) image (first 20 elements)");
    
    std::cout << std::endl;

    // Test 3: Unsharp mask
    std::cout << "Test 3: apply_unsharp_mask" << std::endl;
    std::cout << "=========================" << std::endl;
    
    float unsharp_sigma = 1.5f;
    float unsharp_amount = 0.6f;
    std::vector<float> unsharped;
    Diffusion::apply_unsharp_mask(image, height, width, unsharp_sigma, unsharp_amount, unsharped);
    
    print_image_stats(unsharped, "Unsharp masked image");
    print_vector(unsharped, "Unsharp masked image (first 20 elements)");
    
    std::cout << std::endl;

    // Test 4: Halation
    std::cout << "Test 4: apply_halation_um" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Create halation parameters
    HalationParams halation;
    halation.active = true;
    halation.size_um = {5.0f, 3.0f, 2.0f};
    halation.strength = {0.10f, 0.05f, 0.00f};
    halation.scattering_size_um = {2.0f, 0.0f, 1.5f};
    halation.scattering_strength = {0.02f, 0.00f, 0.01f};
    
    float pixel_size_um = 2.5f;
    std::vector<float> halated = image;  // Copy input
    Diffusion::apply_halation_um(halated, height, width, halation, pixel_size_um);
    
    print_image_stats(halated, "Halated image");
    print_vector(halated, "Halated image (first 20 elements)");
    
    std::cout << std::endl;

    // Test 5: GPU vs CPU comparison
    std::cout << "Test 5: GPU vs CPU comparison" << std::endl;
    std::cout << "=============================" << std::endl;
    
    // Use the same blur method but with try_cuda=true to test GPU path
    std::vector<float> gpu_blurred;
    Diffusion::apply_gaussian_blur(image, height, width, sigma, gpu_blurred, 4.0f, true);
    
    print_image_stats(gpu_blurred, "GPU blurred image");
    print_vector(gpu_blurred, "GPU blurred image (first 20 elements)");
    
    // Compare with CPU result
    float max_diff = 0.0f;
    for (size_t i = 0; i < blurred.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(blurred[i] - gpu_blurred[i]));
    }
    std::cout << "Max absolute difference (CPU vs GPU): " << std::fixed << std::setprecision(15) << max_diff << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
} 