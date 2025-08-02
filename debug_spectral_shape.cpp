#include "cpp/include/config.hpp"
#include "cpp/include/color_filters.hpp"
#include <iostream>

int main() {
    std::cout << "Debug Spectral Shape" << std::endl;
    std::cout << "====================" << std::endl;
    
    // Initialize configuration
    agx::config::initialize_config();
    
    std::cout << "SPECTRAL_SHAPE.wavelengths size: " << agx::config::SPECTRAL_SHAPE.wavelengths.size() << std::endl;
    std::cout << "SPECTRAL_SHAPE.wavelengths shape: (" << agx::config::SPECTRAL_SHAPE.wavelengths.shape().rows 
              << ", " << agx::config::SPECTRAL_SHAPE.wavelengths.shape().cols << ")" << std::endl;
    
    std::cout << "First 5 wavelengths: [";
    for (int i = 0; i < std::min(5, static_cast<int>(agx::config::SPECTRAL_SHAPE.wavelengths.size())); ++i) {
        std::cout << agx::config::SPECTRAL_SHAPE.wavelengths[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Last 5 wavelengths: [";
    int size = agx::config::SPECTRAL_SHAPE.wavelengths.size();
    for (int i = std::max(0, size - 5); i < size; ++i) {
        std::cout << agx::config::SPECTRAL_SHAPE.wavelengths[i];
        if (i < size - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Test sigmoid_erf directly
    std::cout << "\nTesting sigmoid_erf directly:" << std::endl;
    const auto& wl = agx::config::SPECTRAL_SHAPE.wavelengths;
    auto wl_flat = wl.flatten();
    std::cout << "wl_flat size: " << wl_flat.size() << std::endl;
    std::cout << "wl_flat shape: (" << wl_flat.shape().rows << ", " << wl_flat.shape().cols << ")" << std::endl;
    
    // Try to create a truly 1D array
    nc::NdArray<float> wl_1d(wl_flat.size());
    for (int i = 0; i < wl_flat.size(); ++i) {
        wl_1d[i] = wl_flat[i];
    }
    std::cout << "wl_1d size: " << wl_1d.size() << std::endl;
    std::cout << "wl_1d shape: (" << wl_1d.shape().rows << ", " << wl_1d.shape().cols << ")" << std::endl;
    
    auto sig_result = agx::model::sigmoid_erf(wl_1d, 410.0f, 8.0f);
    std::cout << "sigmoid_erf result size: " << sig_result.size() << std::endl;
    std::cout << "sigmoid_erf result shape: (" << sig_result.shape().rows << ", " << sig_result.shape().cols << ")" << std::endl;
    
    // Test compute_band_pass_filter
    std::cout << "\nTesting compute_band_pass_filter:" << std::endl;
    std::array<float, 3> filter_uv = {0.8f, 410.0f, 8.0f};
    std::array<float, 3> filter_ir = {0.6f, 675.0f, 15.0f};
    
    auto result = agx::model::compute_band_pass_filter(filter_uv, filter_ir);
    std::cout << "Result size: " << result.size() << std::endl;
    std::cout << "Result shape: (" << result.shape().rows << ", " << result.shape().cols << ")" << std::endl;
    
    return 0;
} 