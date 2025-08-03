#include "cpp/include/config.hpp"
#include "cpp/include/colour.hpp"
#include <iostream>

int main() {
    std::cout << "=== Debug Spectral Shape ===" << std::endl;
    
    // Initialize config
    agx::config::initialize_config();
    
    std::cout << "SPECTRAL_SHAPE.start: " << agx::config::SPECTRAL_SHAPE.start << std::endl;
    std::cout << "SPECTRAL_SHAPE.end: " << agx::config::SPECTRAL_SHAPE.end << std::endl;
    std::cout << "SPECTRAL_SHAPE.interval: " << agx::config::SPECTRAL_SHAPE.interval << std::endl;
    std::cout << "SPECTRAL_SHAPE.wavelengths.size(): " << agx::config::SPECTRAL_SHAPE.wavelengths.size() << std::endl;
    std::cout << "SPECTRAL_SHAPE.wavelengths.shape(): [" << agx::config::SPECTRAL_SHAPE.wavelengths.shape().rows << ", " << agx::config::SPECTRAL_SHAPE.wavelengths.shape().cols << "]" << std::endl;
    
    std::cout << "First 5 wavelengths: ";
    for (unsigned int i = 0; i < std::min(5u, static_cast<unsigned int>(agx::config::SPECTRAL_SHAPE.wavelengths.size())); ++i) {
        std::cout << agx::config::SPECTRAL_SHAPE.wavelengths[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << std::endl;
    
    std::cout << "Last 5 wavelengths: ";
    for (unsigned int i = std::max(0u, static_cast<unsigned int>(agx::config::SPECTRAL_SHAPE.wavelengths.size()) - 5); i < agx::config::SPECTRAL_SHAPE.wavelengths.size(); ++i) {
        std::cout << agx::config::SPECTRAL_SHAPE.wavelengths[i];
        if (i < agx::config::SPECTRAL_SHAPE.wavelengths.size() - 1) std::cout << ", ";
    }
    std::cout << std::endl;
    
    // Test creating a simple array
    auto test_array = nc::ones<float>({agx::config::SPECTRAL_SHAPE.wavelengths.size()});
    std::cout << "Test array size: " << test_array.size() << std::endl;
    std::cout << "Test array shape: [" << test_array.shape().rows << ", " << test_array.shape().cols << "]" << std::endl;
    
    return 0;
} 