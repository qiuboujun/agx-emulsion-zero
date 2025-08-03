#include "color_filters.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

void test_sigmoid_erf() {
    std::cout << "=== Testing sigmoid_erf ===" << std::endl;
    
    // Test data - same as Python
    std::vector<float> x_data = {400.0f, 450.0f, 500.0f, 550.0f, 600.0f, 650.0f, 700.0f};
    nc::NdArray<float> x(x_data);
    float center = 550.0f;
    float width = 10.0f;
    
    // C++ implementation
    auto result = agx::model::sigmoid_erf(x, center, width);
    
    std::cout << "Input x: [";
    for (unsigned int i = 0; i < x.size(); ++i) {
        std::cout << x[i];
        if (i < x.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Center: " << center << ", Width: " << width << std::endl;
    
    std::cout << "C++ result: [";
    for (unsigned int i = 0; i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_create_combined_dichroic_filter() {
    std::cout << "=== Testing create_combined_dichroic_filter ===" << std::endl;
    
    // Test data - same as Python
    std::vector<float> wavelength_data = {400.0f, 450.0f, 500.0f, 550.0f, 600.0f, 650.0f, 700.0f};
    nc::NdArray<float> wavelength(wavelength_data);
    std::array<float, 3> filtering_amount_percent = {50.0f, 30.0f, 20.0f};
    std::array<float, 4> transitions = {10.0f, 10.0f, 10.0f, 10.0f};
    std::array<float, 4> edges = {510.0f, 495.0f, 605.0f, 590.0f};
    float nd_filter = 5.0f;
    
    // C++ implementation
    auto result = agx::model::create_combined_dichroic_filter(
        wavelength, filtering_amount_percent, transitions, edges, nd_filter
    );
    
    std::cout << "Wavelength: [";
    for (unsigned int i = 0; i < wavelength.size(); ++i) {
        std::cout << wavelength[i];
        if (i < wavelength.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Filtering amount: [" << filtering_amount_percent[0] << ", " 
              << filtering_amount_percent[1] << ", " << filtering_amount_percent[2] << "]" << std::endl;
    std::cout << "Transitions: [" << transitions[0] << ", " << transitions[1] << ", " 
              << transitions[2] << ", " << transitions[3] << "]" << std::endl;
    std::cout << "Edges: [" << edges[0] << ", " << edges[1] << ", " 
              << edges[2] << ", " << edges[3] << "]" << std::endl;
    std::cout << "ND filter: " << nd_filter << std::endl;
    std::cout << "C++ result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "C++ result: [";
    for (unsigned int i = 0; i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_filterset() {
    std::cout << "=== Testing filterset ===" << std::endl;
    
    // Test data - same as Python, using the correct spectral shape
    // Create a 1D array with the correct size
    auto wl_size = agx::config::SPECTRAL_SHAPE.wavelengths.size();
    nc::NdArray<float> illuminant = nc::ones<float>({wl_size}).flatten();
    std::array<float, 3> values = {25.0f, 15.0f, 10.0f};
    std::array<float, 4> edges = {510.0f, 495.0f, 605.0f, 590.0f};
    std::array<float, 4> transitions = {10.0f, 10.0f, 10.0f, 10.0f};
    
    // C++ implementation
    auto result = agx::model::filterset(illuminant, values, edges, transitions);
    
    std::cout << "Illuminant: [";
    for (unsigned int i = 0; i < illuminant.size(); ++i) {
        std::cout << illuminant[i];
        if (i < illuminant.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Values: [" << values[0] << ", " << values[1] << ", " << values[2] << "]" << std::endl;
    std::cout << "C++ result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "C++ result: [";
    for (unsigned int i = 0; i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_compute_band_pass_filter() {
    std::cout << "=== Testing compute_band_pass_filter ===" << std::endl;
    
    // Test data - same as Python
    std::array<float, 3> filter_uv = {0.8f, 410.0f, 8.0f};
    std::array<float, 3> filter_ir = {0.6f, 675.0f, 15.0f};
    
    // C++ implementation
    auto result = agx::model::compute_band_pass_filter(filter_uv, filter_ir);
    
    std::cout << "UV filter params: [" << filter_uv[0] << ", " << filter_uv[1] << ", " << filter_uv[2] << "]" << std::endl;
    std::cout << "IR filter params: [" << filter_ir[0] << ", " << filter_ir[1] << ", " << filter_ir[2] << "]" << std::endl;
    std::cout << "C++ result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "C++ result first 5: [";
    for (unsigned int i = 0; i < std::min(5u, static_cast<unsigned int>(result.size())); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "C++ result last 5: [";
    for (unsigned int i = std::max(0u, static_cast<unsigned int>(result.size()) - 5); i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_dichroic_filters() {
    std::cout << "=== Testing DichroicFilters ===" << std::endl;
    
    // Create filter instance
    agx::model::DichroicFilters filters("thorlabs");
    
    // Test data - same as Python
    auto wl_size = agx::config::SPECTRAL_SHAPE.wavelengths.size();
    nc::NdArray<float> illuminant = nc::ones<float>({wl_size}).flatten();
    std::array<float, 3> values = {0.5f, 0.3f, 0.2f};
    
    // C++ implementation
    auto result = filters.apply(illuminant, values);
    
    std::cout << "Brand: thorlabs" << std::endl;
    std::cout << "Illuminant shape: (" << illuminant.size() << ")" << std::endl;
    std::cout << "Values: [" << values[0] << ", " << values[1] << ", " << values[2] << "]" << std::endl;
    std::cout << "C++ result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "C++ result first 5: [";
    for (unsigned int i = 0; i < std::min(5u, static_cast<unsigned int>(result.size())); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "C++ result last 5: [";
    for (unsigned int i = std::max(0u, static_cast<unsigned int>(result.size()) - 5); i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_generic_filter() {
    std::cout << "=== Testing GenericFilter ===" << std::endl;
    
    // Create filter instance
    agx::model::GenericFilter filter_obj("KG3", "heat_absorbing", "schott");
    
    // Test data - same as Python
    auto wl_size = agx::config::SPECTRAL_SHAPE.wavelengths.size();
    nc::NdArray<float> illuminant = nc::ones<float>({wl_size}).flatten();
    float value = 0.7f;
    
    // C++ implementation
    auto result = filter_obj.apply(illuminant, value);
    
    std::cout << "Name: KG3, Type: heat_absorbing, Brand: schott" << std::endl;
    std::cout << "Illuminant shape: (" << illuminant.size() << ")" << std::endl;
    std::cout << "Value: " << value << std::endl;
    std::cout << "C++ result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "C++ result first 5: [";
    for (unsigned int i = 0; i < std::min(5u, static_cast<unsigned int>(result.size())); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "C++ result last 5: [";
    for (unsigned int i = std::max(0u, static_cast<unsigned int>(result.size()) - 5); i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_color_enlarger() {
    std::cout << "=== Testing color_enlarger ===" << std::endl;
    
    // Test data - same as Python
    auto wl_size = agx::config::SPECTRAL_SHAPE.wavelengths.size();
    nc::NdArray<float> light_source = nc::ones<float>({wl_size}).flatten();
    float y_filter_value = 85.0f;
    float m_filter_value = 45.0f;
    float c_filter_value = 25.0f;
    
    // C++ implementation
    auto result = agx::model::color_enlarger(light_source, y_filter_value, m_filter_value, c_filter_value);
    
    std::cout << "Light source shape: (" << light_source.size() << ")" << std::endl;
    std::cout << "Y filter: " << y_filter_value << ", M filter: " << m_filter_value 
              << ", C filter: " << c_filter_value << std::endl;
    std::cout << "C++ result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "C++ result first 5: [";
    for (unsigned int i = 0; i < std::min(5u, static_cast<unsigned int>(result.size())); ++i) {
        if (std::isnan(result[i])) {
            std::cout << "nan";
        } else {
            std::cout << std::scientific << std::setprecision(8) << result[i];
        }
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "C++ result last 5: [";
    for (unsigned int i = std::max(0u, static_cast<unsigned int>(result.size()) - 5); i < result.size(); ++i) {
        if (std::isnan(result[i])) {
            std::cout << "nan";
        } else {
            std::cout << std::scientific << std::setprecision(8) << result[i];
        }
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

int main() {
    std::cout << "Color Filters C++ Test" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Initialize configuration
    agx::config::initialize_config();
    
    test_sigmoid_erf();
    test_create_combined_dichroic_filter();
    test_filterset();
    test_compute_band_pass_filter();
    test_dichroic_filters();
    test_generic_filter();
    test_color_enlarger();
    
    std::cout << "All C++ tests completed!" << std::endl;
    return 0;
} 