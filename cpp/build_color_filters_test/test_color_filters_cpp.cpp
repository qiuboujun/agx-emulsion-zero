#include "color_filters.hpp"
#include "config.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>

using namespace agx;

void test_sigmoid_erf() {
    std::cout << "=== Testing sigmoid_erf ===" << std::endl;
    
    // Create test data matching Python test
    std::vector<float> x_data = {400.0f, 450.0f, 500.0f, 550.0f, 600.0f, 650.0f, 700.0f};
    nc::NdArray<float> x(x_data);
    float center = 550.0f;
    float width = 10.0f;
    
    auto result = model::sigmoid_erf(x, center, width);
    
    std::cout << "Input x: [";
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << x[i];
        if (i < x.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Center: " << center << ", Width: " << width << std::endl;
    std::cout << "Result: [";
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_create_combined_dichroic_filter() {
    std::cout << "=== Testing create_combined_dichroic_filter ===" << std::endl;
    
    auto wavelength = config::SPECTRAL_SHAPE.wavelengths;
    std::vector<float> filtering_amount_percent = {50.0f, 30.0f, 20.0f};
    std::vector<float> transitions = {10.0f, 10.0f, 10.0f, 10.0f};
    std::vector<float> edges = {510.0f, 495.0f, 605.0f, 590.0f};
    float nd_filter = 5.0f;
    
    auto result = model::create_combined_dichroic_filter(
        wavelength, filtering_amount_percent, transitions, edges, nd_filter);
    
    std::cout << "Wavelength shape: (" << wavelength.size() << ")" << std::endl;
    std::cout << "Filtering amount: [50.0, 30.0, 20.0]" << std::endl;
    std::cout << "Transitions: [10.0, 10.0, 10.0, 10.0]" << std::endl;
    std::cout << "Edges: [510.0, 495.0, 605.0, 590.0]" << std::endl;
    std::cout << "ND filter: " << nd_filter << std::endl;
    std::cout << "Result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "Result first 5 values: [";
    for (size_t i = 0; i < std::min(size_t(5), result.size()); ++i) {
        std::cout << std::fixed << std::setprecision(3) << result[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Result last 5 values: [";
    for (size_t i = std::max(size_t(0), result.size() - 5); i < result.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_filterset() {
    std::cout << "=== Testing filterset ===" << std::endl;
    
    // Create a simple illuminant (flat spectrum)
    auto illuminant = nc::ones<float>(config::SPECTRAL_SHAPE.wavelengths.shape());
    std::vector<float> values = {25.0f, 15.0f, 10.0f};
    std::vector<float> edges = {510.0f, 495.0f, 605.0f, 590.0f};
    std::vector<float> transitions = {10.0f, 10.0f, 10.0f, 10.0f};
    
    auto result = model::filterset(illuminant, values, edges, transitions);
    
    std::cout << "Illuminant shape: (" << illuminant.size() << ")" << std::endl;
    std::cout << "Values: [25.0, 15.0, 10.0]" << std::endl;
    std::cout << "Result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "Result first 5 values: [";
    for (size_t i = 0; i < std::min(size_t(5), result.size()); ++i) {
        std::cout << std::fixed << std::setprecision(1) << result[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Result last 5 values: [";
    for (size_t i = std::max(size_t(0), result.size() - 5); i < result.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_compute_band_pass_filter() {
    std::cout << "=== Testing compute_band_pass_filter ===" << std::endl;
    
    std::vector<float> filter_uv = {0.8f, 410.0f, 8.0f};
    std::vector<float> filter_ir = {0.6f, 675.0f, 15.0f};
    
    auto result = model::compute_band_pass_filter(filter_uv, filter_ir);
    
    std::cout << "UV filter params: [0.8, 410.0, 8.0]" << std::endl;
    std::cout << "IR filter params: [0.6, 675.0, 15.0]" << std::endl;
    std::cout << "Result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "Result first 5 values: [";
    for (size_t i = 0; i < std::min(size_t(5), result.size()); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Result last 5 values: [";
    for (size_t i = std::max(size_t(0), result.size() - 5); i < result.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_dichroic_filters() {
    std::cout << "=== Testing DichroicFilters ===" << std::endl;
    
    model::DichroicFilters filters("thorlabs");
    
    // Create a simple illuminant
    auto illuminant = nc::ones<float>(config::SPECTRAL_SHAPE.wavelengths.shape());
    std::vector<float> values = {0.5f, 0.3f, 0.2f};
    
    auto result = filters.apply(illuminant, values);
    
    std::cout << "Brand: thorlabs" << std::endl;
    std::cout << "Wavelengths shape: (" << filters.get_wavelengths().size() << ")" << std::endl;
    std::cout << "Filters shape: (" << filters.get_filters().shape().rows << ", " 
              << filters.get_filters().shape().cols << ")" << std::endl;
    std::cout << "Values: [0.5, 0.3, 0.2]" << std::endl;
    std::cout << "Result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "Result first 5 values: [";
    for (size_t i = 0; i < std::min(size_t(5), result.size()); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Result last 5 values: [";
    for (size_t i = std::max(size_t(0), result.size() - 5); i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_generic_filter() {
    std::cout << "=== Testing GenericFilter ===" << std::endl;
    
    model::GenericFilter filter_obj("KG3", "heat_absorbing", "schott");
    
    // Create a simple illuminant
    auto illuminant = nc::ones<float>(config::SPECTRAL_SHAPE.wavelengths.shape());
    float value = 0.7f;
    
    auto result = filter_obj.apply(illuminant, value);
    
    std::cout << "Name: KG3, Type: heat_absorbing, Brand: schott" << std::endl;
    std::cout << "Wavelengths shape: (" << filter_obj.get_wavelengths().size() << ")" << std::endl;
    std::cout << "Transmittance shape: (" << filter_obj.get_transmittance().size() << ")" << std::endl;
    std::cout << "Value: " << value << std::endl;
    std::cout << "Result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "Result first 5 values: [";
    for (size_t i = 0; i < std::min(size_t(5), result.size()); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Result last 5 values: [";
    for (size_t i = std::max(size_t(0), result.size() - 5); i < result.size(); ++i) {
        std::cout << std::scientific << std::setprecision(8) << result[i];
        if (i < result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
}

void test_color_enlarger() {
    std::cout << "=== Testing color_enlarger ===" << std::endl;
    
    // Create a simple light source
    auto light_source = nc::ones<float>(config::SPECTRAL_SHAPE.wavelengths.shape());
    float y_filter_value = 85.0f;
    float m_filter_value = 45.0f;
    float c_filter_value = 25.0f;
    
    auto result = model::color_enlarger(light_source, y_filter_value, m_filter_value, c_filter_value);
    
    std::cout << "Light source shape: (" << light_source.size() << ")" << std::endl;
    std::cout << "Y filter: " << y_filter_value << ", M filter: " << m_filter_value 
              << ", C filter: " << c_filter_value << std::endl;
    std::cout << "Enlarger steps: " << config::ENLARGER_STEPS << std::endl;
    std::cout << "Result shape: (" << result.size() << ")" << std::endl;
    
    std::cout << "Result first 5 values: [";
    for (size_t i = 0; i < std::min(size_t(5), result.size()); ++i) {
        if (std::isnan(result[i])) {
            std::cout << "nan";
        } else {
            std::cout << std::scientific << std::setprecision(8) << result[i];
        }
        if (i < 4) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "Result last 5 values: [";
    for (size_t i = std::max(size_t(0), result.size() - 5); i < result.size(); ++i) {
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
    config::initialize_config();
    
    test_sigmoid_erf();
    test_create_combined_dichroic_filter();
    test_filterset();
    test_compute_band_pass_filter();
    test_dichroic_filters();
    test_generic_filter();
    test_color_enlarger();
    
    std::cout << "All tests completed!" << std::endl;
    return 0;
} 