#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include "../../src/model/couplers.cpp"  // Include the implementation directly

using namespace agx_emulsion;

// Helper function to print a 3x3 matrix
void print_matrix(const std::array<std::array<double, 3>, 3>& matrix, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << std::fixed << std::setprecision(10) << matrix[i][j];
            if (j < 2) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Helper function to print a 2D vector
void print_2d_vector(const std::vector<std::vector<double>>& vec, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = 0; j < vec[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(10) << vec[i][j];
            if (j < vec[i].size() - 1) std::cout << ", ";
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
}

// Helper function to print a 3D vector (H x W x 3)
void print_3d_vector(const std::vector<std::vector<std::array<double, 3>>>& vec, const std::string& name) {
    std::cout << name << ":" << std::endl;
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = 0; j < vec[i].size(); ++j) {
            std::cout << "[" << i << "," << j << "]: ";
            for (int k = 0; k < 3; ++k) {
                std::cout << std::fixed << std::setprecision(10) << vec[i][j][k];
                if (k < 2) std::cout << ", ";
            }
            std::cout << std::endl;
        }
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "=== C++ Couplers Test Results ===" << std::endl << std::endl;
    
    // Test 1: compute_dir_couplers_matrix
    std::cout << "Test 1: compute_dir_couplers_matrix" << std::endl;
    std::cout << "==================================" << std::endl;
    
    std::array<double, 3> amount_rgb = {0.7, 0.7, 0.5};
    double layer_diffusion = 1.0;
    
    std::cout << "Input amount_rgb: [" << amount_rgb[0] << ", " << amount_rgb[1] << ", " << amount_rgb[2] << "]" << std::endl;
    std::cout << "Input layer_diffusion: " << layer_diffusion << std::endl << std::endl;
    
    auto matrix = Couplers::compute_dir_couplers_matrix(amount_rgb, layer_diffusion);
    print_matrix(matrix, "Output matrix");
    
    // Test 2: compute_density_curves_before_dir_couplers
    std::cout << "Test 2: compute_density_curves_before_dir_couplers" << std::endl;
    std::cout << "=================================================" << std::endl;
    
    // Create fixed test data
    std::vector<double> log_exposure = {-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0};
    std::vector<std::vector<double>> density_curves = {
        {0.1, 0.2, 0.3},
        {0.2, 0.3, 0.4},
        {0.3, 0.4, 0.5},
        {0.4, 0.5, 0.6},
        {0.5, 0.6, 0.7},
        {0.6, 0.7, 0.8},
        {0.7, 0.8, 0.9}
    };
    
    std::cout << "Input log_exposure: [";
    for (size_t i = 0; i < log_exposure.size(); ++i) {
        std::cout << log_exposure[i];
        if (i < log_exposure.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl << std::endl;
    
    std::cout << "Input density_curves:" << std::endl;
    print_2d_vector(density_curves, "density_curves");
    
    auto corrected_curves = Couplers::compute_density_curves_before_dir_couplers(
        density_curves, log_exposure, matrix, 0.1);
    print_2d_vector(corrected_curves, "Output corrected_curves");
    
    // Test 3: compute_exposure_correction_dir_couplers
    std::cout << "Test 3: compute_exposure_correction_dir_couplers" << std::endl;
    std::cout << "================================================" << std::endl;
    
    // Create fixed 3D test data (2x2x3)
    std::vector<std::vector<std::array<double, 3>>> log_raw = {
        {{0.1, 0.2, 0.3}, {0.4, 0.5, 0.6}},
        {{0.7, 0.8, 0.9}, {1.0, 1.1, 1.2}}
    };
    
    std::vector<std::vector<std::array<double, 3>>> density_cmy = {
        {{0.2, 0.3, 0.4}, {0.5, 0.6, 0.7}},
        {{0.8, 0.9, 1.0}, {1.1, 1.2, 1.3}}
    };
    
    std::array<double, 3> density_max = {2.0, 2.2, 2.4};
    int diffusion_size_pixel = 1;
    
    std::cout << "Input log_raw:" << std::endl;
    print_3d_vector(log_raw, "log_raw");
    
    std::cout << "Input density_cmy:" << std::endl;
    print_3d_vector(density_cmy, "density_cmy");
    
    std::cout << "Input density_max: [" << density_max[0] << ", " << density_max[1] << ", " << density_max[2] << "]" << std::endl;
    std::cout << "Input diffusion_size_pixel: " << diffusion_size_pixel << std::endl << std::endl;
    
    auto corrected_exposure = Couplers::compute_exposure_correction_dir_couplers(
        log_raw, density_cmy, density_max, matrix, diffusion_size_pixel, 0.1);
    print_3d_vector(corrected_exposure, "Output corrected_exposure");
    
    return 0;
} 