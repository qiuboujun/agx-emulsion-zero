#include "density_curves.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <array>

using namespace agx_emulsion;

void print_vector(const std::vector<double>& vec, const std::string& name) {
    std::cout << name << ": [";
    for (size_t i = 0; i < vec.size(); ++i) {
        std::cout << std::fixed << std::setprecision(10) << vec[i];
        if (i < vec.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

void print_matrix(const Matrix& mat, const std::string& name) {
    std::cout << name << " (" << mat.rows << "x" << mat.cols << "):" << std::endl;
    for (size_t r = 0; r < mat.rows; ++r) {
        std::cout << "  Row " << r << ": [";
        for (size_t c = 0; c < mat.cols; ++c) {
            std::cout << std::fixed << std::setprecision(10) << mat(r, c);
            if (c < mat.cols - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

int main() {
    std::cout << "=== C++ Density Curves Test Results ===" << std::endl << std::endl;

    // Test 1: Fixed input log-exposure grid
    std::cout << "Test 1: density_curve_model_norm_cdfs" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    std::vector<double> loge;
    for (int i = 0; i <= 10; ++i) { // [-2.0, -1.5, ..., 3.0]
        loge.push_back(-2.0 + 0.5 * i);
    }
    
    print_vector(loge, "Input log_exposure");
    
    // Default parameters
    DensityParams p;
    std::cout << "Parameters: center=[" << p.center[0] << ", " << p.center[1] << ", " << p.center[2] << "]" << std::endl;
    std::cout << "           amplitude=[" << p.amplitude[0] << ", " << p.amplitude[1] << ", " << p.amplitude[2] << "]" << std::endl;
    std::cout << "           sigma=[" << p.sigma[0] << ", " << p.sigma[1] << ", " << p.sigma[2] << "]" << std::endl;
    
    // Test negative curve
    auto negative_curve = density_curve_model_norm_cdfs(loge, p, CurveType::Negative, 3);
    print_vector(negative_curve, "Negative curve output");
    
    // Test positive curve
    auto positive_curve = density_curve_model_norm_cdfs(loge, p, CurveType::Positive, 3);
    print_vector(positive_curve, "Positive curve output");
    
    std::cout << std::endl;

    // Test 2: distribution_model_norm_cdfs
    std::cout << "Test 2: distribution_model_norm_cdfs" << std::endl;
    std::cout << "====================================" << std::endl;
    
    auto distribution = distribution_model_norm_cdfs(loge, p, 3);
    print_matrix(distribution, "Distribution matrix");
    
    std::cout << std::endl;

    // Test 3: compute_density_curves (3-channel)
    std::cout << "Test 3: compute_density_curves (3-channel)" << std::endl;
    std::cout << "==========================================" << std::endl;
    
    std::array<DensityParams,3> params_rgb{p, p, p};
    auto density_curves = compute_density_curves(loge, params_rgb, CurveType::Negative);
    print_matrix(density_curves, "3-channel density curves");
    
    std::cout << std::endl;

    // Test 4: interpolate_exposure_to_density
    std::cout << "Test 4: interpolate_exposure_to_density" << std::endl;
    std::cout << "=======================================" << std::endl;
    
    // Create a simple test matrix (2x3)
    Matrix log_exposure_rgb(2, 3);
    log_exposure_rgb(0, 0) = -1.0; log_exposure_rgb(0, 1) = -0.5; log_exposure_rgb(0, 2) = 0.0;
    log_exposure_rgb(1, 0) = 1.0;  log_exposure_rgb(1, 1) = 1.5;  log_exposure_rgb(1, 2) = 2.0;
    
    std::array<double,3> gamma_factor{1.0, 1.0, 1.0};
    
    auto interpolated = interpolate_exposure_to_density(log_exposure_rgb, density_curves, loge, gamma_factor);
    print_matrix(log_exposure_rgb, "Input log_exposure_rgb");
    print_matrix(interpolated, "Interpolated density");
    
    std::cout << std::endl;

    // Test 5: apply_gamma_shift_correction
    std::cout << "Test 5: apply_gamma_shift_correction" << std::endl;
    std::cout << "====================================" << std::endl;
    
    std::array<double,3> gamma_correction{1.1, 0.9, 1.0};
    std::array<double,3> log_exp_correction{0.1, -0.1, 0.0};
    
    auto corrected = apply_gamma_shift_correction(loge, density_curves, gamma_correction, log_exp_correction);
    print_matrix(corrected, "Gamma-shift corrected density curves");
    
    std::cout << std::endl;

    // Test 6: GPU vs CPU comparison
    std::cout << "Test 6: GPU vs CPU comparison" << std::endl;
    std::cout << "=============================" << std::endl;
    
    std::vector<double> gpu_curve;
    bool ran_gpu = gpu_density_curve_model_norm_cdfs(loge, p, CurveType::Negative, 3, gpu_curve);
    
    std::cout << "Ran on: " << (ran_gpu ? "GPU" : "CPU fallback") << std::endl;
    print_vector(gpu_curve, "GPU/CPU curve output");
    
    // Compare with CPU
    double max_diff = 0.0;
    for (size_t i = 0; i < negative_curve.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(negative_curve[i] - gpu_curve[i]));
    }
    std::cout << "Max absolute difference (CPU vs GPU): " << std::fixed << std::setprecision(15) << max_diff << std::endl;
    
    std::cout << std::endl;
    std::cout << "=== Test completed ===" << std::endl;
    
    return 0;
} 