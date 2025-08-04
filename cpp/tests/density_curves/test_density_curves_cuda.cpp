#include "density_curves.hpp"
#include <iostream>
#include <iomanip>

using namespace agx_emulsion;

int main() {
    std::cout << "=== CUDA Density Curves Test ===" << std::endl;
    
    // Simple test data
    std::vector<double> loge = {-1.0, 0.0, 1.0, 2.0};
    DensityParams p;
    
    std::cout << "Input log_exposure: [";
    for (size_t i = 0; i < loge.size(); ++i) {
        std::cout << std::fixed << std::setprecision(1) << loge[i];
        if (i < loge.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Test GPU implementation
    std::vector<double> gpu_result;
    bool ran_gpu = gpu_density_curve_model_norm_cdfs(loge, p, CurveType::Negative, 3, gpu_result);
    
    std::cout << "Ran on: " << (ran_gpu ? "GPU" : "CPU fallback") << std::endl;
    std::cout << "Result: [";
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        std::cout << std::fixed << std::setprecision(10) << gpu_result[i];
        if (i < gpu_result.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // Test CPU implementation for comparison
    std::vector<double> cpu_result = density_curve_model_norm_cdfs(loge, p, CurveType::Negative, 3);
    
    // Compare results
    double max_diff = 0.0;
    for (size_t i = 0; i < gpu_result.size(); ++i) {
        max_diff = std::max(max_diff, std::abs(gpu_result[i] - cpu_result[i]));
    }
    
    std::cout << "Max difference (GPU vs CPU): " << std::fixed << std::setprecision(15) << max_diff << std::endl;
    
    if (max_diff < 1e-10) {
        std::cout << "✓ SUCCESS: GPU and CPU results match!" << std::endl;
        return 0;
    } else {
        std::cout << "❌ FAILURE: GPU and CPU results differ!" << std::endl;
        return 1;
    }
} 