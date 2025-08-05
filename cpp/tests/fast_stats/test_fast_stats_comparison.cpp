#include "fast_stats.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

using namespace agx_emulsion;

void print_comparison_results() {
    std::cout << "=== FastStats C++/CUDA vs Python Comparison ===" << std::endl;
    std::cout << "=" << std::string(50, '=') << std::endl;
    
    // Test case 1: Known values (same as Python test)
    std::cout << "\n1. Test Case: Known Values" << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    std::vector<float> data1 = {1.5f, 2.3f, 3.7f, 4.2f, 5.8f};
    double expected_mean = 3.5;
    double expected_std = 1.5006665185843255;
    
    std::cout << "Input data: [";
    for (size_t i = 0; i < data1.size(); ++i) {
        std::cout << data1[i];
        if (i < data1.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    std::cout << "Expected mean: " << expected_mean << std::endl;
    std::cout << "Expected std: " << expected_std << std::endl;
    
    // C++ CPU computation
    auto [cpp_mean, cpp_std] = FastStats::mean_stddev(data1);
    std::cout << "\nC++ CPU results:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(15) << cpp_mean << std::endl;
    std::cout << "  Std:  " << std::fixed << std::setprecision(15) << cpp_std << std::endl;
    
    // C++ GPU computation
    try {
        auto [gpu_mean, gpu_std] = FastStats::compute_gpu(data1.data(), data1.size());
        std::cout << "\nC++ GPU results:" << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(15) << gpu_mean << std::endl;
        std::cout << "  Std:  " << std::fixed << std::setprecision(15) << gpu_std << std::endl;
        
        // Compare CPU vs GPU
        double cpu_gpu_mean_diff = std::abs(cpp_mean - gpu_mean);
        double cpu_gpu_std_diff = std::abs(cpp_std - gpu_std);
        std::cout << "\nCPU vs GPU differences:" << std::endl;
        std::cout << "  Mean diff: " << std::fixed << std::setprecision(15) << cpu_gpu_mean_diff << std::endl;
        std::cout << "  Std diff:  " << std::fixed << std::setprecision(15) << cpu_gpu_std_diff << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "\nGPU not available: " << e.what() << std::endl;
    }
    
    // Differences from expected
    double mean_diff = std::abs(cpp_mean - expected_mean);
    double std_diff = std::abs(cpp_std - expected_std);
    std::cout << "\nDifferences from expected:" << std::endl;
    std::cout << "  Mean diff: " << std::fixed << std::setprecision(15) << mean_diff << std::endl;
    std::cout << "  Std diff:  " << std::fixed << std::setprecision(15) << std_diff << std::endl;
    
    // Test case 2: Large dataset (same as Python test)
    std::cout << "\n\n2. Test Case: Large Dataset (Normal Distribution)" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // Generate the same large dataset as Python (using same seed)
    std::vector<float> large_data(10000);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(10.0f, 2.0f);
    
    for (size_t i = 0; i < large_data.size(); ++i) {
        large_data[i] = dist(gen);
    }
    
    std::cout << "Dataset size: " << large_data.size() << std::endl;
    std::cout << "Expected mean: ~10.0" << std::endl;
    std::cout << "Expected std:  ~2.0" << std::endl;
    
    // C++ CPU computation
    auto [cpp_large_mean, cpp_large_std] = FastStats::mean_stddev(large_data);
    std::cout << "\nC++ CPU results:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(15) << cpp_large_mean << std::endl;
    std::cout << "  Std:  " << std::fixed << std::setprecision(15) << cpp_large_std << std::endl;
    
    // C++ GPU computation
    try {
        auto [gpu_large_mean, gpu_large_std] = FastStats::compute_gpu(large_data.data(), large_data.size());
        std::cout << "\nC++ GPU results:" << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(15) << gpu_large_mean << std::endl;
        std::cout << "  Std:  " << std::fixed << std::setprecision(15) << gpu_large_std << std::endl;
        
        // Compare CPU vs GPU
        double cpu_gpu_large_mean_diff = std::abs(cpp_large_mean - gpu_large_mean);
        double cpu_gpu_large_std_diff = std::abs(cpp_large_std - gpu_large_std);
        std::cout << "\nCPU vs GPU differences:" << std::endl;
        std::cout << "  Mean diff: " << std::fixed << std::setprecision(15) << cpu_gpu_large_mean_diff << std::endl;
        std::cout << "  Std diff:  " << std::fixed << std::setprecision(15) << cpu_gpu_large_std_diff << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "\nGPU not available: " << e.what() << std::endl;
    }
    
    // Test case 3: Edge cases
    std::cout << "\n\n3. Test Case: Edge Cases" << std::endl;
    std::cout << std::string(25, '-') << std::endl;
    
    // Empty array
    std::vector<float> empty_data;
    auto [cpp_empty_mean, cpp_empty_std] = FastStats::mean_stddev(empty_data);
    std::cout << "Empty array:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(15) << cpp_empty_mean << std::endl;
    std::cout << "  Std:  " << std::fixed << std::setprecision(15) << cpp_empty_std << std::endl;
    
    // Single element
    std::vector<float> single_data = {42.0f};
    auto [cpp_single_mean, cpp_single_std] = FastStats::mean_stddev(single_data);
    std::cout << "Single element [42.0]:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(15) << cpp_single_mean << std::endl;
    std::cout << "  Std:  " << std::fixed << std::setprecision(15) << cpp_single_std << std::endl;
    
    // Test case 4: Fixed pattern (same as diffusion test)
    std::cout << "\n\n4. Test Case: Fixed Pattern (Sine/Cosine)" << std::endl;
    std::cout << std::string(40, '-') << std::endl;
    
    // Create the same fixed pattern as used in diffusion tests
    std::vector<float> fixed_data(100);
    for (int i = 0; i < 100; ++i) {
        float x = i / 99.0f;
        float val = 0.5f + 0.3f * std::sin(2.0f * M_PI * x) + 0.2f * std::cos(4.0f * M_PI * x);
        fixed_data[i] = val;
    }
    
    std::cout << "Fixed pattern size: " << fixed_data.size() << std::endl;
    std::cout << "First 10 values: [";
    for (int i = 0; i < 10; ++i) {
        std::cout << std::fixed << std::setprecision(6) << fixed_data[i];
        if (i < 9) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    // C++ CPU computation
    auto [cpp_fixed_mean, cpp_fixed_std] = FastStats::mean_stddev(fixed_data);
    std::cout << "\nC++ CPU results:" << std::endl;
    std::cout << "  Mean: " << std::fixed << std::setprecision(15) << cpp_fixed_mean << std::endl;
    std::cout << "  Std:  " << std::fixed << std::setprecision(15) << cpp_fixed_std << std::endl;
    
    // C++ GPU computation
    try {
        auto [gpu_fixed_mean, gpu_fixed_std] = FastStats::compute_gpu(fixed_data.data(), fixed_data.size());
        std::cout << "\nC++ GPU results:" << std::endl;
        std::cout << "  Mean: " << std::fixed << std::setprecision(15) << gpu_fixed_mean << std::endl;
        std::cout << "  Std:  " << std::fixed << std::setprecision(15) << gpu_fixed_std << std::endl;
        
        // Compare CPU vs GPU
        double cpu_gpu_fixed_mean_diff = std::abs(cpp_fixed_mean - gpu_fixed_mean);
        double cpu_gpu_fixed_std_diff = std::abs(cpp_fixed_std - gpu_fixed_std);
        std::cout << "\nCPU vs GPU differences:" << std::endl;
        std::cout << "  Mean diff: " << std::fixed << std::setprecision(15) << cpu_gpu_fixed_mean_diff << std::endl;
        std::cout << "  Std diff:  " << std::fixed << std::setprecision(15) << cpu_gpu_fixed_std_diff << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "\nGPU not available: " << e.what() << std::endl;
    }
    
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << "C++/CUDA computation complete!" << std::endl;
    std::cout << "Compare with Python results above." << std::endl;
}

int main() {
    print_comparison_results();
    return 0;
} 