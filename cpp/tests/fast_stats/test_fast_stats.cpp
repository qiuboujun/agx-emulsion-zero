#include "fast_stats.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <vector>
#include <random>

using namespace agx_emulsion;

void test_cpu_functions() {
    std::cout << "Testing CPU functions..." << std::endl;
    
    // Test with known values
    std::vector<float> data = {1.5f, 2.3f, 3.7f, 4.2f, 5.8f};
    double expected_mean = 3.5;
    double expected_std = 1.5006665185843255;

    // Test individual functions
    double cpu_mean = FastStats::mean(data);
    double cpu_std = FastStats::stddev(data);
    
    std::cout << "  Individual functions:" << std::endl;
    std::cout << "    CPU Mean: " << cpu_mean << " (expected: " << expected_mean << ")" << std::endl;
    std::cout << "    CPU Std: " << cpu_std << " (expected: " << expected_std << ")" << std::endl;
    
    assert(std::abs(cpu_mean - expected_mean) < 1e-5);
    assert(std::abs(cpu_std - expected_std) < 1e-5);

    // Test combined function
    auto [combined_mean, combined_std] = FastStats::mean_stddev(data);
    
    std::cout << "  Combined function:" << std::endl;
    std::cout << "    CPU Mean: " << combined_mean << " (expected: " << expected_mean << ")" << std::endl;
    std::cout << "    CPU Std: " << combined_std << " (expected: " << expected_std << ")" << std::endl;
    
    assert(std::abs(combined_mean - expected_mean) < 1e-5);
    assert(std::abs(combined_std - expected_std) < 1e-5);

    // Test empty vector
    std::vector<float> empty_data;
    auto [empty_mean, empty_std] = FastStats::mean_stddev(empty_data);
    assert(empty_mean == 0.0);
    assert(empty_std == 0.0);
    
    std::cout << "  Empty vector test passed" << std::endl;
    
    // Test single element
    std::vector<float> single_data = {42.0f};
    auto [single_mean, single_std] = FastStats::mean_stddev(single_data);
    assert(single_mean == 42.0);
    assert(single_std == 0.0);
    
    std::cout << "  Single element test passed" << std::endl;
    
    std::cout << "CPU tests passed!" << std::endl;
}

void test_gpu_functions() {
    std::cout << "Testing GPU functions..." << std::endl;
    
    // Test with known values
    std::vector<float> data = {1.5f, 2.3f, 3.7f, 4.2f, 5.8f};
    double expected_mean = 3.5;
    double expected_std = 1.5006665185843255;

    try {
        auto [gpu_mean, gpu_std] = FastStats::compute_gpu(data.data(), data.size());
        
        std::cout << "  GPU Mean: " << gpu_mean << " (expected: " << expected_mean << ")" << std::endl;
        std::cout << "  GPU Std: " << gpu_std << " (expected: " << expected_std << ")" << std::endl;
        
        assert(std::abs(gpu_mean - expected_mean) < 1e-3);
        assert(std::abs(gpu_std - expected_std) < 1e-3);
        
        std::cout << "GPU tests passed!" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "GPU not available, using CPU fallback: " << e.what() << std::endl;
        
        // Test fallback behavior
        auto [fallback_mean, fallback_std] = FastStats::compute_gpu(data.data(), data.size());
        assert(std::abs(fallback_mean - expected_mean) < 1e-5);
        assert(std::abs(fallback_std - expected_std) < 1e-5);
        
        std::cout << "CPU fallback tests passed!" << std::endl;
    }
}

void test_large_dataset() {
    std::cout << "Testing large dataset..." << std::endl;
    
    // Generate a large dataset with known statistics
    std::vector<float> large_data(10000);
    std::mt19937 gen(42);  // Fixed seed for reproducibility
    std::normal_distribution<float> dist(10.0f, 2.0f);
    
    for (size_t i = 0; i < large_data.size(); ++i) {
        large_data[i] = dist(gen);
    }
    
    // Compute expected values (approximate for normal distribution)
    double expected_mean = 10.0;
    double expected_std = 2.0;
    
    // Test CPU
    auto [cpu_mean, cpu_std] = FastStats::mean_stddev(large_data);
    
    std::cout << "  Large dataset CPU:" << std::endl;
    std::cout << "    Mean: " << cpu_mean << " (expected: ~" << expected_mean << ")" << std::endl;
    std::cout << "    Std: " << cpu_std << " (expected: ~" << expected_std << ")" << std::endl;
    
    // Allow some tolerance for random sampling
    assert(std::abs(cpu_mean - expected_mean) < 0.1);
    assert(std::abs(cpu_std - expected_std) < 0.1);
    
    // Test GPU
    try {
        auto [gpu_mean, gpu_std] = FastStats::compute_gpu(large_data.data(), large_data.size());
        
        std::cout << "  Large dataset GPU:" << std::endl;
        std::cout << "    Mean: " << gpu_mean << " (expected: ~" << expected_mean << ")" << std::endl;
        std::cout << "    Std: " << gpu_std << " (expected: ~" << expected_std << ")" << std::endl;
        
        // Compare GPU with CPU results
        assert(std::abs(gpu_mean - cpu_mean) < 1e-3);
        assert(std::abs(gpu_std - cpu_std) < 1e-3);
        
        std::cout << "Large dataset GPU tests passed!" << std::endl;
    } catch (const std::runtime_error& e) {
        std::cout << "GPU not available for large dataset test" << std::endl;
    }
    
    std::cout << "Large dataset tests passed!" << std::endl;
}

int main() {
    std::cout << "=== FastStats Test Suite ===" << std::endl << std::endl;
    
    test_cpu_functions();
    std::cout << std::endl;
    
    test_gpu_functions();
    std::cout << std::endl;
    
    test_large_dataset();
    std::cout << std::endl;
    
    std::cout << "🎉 All FastStats tests passed!" << std::endl;
    return 0;
} 