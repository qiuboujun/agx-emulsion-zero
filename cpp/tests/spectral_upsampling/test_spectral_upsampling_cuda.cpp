#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include "../../src/utils/spectral_upsampling.cpp"  // Include the CPU implementation
#include "../../src/utils/spectral_upsampling.cu"   // Include the CUDA implementation

using namespace std;

// Helper function to print a vector of coordinate pairs
void print_coord_pairs(const vector<float>& coords, const string& name, int max_pairs = 5) {
    cout << name << ":" << endl;
    for (size_t i = 0; i < coords.size() && i < static_cast<size_t>(max_pairs * 2); i += 2) {
        cout << "  [" << i/2 << "]: [" << fixed << setprecision(10) << coords[i] 
             << ", " << coords[i+1] << "]" << endl;
    }
    if (coords.size() > static_cast<size_t>(max_pairs * 2)) {
        cout << "  ... (showing first " << max_pairs << " pairs of " << coords.size()/2 << " total pairs)" << endl;
    }
}

int main() {
    cout << "=== CUDA Spectral Upsampling Test Results ===" << endl << endl;
    
    // Test 1: tri2quad CUDA transformation
    cout << "Test 1: tri2quad CUDA transformation" << endl;
    cout << "=====================================" << endl;
    
    // Create input data for CUDA (flattened array of coordinate pairs)
    vector<float> tri_coords_flat = {
        0.0f, 0.0f,      // [0, 0]
        0.5f, 0.5f,      // [0.5, 0.5]
        1.0f, 0.0f,      // [1.0, 0.0]
        0.25f, 0.25f,    // [0.25, 0.25]
        0.75f, 0.25f     // [0.75, 0.25]
    };
    
    cout << "Input triangular coordinates (flattened):" << endl;
    print_coord_pairs(tri_coords_flat, "tri_coords_flat");
    
    vector<float> quad_coords_flat;
    try {
        tri2quad_cuda(quad_coords_flat, tri_coords_flat);
        cout << "CUDA tri2quad successful!" << endl;
        print_coord_pairs(quad_coords_flat, "Output square coordinates (CUDA)");
    } catch (const exception& e) {
        cout << "CUDA tri2quad failed: " << e.what() << endl;
        cout << "This is expected if CUDA is not available." << endl;
    }
    cout << endl;
    
    // Test 2: quad2tri CUDA transformation
    cout << "Test 2: quad2tri CUDA transformation" << endl;
    cout << "====================================" << endl;
    
    // Create input data for CUDA (flattened array of coordinate pairs)
    vector<float> quad_coords_input = {
        0.0f, 0.0f,      // [0, 0]
        0.5f, 0.5f,      // [0.5, 0.5]
        1.0f, 1.0f,      // [1.0, 1.0]
        0.25f, 0.25f,    // [0.25, 0.25]
        0.75f, 0.75f     // [0.75, 0.75]
    };
    
    cout << "Input square coordinates (flattened):" << endl;
    print_coord_pairs(quad_coords_input, "quad_coords_input");
    
    vector<float> tri_coords_output;
    try {
        quad2tri_cuda(tri_coords_output, quad_coords_input);
        cout << "CUDA quad2tri successful!" << endl;
        print_coord_pairs(tri_coords_output, "Output triangular coordinates (CUDA)");
    } catch (const exception& e) {
        cout << "CUDA quad2tri failed: " << e.what() << endl;
        cout << "This is expected if CUDA is not available." << endl;
    }
    cout << endl;
    
    // Test 3: Compare CPU vs CUDA results (if CUDA is available)
    cout << "Test 3: CPU vs CUDA comparison" << endl;
    cout << "==============================" << endl;
    
    if (quad_coords_flat.size() == tri_coords_flat.size()) {
        cout << "Comparing CPU and CUDA tri2quad results:" << endl;
        bool match = true;
        for (size_t i = 0; i < quad_coords_flat.size(); ++i) {
            float cpu_result = 0.0f;
            if (i % 2 == 0) {
                // x coordinate
                auto cpu_pair = SpectralUpsampling::tri2quad(tri_coords_flat[i], tri_coords_flat[i+1]);
                cpu_result = cpu_pair.first;
            } else {
                // y coordinate
                auto cpu_pair = SpectralUpsampling::tri2quad(tri_coords_flat[i-1], tri_coords_flat[i]);
                cpu_result = cpu_pair.second;
            }
            
            float cuda_result = quad_coords_flat[i];
            float diff = abs(cpu_result - cuda_result);
            if (diff > 1e-6) {
                cout << "  Mismatch at index " << i << ": CPU=" << cpu_result << ", CUDA=" << cuda_result << endl;
                match = false;
            }
        }
        if (match) {
            cout << "✓ CPU and CUDA tri2quad results match!" << endl;
        } else {
            cout << "✗ CPU and CUDA tri2quad results differ!" << endl;
        }
    } else {
        cout << "CUDA results not available for comparison." << endl;
    }
    cout << endl;
    
    // Test 4: Large batch test (if CUDA is available)
    cout << "Test 4: Large batch test" << endl;
    cout << "========================" << endl;
    
    // Create a larger batch of coordinates
    vector<float> large_batch;
    const int num_coords = 1000;
    large_batch.reserve(num_coords * 2);
    
    for (int i = 0; i < num_coords; ++i) {
        float tx = static_cast<float>(i) / static_cast<float>(num_coords - 1);
        float ty = static_cast<float>(i % 100) / 99.0f;
        large_batch.push_back(tx);
        large_batch.push_back(ty);
    }
    
    cout << "Created batch of " << num_coords << " coordinate pairs." << endl;
    
    vector<float> large_batch_output;
    try {
        tri2quad_cuda(large_batch_output, large_batch);
        cout << "CUDA large batch tri2quad successful!" << endl;
        cout << "Output size: " << large_batch_output.size() << " elements" << endl;
        
        // Verify a few sample results
        cout << "Sample results:" << endl;
        for (int i = 0; i < 5; ++i) {
            size_t idx = i * 2;
            auto cpu_pair = SpectralUpsampling::tri2quad(large_batch[idx], large_batch[idx+1]);
            float cuda_x = large_batch_output[idx];
            float cuda_y = large_batch_output[idx+1];
            cout << "  [" << i << "]: CPU=[" << cpu_pair.first << ", " << cpu_pair.second 
                 << "], CUDA=[" << cuda_x << ", " << cuda_y << "]" << endl;
        }
    } catch (const exception& e) {
        cout << "CUDA large batch test failed: " << e.what() << endl;
        cout << "This is expected if CUDA is not available." << endl;
    }
    
    return 0;
} 