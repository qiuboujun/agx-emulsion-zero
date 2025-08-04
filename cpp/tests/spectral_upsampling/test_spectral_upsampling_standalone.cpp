#include <iostream>
#include <iomanip>
#include <vector>
#include <array>
#include "../../src/utils/spectral_upsampling.cpp"  // Include the implementation directly

using namespace std;

// Helper function to print a pair of coordinates
void print_coords(const pair<float, float>& coords, const string& name) {
    cout << name << ": [" << fixed << setprecision(10) << coords.first 
         << ", " << coords.second << "]" << endl;
}

// Helper function to print a vector of floats
void print_vector(const vector<float>& vec, const string& name, int max_elements = 10) {
    cout << name << ": [";
    for (size_t i = 0; i < vec.size() && i < static_cast<size_t>(max_elements); ++i) {
        cout << fixed << setprecision(10) << vec[i];
        if (i < vec.size() - 1 && i < static_cast<size_t>(max_elements) - 1) {
            cout << ", ";
        }
    }
    if (vec.size() > static_cast<size_t>(max_elements)) {
        cout << ", ... (showing first " << max_elements << " of " << vec.size() << " elements)";
    }
    cout << "]" << endl;
}

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
    cout << "=== C++ Spectral Upsampling Test Results ===" << endl << endl;
    
    // Test 1: tri2quad coordinate transformation
    cout << "Test 1: tri2quad coordinate transformation" << endl;
    cout << "===========================================" << endl;
    
    // Test cases for triangular to square coordinates
    vector<pair<float, float>> tri_coords = {
        {0.0f, 0.0f},      // Corner of triangle
        {0.5f, 0.5f},      // Middle of triangle
        {1.0f, 0.0f},      // Another corner
        {0.25f, 0.25f},    // Quarter point
        {0.75f, 0.25f}     // Three-quarter point
    };
    
    for (size_t i = 0; i < tri_coords.size(); ++i) {
        auto& tri = tri_coords[i];
        cout << "Input triangular coordinates [" << i << "]: [" << tri.first << ", " << tri.second << "]" << endl;
        
        auto quad = SpectralUpsampling::tri2quad(tri.first, tri.second);
        print_coords(quad, "Output square coordinates");
        cout << endl;
    }
    
    // Test 2: quad2tri coordinate transformation
    cout << "Test 2: quad2tri coordinate transformation" << endl;
    cout << "===========================================" << endl;
    
    // Test cases for square to triangular coordinates
    vector<pair<float, float>> quad_coords = {
        {0.0f, 0.0f},      // Corner of square
        {0.5f, 0.5f},      // Center of square
        {1.0f, 1.0f},      // Opposite corner
        {0.25f, 0.25f},    // Quarter point
        {0.75f, 0.75f}     // Three-quarter point
    };
    
    for (size_t i = 0; i < quad_coords.size(); ++i) {
        auto& quad = quad_coords[i];
        cout << "Input square coordinates [" << i << "]: [" << quad.first << ", " << quad.second << "]" << endl;
        
        auto tri = SpectralUpsampling::quad2tri(quad.first, quad.second);
        print_coords(tri, "Output triangular coordinates");
        cout << endl;
    }
    
    // Test 3: Round-trip transformation (tri2quad then quad2tri)
    cout << "Test 3: Round-trip transformation (tri2quad -> quad2tri)" << endl;
    cout << "========================================================" << endl;
    
    for (size_t i = 0; i < tri_coords.size(); ++i) {
        auto& original_tri = tri_coords[i];
        cout << "Original triangular coordinates [" << i << "]: [" << original_tri.first << ", " << original_tri.second << "]" << endl;
        
        auto quad = SpectralUpsampling::tri2quad(original_tri.first, original_tri.second);
        print_coords(quad, "After tri2quad");
        
        auto recovered_tri = SpectralUpsampling::quad2tri(quad.first, quad.second);
        print_coords(recovered_tri, "After quad2tri (recovered)");
        
        float error = sqrt(pow(original_tri.first - recovered_tri.first, 2) + 
                          pow(original_tri.second - recovered_tri.second, 2));
        cout << "Round-trip error: " << fixed << setprecision(10) << error << endl;
        cout << endl;
    }
    
    // Test 4: computeSpectraFromCoeffs
    cout << "Test 4: computeSpectraFromCoeffs" << endl;
    cout << "================================" << endl;
    
    // Test coefficient sets
    vector<array<float, 4>> coeff_sets = {
        {1.0f, 0.0f, 0.0f, 1.0f},      // Simple case
        {0.5f, 0.1f, 0.2f, 1.0f},      // More complex case
        {0.0f, 1.0f, 0.5f, 2.0f},      // Another case
        {0.1f, 0.2f, 0.3f, 0.5f}      // Small coefficients
    };
    
    for (size_t i = 0; i < coeff_sets.size(); ++i) {
        auto& coeffs = coeff_sets[i];
        cout << "Input coefficients [" << i << "]: [" << coeffs[0] << ", " << coeffs[1] 
             << ", " << coeffs[2] << ", " << coeffs[3] << "]" << endl;
        
        auto spectra = SpectralUpsampling::computeSpectraFromCoeffs(coeffs);
        print_vector(spectra, "Output spectrum", 10);
        cout << "Spectrum length: " << spectra.size() << " samples" << endl;
        cout << endl;
    }
    
    // Test 5: Edge cases
    cout << "Test 5: Edge cases" << endl;
    cout << "==================" << endl;
    
    // Test edge cases for tri2quad
    vector<pair<float, float>> edge_tri_coords = {
        {0.999f, 0.001f},  // Near boundary
        {0.001f, 0.999f},  // Near boundary
        {0.5f, 0.0f},      // On edge
        {0.0f, 0.5f}       // On edge
    };
    
    for (size_t i = 0; i < edge_tri_coords.size(); ++i) {
        auto& tri = edge_tri_coords[i];
        cout << "Edge case triangular coordinates [" << i << "]: [" << tri.first << ", " << tri.second << "]" << endl;
        
        auto quad = SpectralUpsampling::tri2quad(tri.first, tri.second);
        print_coords(quad, "Output square coordinates");
        cout << endl;
    }
    
    // Test edge case for computeSpectraFromCoeffs (zero coefficient)
    array<float, 4> zero_coeffs = {0.0f, 0.0f, 0.0f, 0.0f};
    cout << "Edge case coefficients (all zero): [0.0, 0.0, 0.0, 0.0]" << endl;
    auto zero_spectra = SpectralUpsampling::computeSpectraFromCoeffs(zero_coeffs);
    print_vector(zero_spectra, "Output spectrum (zero coeffs)", 5);
    cout << endl;
    
    return 0;
} 