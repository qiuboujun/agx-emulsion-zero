#pragma once

#include "NumCpp.hpp"
#include <string>
#include <fstream>
#include <sstream>

namespace colour {

/**
 * @brief A C++ struct to replicate colour.SpectralShape.
 * It defines the range and interval of wavelengths for spectral data.
 */
struct SpectralShape {
    float start;
    float end;
    float interval;
    nc::NdArray<float> wavelengths;

    SpectralShape(float start_wl, float end_wl, float interval_wl)
        : start(start_wl), end(end_wl), interval(interval_wl) {
        // The + interval/2 is a small tolerance to ensure the endpoint is included,
        // matching the behavior of np.arange and colour.SpectralShape.
        wavelengths = nc::arange(start, end + interval / 2.0f, interval);
    }
};

/**
 * @brief Loads the CIE 1931 2-degree standard observer color matching functions.
 * In Python, this is colour.MSDS_CMFS["CIE 1931 2 Degree Standard Observer"].
 * This function reads the data from a bundled CSV file.
 * @return An nc::NdArray of shape [N, 4] with columns [wavelength, x, y, z].
 */
inline nc::NdArray<float> get_cie_1931_2_degree_cmfs() {
    std::string filename = "./cpp/data/CIE_1931_2_Degree_CMFS.csv";
    std::ifstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    
    std::vector<std::vector<float>> data;
    std::string line;
    
    // Skip header if it exists and read data
    while (std::getline(file, line)) {
        std::istringstream iss(line);
        std::string token;
        std::vector<float> row;
        
        while (std::getline(iss, token, ',')) {
            row.push_back(std::stof(token));
        }
        
        if (row.size() == 4) { // wavelength, x, y, z
            data.push_back(row);
        }
    }
    
    file.close();
    
    if (data.empty()) {
        throw std::runtime_error("No valid data found in file: " + filename);
    }
    
    // Convert to NumCpp array
    nc::NdArray<float> result(data.size(), 4);
    for (size_t i = 0; i < data.size(); ++i) {
        for (size_t j = 0; j < 4; ++j) {
            result(i, j) = data[i][j];
        }
    }
    
    return result;
}

/**
 * @brief Resamples a spectral distribution to a new spectral shape.
 * This is the C++ equivalent of the .align() method.
 * @param spectral_data The input spectral data, with wavelengths in the first column.
 * @param shape The target SpectralShape to align to.
 * @return A new nc::NdArray containing the resampled data, aligned to the new wavelengths.
 */
inline nc::NdArray<float> align(const nc::NdArray<float>& spectral_data, const SpectralShape& shape) {
    // For now, implement a simple linear interpolation
    // This is a basic implementation - in practice, you might want more sophisticated interpolation
    
    const nc::NdArray<float>& target_wavelengths = shape.wavelengths;
    
    // Extract source wavelengths from the first column
    std::vector<float> source_wavelengths;
    for (size_t i = 0; i < spectral_data.shape().rows; ++i) {
        source_wavelengths.push_back(spectral_data(i, 0));
    }
    
    nc::NdArray<float> result(target_wavelengths.size(), spectral_data.shape().cols);
    
    for (size_t i = 0; i < target_wavelengths.size(); ++i) {
        float target_wl = target_wavelengths[i];
        
        // Find the two closest source wavelengths
        size_t lower_idx = 0;
        size_t upper_idx = source_wavelengths.size() - 1;
        bool found_range = false;
        
        for (size_t j = 0; j < source_wavelengths.size() - 1; ++j) {
            if (source_wavelengths[j] <= target_wl && source_wavelengths[j + 1] >= target_wl) {
                lower_idx = j;
                upper_idx = j + 1;
                found_range = true;
                break;
            }
        }
        
        // If target wavelength is outside source range, use nearest neighbor
        if (!found_range) {
            if (target_wl <= source_wavelengths[0]) {
                lower_idx = upper_idx = 0;
            } else {
                lower_idx = upper_idx = source_wavelengths.size() - 1;
            }
        }
        
        // Linear interpolation or nearest neighbor
        if (lower_idx == upper_idx) {
            // Nearest neighbor
            for (size_t col = 0; col < spectral_data.shape().cols; ++col) {
                result(i, col) = spectral_data(lower_idx, col);
            }
        } else {
            // Linear interpolation
            float wl_lower = source_wavelengths[lower_idx];
            float wl_upper = source_wavelengths[upper_idx];
            float alpha = (target_wl - wl_lower) / (wl_upper - wl_lower);
            
            // Interpolate all columns
            for (size_t col = 0; col < spectral_data.shape().cols; ++col) {
                float val_lower = spectral_data(lower_idx, col);
                float val_upper = spectral_data(upper_idx, col);
                
                // Handle zero values properly - preserve exact zeros
                if (std::abs(val_lower) < 1e-10 && std::abs(val_upper) < 1e-10) {
                    result(i, col) = 0.0f;
                } else {
                    result(i, col) = val_lower + alpha * (val_upper - val_lower);
                }
            }
        }
    }
    
    return result;
}

} // namespace colour
