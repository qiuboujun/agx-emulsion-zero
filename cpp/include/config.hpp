#pragma once

#include "NumCpp.hpp"
#include "colour.hpp"

namespace agx {
namespace config {

// Constants
constexpr int ENLARGER_STEPS = 170;

// Define global variables directly in the header using 'inline'.
// This prevents "multiple definition" linker errors.
inline nc::NdArray<float> LOG_EXPOSURE;
inline colour::SpectralShape SPECTRAL_SHAPE(380, 780, 5); // Can be initialized directly
inline nc::NdArray<float> STANDARD_OBSERVER_CMFS;

/**
 * @brief Initializes all global configuration data.
 * This function should be called once at the start of the application.
 */
inline void initialize_config() {
    // Initialize LOG_EXPOSURE
    LOG_EXPOSURE = nc::linspace<float>(-3.0, 4.0, 256);

    // Load the base CMFs from the bundled data
    auto cmfs_raw = colour::get_cie_1931_2_degree_cmfs();
    
    // Align them to our project's spectral shape and assign to the global variable
    auto aligned_cmfs = colour::align(cmfs_raw, SPECTRAL_SHAPE);
    
    // Extract only the x, y, z columns (skip wavelength column) to match Python colour library
    size_t num_rows = aligned_cmfs.shape().rows;
    STANDARD_OBSERVER_CMFS = nc::NdArray<float>(num_rows, 3);
    
    for (size_t i = 0; i < num_rows; ++i) {
        STANDARD_OBSERVER_CMFS(i, 0) = aligned_cmfs(i, 1); // x
        STANDARD_OBSERVER_CMFS(i, 1) = aligned_cmfs(i, 2); // y
        STANDARD_OBSERVER_CMFS(i, 2) = aligned_cmfs(i, 3); // z
    }
}

} // namespace config
} // namespace agx
