#pragma once

#include "NumCpp.hpp"
#include <string>
#include <utility>
#include <functional>

namespace agx { namespace utils {

// Build a 3D LUT for camera RGB->RAW mapping using our rgb_to_raw_hanatos2025 implementation.
// Returns a flattened (L*L*L x 3) LUT.
nc::NdArray<float> create_lut_3d_camera_rgb_to_raw(
    int steps,
    const nc::NdArray<float>& sensitivity,           // K x 3
    const std::string& color_space,
    bool apply_cctf_decoding,
    const std::string& reference_illuminant,
    const nc::NdArray<float>& spectra_lut            // (L2*L2 x K) preloaded
);

// Apply a 3D LUT to RGB data (H*W x 3) using cubic interpolation.
// Wrapper over apply_lut_cubic_3d in fast_interp_lut.
nc::NdArray<float> apply_lut_3d(const nc::NdArray<float>& lut_flat,
                                const nc::NdArray<float>& rgb_hw_by3,
                                int height,
                                int width);

// Convenience: compute camera raw using LUT with given resolution; returns (raw, lut)
std::pair<nc::NdArray<float>, nc::NdArray<float>> compute_camera_with_lut(
    const nc::NdArray<float>& rgb_hw_by3,
    int height,
    int width,
    const nc::NdArray<float>& sensitivity,
    const std::string& color_space,
    bool apply_cctf_decoding,
    const std::string& reference_illuminant,
    const nc::NdArray<float>& spectra_lut,
    int steps);

} } // namespace agx::utils

#pragma once

#include "NumCpp.hpp"

namespace agx {
namespace utils {

/**
 * @brief Creates a 3D lookup table from a function.
 * 
 * @param function A function that takes a 3D coordinate array and returns transformed values
 * @param xmin Minimum value for the input range
 * @param xmax Maximum value for the input range
 * @param steps Number of steps in each dimension
 * @return nc::NdArray<float> The 3D LUT flattened to (steps^3, 3)
 */
nc::NdArray<float> _create_lut_3d(
    const std::function<nc::NdArray<float>(const nc::NdArray<float>&)>& function,
    float xmin, 
    float xmax, 
    int steps);

/**
 * @brief Computes data transformation using a 3D LUT.
 * 
 * @param data Input data array
 * @param function Function to create the LUT from
 * @param xmin Minimum value for the input range
 * @param xmax Maximum value for the input range
 * @param steps Number of steps in each dimension
 * @return std::pair<nc::NdArray<float>, nc::NdArray<float>> Pair of (transformed_data, lut)
 */
std::pair<nc::NdArray<float>, nc::NdArray<float>> compute_with_lut(
    const nc::NdArray<float>& data,
    const std::function<nc::NdArray<float>(const nc::NdArray<float>&)>& function,
    float xmin,
    float xmax,
    int steps);

/**
 * @brief Performs a warmup for both 3D and 2D LUT functions.
 * This ensures that any initialization overhead is incurred only once.
 */
void warmup_luts();

} // namespace utils
} // namespace agx 