#pragma once

#include "NumCpp.hpp"
#include <functional>

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