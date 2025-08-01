#pragma once

#include "NumCpp.hpp"

namespace agx {
namespace utils {

/**
 * @brief Measures gamma from density curves using cubic interpolation.
 * 
 * @param log_exposure Array of log exposure values
 * @param density_curves Array of density values (rows: exposures, cols: RGB)
 * @param density_0 Lower density point for gamma calculation (default: 0.25)
 * @param density_1 Upper density point for gamma calculation (default: 1.0)
 * @return nc::NdArray<float> Gamma values for each channel (RGB)
 */
nc::NdArray<float> measure_gamma(const nc::NdArray<float>& log_exposure, 
                                const nc::NdArray<float>& density_curves,
                                float density_0 = 0.25f, 
                                float density_1 = 1.0f);

/**
 * @brief Measures slopes at specific exposure points using cubic spline interpolation.
 * 
 * @param log_exposure Array of log exposure values
 * @param density_curves Array of density values (rows: exposures, cols: RGB)
 * @param log_exposure_reference Reference exposure point (default: 0.0)
 * @param log_exposure_range Range around reference point (default: log10(2^2))
 * @return nc::NdArray<float> Gamma values for each channel (RGB)
 */
nc::NdArray<float> measure_slopes_at_exposure(const nc::NdArray<float>& log_exposure,
                                             const nc::NdArray<float>& density_curves,
                                             float log_exposure_reference = 0.0f,
                                             float log_exposure_range = std::log10(4.0f));

} // namespace utils
} // namespace agx 