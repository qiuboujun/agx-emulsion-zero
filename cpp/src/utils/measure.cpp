#include "NumCpp.hpp"
#include <functional>
#include <cmath>
#include "scipy.hpp"
#include <algorithm>
#include <vector>

namespace agx {
namespace utils {

/**
 * @brief Measures gamma from density curves using cubic interpolation.
 * 
 * This function calculates gamma (slope) between two density points using
 * cubic interpolation to find the corresponding log exposure values.
 * 
 * @param log_exposure Array of log exposure values
 * @param density_curves Array of density values (rows: exposures, cols: RGB)
 * @param density_0 Lower density point for gamma calculation (default: 0.25)
 * @param density_1 Upper density point for gamma calculation (default: 1.0)
 * @return nc::NdArray<float> Gamma values for each channel (RGB)
 */
nc::NdArray<float> measure_gamma(const nc::NdArray<float>& log_exposure, 
                                const nc::NdArray<float>& density_curves,
                                float density_0, 
                                float density_1) {
    nc::NdArray<float> gamma(1, 3);
    
    for (int i = 0; i < 3; ++i) {
        // Extract the i-th channel density curve using proper slicing
        nc::NdArray<float> density_channel = density_curves(nc::Slice(0, density_curves.shape().rows), i);
        
        // Convert to NumCpp arrays (no need to sort manually - interp1d handles it)
        nc::NdArray<double> density_channel_double = density_channel.astype<double>();
        nc::NdArray<double> log_exposure_double = log_exposure.astype<double>();

        // Create interp1d interpolator: density -> log_exposure (inverse interpolation)
        // This matches Python's interp1d(density_curves[:, i], log_exposure, kind='cubic')
        scipy::interpolate::interp1d interp(density_channel_double, log_exposure_double, 
                                           scipy::interpolate::interp1d::Kind::Cubic);

        // Interpolate to find log exposure values at the target densities
        double loge0 = interp(static_cast<double>(density_0));
        double loge1 = interp(static_cast<double>(density_1));

        // Calculate gamma
        gamma[i] = static_cast<float>((density_1 - density_0) / (loge1 - loge0));
    }
    
    return gamma;
}

/**
 * @brief Measures slopes at specific exposure points using cubic spline interpolation.
 * 
 * This function calculates the slope (gamma) at a reference exposure point
 * over a specified range using cubic spline interpolation.
 * 
 * @param log_exposure Array of log exposure values
 * @param density_curves Array of density values (rows: exposures, cols: RGB)
 * @param log_exposure_reference Reference exposure point (default: 0.0)
 * @param log_exposure_range Range around reference point (default: log10(2^2))
 * @return nc::NdArray<float> Gamma values for each channel (RGB)
 */
nc::NdArray<float> measure_slopes_at_exposure(const nc::NdArray<float>& log_exposure,
                                             const nc::NdArray<float>& density_curves,
                                             float log_exposure_reference,
                                             float log_exposure_range) {
    float le_ref = log_exposure_reference;
    float log_exposure_0 = le_ref - log_exposure_range / 2.0f;
    float log_exposure_1 = le_ref + log_exposure_range / 2.0f;
    
    nc::NdArray<float> gamma(1, 3);
    
    for (int i = 0; i < 3; ++i) {
        // Extract the i-th channel density curve using proper slicing
        nc::NdArray<float> density_channel = density_curves(nc::Slice(0, density_curves.shape().rows), i);
        
        // Create mask for non-NaN values (equivalent to ~np.isnan)
        std::vector<double> valid_log_exposure;
        std::vector<double> valid_density;
        
        for (size_t j = 0; j < density_channel.size(); ++j) {
            if (!std::isnan(density_channel[j])) {
                valid_log_exposure.push_back(static_cast<double>(log_exposure[j]));
                valid_density.push_back(static_cast<double>(density_channel[j]));
            }
        }
        
        if (valid_density.size() < 2) {
            gamma[i] = std::numeric_limits<float>::quiet_NaN();
            continue;
        }
        
        // Convert to NumCpp arrays
        nc::NdArray<double> valid_log_exposure_nc(valid_log_exposure);
        nc::NdArray<double> valid_density_nc(valid_density);
        
        // Create cubic spline interpolator: log_exposure -> density
        scipy::interpolate::CubicSpline interp(valid_log_exposure_nc, valid_density_nc);
        
        // Interpolate to find density values at the target log exposures
        double density_1 = interp(static_cast<double>(log_exposure_1));
        double density_0 = interp(static_cast<double>(log_exposure_0));
        
        // Calculate gamma
        gamma[i] = static_cast<float>((density_1 - density_0) / (log_exposure_1 - log_exposure_0));
    }
    
    return gamma;
}

} // namespace utils
} // namespace agx
