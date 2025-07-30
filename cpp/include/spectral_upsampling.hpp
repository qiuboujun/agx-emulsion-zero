#pragma once

#include <string>
#include <vector>
#include <utility> // For std::pair

// The only necessary include for the NumCpp library
#include "NumCpp.hpp"

// Assumed headers for your custom libraries, which should also use NumCpp.
#include "colour.hpp"
#include "scipy.hpp"

// Forward-declare functions from other modules that are used here
namespace agx {
    namespace model {
        // This function is expected to be defined elsewhere and return a NumCpp array
        nc::NdArray<float> standard_illuminant(const std::string& illuminant_label);
    }
}

namespace agx {
namespace utils {

//================================================================================
// LUT Generation
//================================================================================

/**
 * @brief Loads the coefficients LUT from a binary file.
 * @param filename The path to the .lut file.
 * @return An nc::NdArray containing the LUT coefficients.
 */
nc::NdArray<float> load_coeffs_lut(const std::string& filename = "hanatos_irradiance_xy_coeffs_250304.lut");

/**
 * @brief Converts triangular coordinates into square coordinates using a CUDA kernel.
 */
nc::NdArray<float> tri2quad(const nc::NdArray<float>& tc);

/**
 * @brief Converts square coordinates into triangular coordinates using a CUDA kernel.
 */
nc::NdArray<float> quad2tri(const nc::NdArray<float>& xy);

/**
 * @brief Fetches spectral upsampling coefficients by interpolating a LUT.
 */
nc::NdArray<float> fetch_coeffs(const nc::NdArray<float>& tc, const nc::NdArray<float>& lut_coeffs);

/**
 * @brief Computes spectra from coefficients, including smoothing and resampling.
 */
nc::NdArray<float> compute_spectra_from_coeffs(const nc::NdArray<float>& coeffs, int smooth_steps = 1);

/**
 * @brief Generates the full spectral LUT.
 */
nc::NdArray<float> compute_lut_spectra(int lut_size = 128, int smooth_steps = 1, const std::string& lut_coeffs_filename = "hanatos_irradiance_xy_coeffs_250304.lut");

/**
 * @brief Loads a pre-computed spectral LUT from a .npy file.
 */
nc::NdArray<float> load_spectra_lut(const std::string& filename = "irradiance_xy_tc.npy");

/**
 * @brief Calculates the xy chromaticity of a standard illuminant.
 */
nc::NdArray<float> illuminant_to_xy(const std::string& illuminant_label);

/**
 * @brief Converts RGB to triangular coordinates and a brightness factor.
 */
std::pair<nc::NdArray<float>, nc::NdArray<float>> rgb_to_tc_b(
    const nc::NdArray<float>& rgb,
    const std::string& color_space = "ITU-R BT.2020",
    bool apply_cctf_decoding = false,
    const std::string& reference_illuminant = "D55"
);

//================================================================================
// Band Pass Filter
//================================================================================

/**
 * @brief Computes a spectral band pass filter.
 */
nc::NdArray<float> compute_band_pass_filter(
    const nc::NdArray<float>& filter_uv = {1.0f, 410.0f, 8.0f},
    const nc::NdArray<float>& filter_ir = {1.0f, 675.0f, 15.0f}
);

//================================================================================
// Spectral Recovery Methods
//================================================================================

/**
 * @brief Converts RGB to raw sensor response using Mallett et al. (2019).
 */
nc::NdArray<float> rgb_to_raw_mallett2019(
    const nc::NdArray<float>& RGB,
    const nc::NdArray<float>& sensitivity,
    const std::string& color_space = "sRGB",
    bool apply_cctf_decoding = true,
    const std::string& reference_illuminant = "D65"
);

/**
 * @brief Converts RGB to raw sensor response using the Hanatos (2025) method.
 */
nc::NdArray<float> rgb_to_raw_hanatos2025(
    const nc::NdArray<float>& rgb,
    const nc::NdArray<float>& sensitivity,
    const std::string& color_space,
    bool apply_cctf_decoding,
    const std::string& reference_illuminant
);

/**
 * @brief Converts an RGB value to a full spectrum.
 */
nc::NdArray<float> rgb_to_spectrum(
    const nc::NdArray<float>& rgb,
    const std::string& color_space,
    bool apply_cctf_decoding,
    const std::string& reference_illuminant
);

/**
 * @brief An example function demonstrating the main workflow, corresponding to `if __name__ == '__main__':`.
 */
void run_spectral_upsampling_example();

} // namespace utils
} // namespace agx