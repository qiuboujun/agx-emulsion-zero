#pragma once

#include "NumCpp.hpp"
#include "io.hpp"
#include "config.hpp"        // For load_densitometer_data
#include <cmath>         // For std::pow, std::isnan
#include <utility>       // For std::pair
#include <string>
#include <fstream>
#include <sstream>

namespace agx {
namespace utils {
/**
 * @brief Convert density to light transmittance.
 *
 * This function calculates the transmitted light intensity based on the given density and initial light intensity.
 * It uses the formula **transmittance = 10^(-density)** and then multiplies by the light intensity.
 * Any NaN values in the result are replaced with 0.
 *
 * @param density A float or nc::NdArray<float> representing the density value(s) that affect light transmittance.
 * @param light A float or nc::NdArray<float> for the initial light intensity value(s), same shape or broadcastable to `density`.
 * @return nc::NdArray<float> The light intensity after passing through the medium with the given density (same shape as input).
 */
inline nc::NdArray<float> density_to_light(const nc::NdArray<float>& density, const nc::NdArray<float>& light) {
    // Compute transmittance = 10^(-density)
    nc::NdArray<float> transmitted = nc::power(10.0f, -density);
    // Multiply by light (broadcast if necessary)
    if (light.shape().rows == 1 && light.shape().cols == density.shape().cols) {
        // Light is 1xN, broadcast across all rows of density
        for (size_t j = 0; j < density.shape().cols; ++j) {
            transmitted(nc::Slice(), j) *= light(0, j);
        }
    } else if (light.shape().cols == 1 && light.shape().rows == density.shape().cols) {
        // Light is Nx1, broadcast across columns
        for (size_t j = 0; j < density.shape().cols; ++j) {
            transmitted(nc::Slice(), j) *= light(j, 0);
        }
    } else if (light.size() == 1) {
        // Light is a scalar in an NdArray
        transmitted *= light[0];
    } else {
        // Shapes match or are already aligned
        transmitted *= light;
    }
    // Replace NaN values with 0
    for (auto it = transmitted.begin(); it != transmitted.end(); ++it) {
        if (std::isnan(*it)) {
            *it = 0.0f;
        }
    }
    return transmitted;
}

// Overload for single float inputs (convenience)
inline float density_to_light(float density, float light) {
    float transmitted = std::pow(10.0f, -density) * light;
    return std::isnan(transmitted) ? 0.0f : transmitted;
}

/**
 * @brief Compute densitometer correction factors for density measurements.
 *
 * This function computes a correction factor for each channel (assuming 3 color channels) given the dye densities and a densitometer type.
 * It uses the loaded densitometer spectral responsivities to weight the dye densities and returns 1 / (responsivity ⋅ dye_density) for each channel.
 *
 * @param dye_density An nc::NdArray<float> of shape (M,4) or similar, containing spectral dye density data. Only the first 3 columns (e.g. C, M, Y dye densities) are used.
 * @param type A string specifying the densitometer type (e.g., "status_A"). Default is "status_A".
 * @return nc::NdArray<float> A 1x3 array containing the densitometer correction factors for the three channels.
 */
inline nc::NdArray<float> compute_densitometer_correction(const nc::NdArray<float>& dye_density, const std::string& type = "status_A") {
    // Load densitometer spectral responsivities (shape: [N,3] for R,G,B channels)
    nc::NdArray<float> responsivities = agx::utils::load_densitometer_data(type);
    // Use only the first 3 columns of dye_density (assume shape [N, >=4])
    nc::NdArray<float> dye = dye_density(nc::Slice(), nc::Slice(0, 3)).copy();
    // Replace NaNs in dye_density with 0
    for (auto it = dye.begin(); it != dye.end(); ++it) {
        if (std::isnan(*it)) {
            *it = 0.0f;
        }
    }
    // Element-wise product of responsivities and dye_density (shape [N,3])
    nc::NdArray<float> product = responsivities * dye;
    // Sum over the spectral axis (summing each column, result is 1x3)
    nc::NdArray<float> sums = nc::sum(product, nc::Axis::ROW);
    // Compute correction = 1 / sums for each channel
    nc::NdArray<float> correction(1, sums.shape().cols);
    for (size_t j = 0; j < sums.shape().cols; ++j) {
        correction(0, j) = 1.0f / sums(0, j);
    }
    return correction;
}

/**
 * @brief Computes the ACES (Academy Color Encoding System) conversion matrix for the given sensor sensitivity and illuminant.
 *
 * This function calculates the 3x3 matrix that converts from ACES2065-1 color space to the camera's raw RGB space (i.e., the ACES Input Device Transform matrix inverse).
 * It takes into account the spectral sensitivity of the camera (sensor), the illuminant spectral distribution, the CIE 1931 2° standard observer, 
 * and performs chromatic adaptation to ACES white point (D60).
 *
 * @param sensitivity nc::NdArray<float> of shape [N,3] representing the camera RGB spectral sensitivity curves.
 * @param illuminant nc::NdArray<float> of shape [N] (or [N,1]) representing the illuminant spectral power distribution (aligned to the same wavelengths as sensitivity).
 * @return nc::NdArray<float> A 3x3 matrix (nc::NdArray) that converts from ACES2065-1 RGB to raw camera RGB.
 */
inline nc::NdArray<float> compute_aces_conversion_matrix(const nc::NdArray<float>& sensitivity, const nc::NdArray<float>& illuminant) {
    // Dimensions check
    size_t N = sensitivity.shape().rows;
    if (sensitivity.shape().cols != 3 || illuminant.flatten().size() != N) {
        throw std::invalid_argument("Sensitivity must be N×3 and illuminant length must match N.");
    }
    nc::NdArray<float> illum = illuminant.flatten();
    
    // Use the new matrix_idt function from colour.hpp
    auto [M, RGB_w] = colour::matrix_idt(sensitivity, illum);
    
    // Invert the IDT matrix to get ACES to camera (raw) matrix
    // This is the same as np.linalg.inv(M) in Python
    float det = M(0,0)*M(1,1)*M(2,2) + M(0,1)*M(1,2)*M(2,0) + M(0,2)*M(1,0)*M(2,1)
              - M(0,2)*M(1,1)*M(2,0) - M(0,1)*M(1,0)*M(2,2) - M(0,0)*M(1,2)*M(2,1);
    
    if (std::fabs(det) < 1e-12) {
        throw std::runtime_error("Singular matrix in ACES conversion computation");
    }
    
    // Compute cofactors for inverse
    float C00 =  M(1,1)*M(2,2) - M(1,2)*M(2,1);
    float C01 = -(M(1,0)*M(2,2) - M(1,2)*M(2,0));
    float C02 =  M(1,0)*M(2,1) - M(1,1)*M(2,0);
    float C10 = -(M(0,1)*M(2,2) - M(0,2)*M(2,1));
    float C11 =  M(0,0)*M(2,2) - M(0,2)*M(2,0);
    float C12 = -(M(0,0)*M(1,2) - M(0,2)*M(1,0));
    float C20 =  M(0,1)*M(2,0) - M(0,0)*M(2,1);
    float C21 = -(M(0,1)*M(1,2) - M(0,2)*M(1,1));
    float C22 =  M(0,0)*M(1,1) - M(0,1)*M(1,0);
    
    // Adjugate transpose for inverse
    nc::NdArray<float> M_inv(3, 3);
    M_inv(0, 0) = C00 / det;
    M_inv(0, 1) = C10 / det;
    M_inv(0, 2) = C20 / det;
    M_inv(1, 0) = C01 / det;
    M_inv(1, 1) = C11 / det;
    M_inv(1, 2) = C21 / det;
    M_inv(2, 0) = C02 / det;
    M_inv(2, 1) = C12 / det;
    M_inv(2, 2) = C22 / det;
    
    return M_inv;
}

/**
 * @brief Converts RGB values to raw camera RGB values using the ACES Input Device Transform (IDT).
 *
 * This function converts an input RGB image or value from a given color space into the camera's raw RGB space, using the ACES IDT procedure.
 * It first transforms the input RGB to ACES2065-1 color space (linear AP0, D60 white), then applies the ACES-to-raw conversion matrix (from compute_aces_conversion_matrix).
 * Finally, it normalizes the output such that a mid-gray (18% reflectance) in the input becomes [1,1,1] in the raw output.
 *
 * @param RGB An nc::NdArray<float> containing the input RGB values. This can be a single RGB triplet (shape 1x3) or an array of shape [M,3] (or [H*W,3] for an image).
 * @param illuminant An nc::NdArray<float> for the illuminant spectral distribution (to compute the IDT matrix if needed).
 * @param sensitivity An nc::NdArray<float> for the camera spectral sensitivity (dimensions matching illuminant; used for IDT matrix computation).
 * @param midgray_rgb (Optional) An nc::NdArray<float> for the mid-gray RGB in the input color space. Default is [0.184, 0.184, 0.184] (18% gray) for each channel.
 * @param color_space (Optional) The color space of the input RGB values (e.g., "sRGB"). Default is "sRGB".
 * @param apply_cctf_decoding (Optional) Whether to apply the decoding of the input color component transfer function (gamma). Default is true (needed for sRGB).
 * @param aces_conversion_matrix (Optional) A precomputed 3x3 ACES-to-raw conversion matrix (from compute_aces_conversion_matrix). If not provided, it will be computed.
 * @return std::pair<nc::NdArray<float>, nc::NdArray<float>> A pair containing:
 *         - first: the raw camera RGB values (same shape as input RGB).
 *         - second: the raw mid-gray value (as a 1x3 array, typically [1,1,1]).
 */
inline std::pair<nc::NdArray<float>, nc::NdArray<float>> 
rgb_to_raw_aces_idt(const nc::NdArray<float>& RGB,
                    const nc::NdArray<float>& illuminant,
                    const nc::NdArray<float>& sensitivity,
                    nc::NdArray<float> midgray_rgb = nc::NdArray<float>(),
                    const std::string& color_space = "sRGB",
                    bool apply_cctf_decoding = true,
                    nc::NdArray<float> aces_conversion_matrix = nc::NdArray<float>()) {
    // Determine mid-gray values (default to 0.184 for each channel if not provided)
    float midgray_val = 0.184f;
    if (midgray_rgb.size() > 0) {
        // Use the first element of each channel if provided
        nc::NdArray<float> mg = midgray_rgb.flatten();
        if (mg.size() >= 3) {
            // Assume neutral grey, take first channel as representative (or all three if they differ)
            midgray_val = mg[0];
        }
    }
    // Convert RGB to ACES2065-1 using color science API
    // This handles CCTF decoding, color space conversion, and chromatic adaptation automatically
    nc::NdArray<float> aces = colour::RGB_to_RGB(RGB, color_space, "ACES2065-1",
                                                apply_cctf_decoding,  // Apply CCTF decoding as requested
                                                false);               // No CCTF encoding (ACES is linear)
    // Compute the ACES-to-raw conversion matrix if not provided
    nc::NdArray<float> aces_to_raw;
    if (aces_conversion_matrix.size() == 0) {
        aces_to_raw = compute_aces_conversion_matrix(sensitivity, illuminant);
    } else {
        aces_to_raw = aces_conversion_matrix;
    }
    // Multiply ACES values by aces_to_raw matrix to get raw values
    // We will perform: raw = aces @ (aces_to_raw)^T  (broadcasting each row of aces)
    nc::NdArray<float> raw = nc::dot(aces, aces_to_raw.transpose());
    // Divide by mid-gray (normalize such that input mid-gray maps to [1,1,1])
    if (raw.shape().cols == 3) {
        for (size_t j = 0; j < 3; ++j) {
            raw(nc::Slice(), j) /= midgray_val;
        }
    } else {
        // If raw is a single 1x3 vector
        for (uint32_t j = 0; j < raw.size(); ++j) {
            raw[j] /= midgray_val;
        }
    }
    // Prepare raw_midgray array = [1,1,1]
    nc::NdArray<float> raw_midgray(1, 3);
    raw_midgray.fill(1.0f);
    return { raw, raw_midgray };
}

} // namespace utils
} // namespace agx