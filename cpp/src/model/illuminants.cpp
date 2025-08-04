// SPDX-License-Identifier: MIT
//
// Implementation of illuminant helper functions.
//
// This file provides C++ versions of a subset of the Python
// ``agx_emulsion/model/illuminants.py`` functions.  It relies on the
// NumCpp library for numerical array handling and reuses the global
// filter instances from ``color_filters.hpp`` to reproduce the
// tungsten + filter illuminant types.  Because the C++ port does not
// include the full ``colour`` Python package, only a limited set of
// illuminant labels are supported.  See the header for details.

#include "illuminants.hpp"

#include <cmath>
#include <stdexcept>
#include <string>

namespace agx {
namespace model {

namespace {
// Physical constants used by Planck's law (SI units)
static constexpr double PLANCK_CONSTANT = 6.62607015e-34;  // J·s
static constexpr double SPEED_OF_LIGHT  = 2.99792458e8;     // m/s
static constexpr double BOLTZMANN_CONST = 1.380649e-23;     // J/K
}

nc::NdArray<float> black_body_spectrum(double temperature)
{
    // Retrieve the global wavelength grid (in nanometres) and flatten to a 1D array
    auto wl = agx::config::SPECTRAL_SHAPE.wavelengths.flatten();
    const std::size_t n = wl.size();
    nc::NdArray<float> values = nc::zeros<float>({1, n}).flatten();
    // Compute the spectral power distribution according to Planck's law
    for (std::size_t i = 0; i < n; ++i) {
        // Convert wavelength from nanometres to metres
        double lambda_m = static_cast<double>(wl[i]) * 1.0e-9;
        // Guard against zero wavelength to avoid division by zero
        if (lambda_m <= 0.0) {
            values[i] = 0.0f;
            continue;
        }
        // Exponential term: hc / (λ k T)
        const double exponent = (PLANCK_CONSTANT * SPEED_OF_LIGHT) / (lambda_m * BOLTZMANN_CONST * temperature);
        // Compute numerator (2hc^2)
        const double numerator = 2.0 * PLANCK_CONSTANT * SPEED_OF_LIGHT * SPEED_OF_LIGHT;
        // Compute denominator λ^5 (exp(exponent) - 1)
        const double lambda5 = std::pow(lambda_m, 5);
        double denom = std::exp(exponent) - 1.0;
        // Handle potential overflow: if exponent is huge, exp() may overflow and denom becomes inf
        if (!std::isfinite(denom) || denom <= 0.0) {
            values[i] = 0.0f;
        } else {
            double intensity = numerator / (lambda5 * denom);
            values[i] = static_cast<float>(intensity);
        }
    }
    return values;
}

nc::NdArray<float> standard_illuminant(const std::string& type)
{
    nc::NdArray<float> spectrum;
    // Detect black‑body illuminant labels of the form "BBXXXX"
    if (type.size() >= 2 && type.rfind("BB", 0) == 0) {
        // Extract temperature substring after "BB"
        std::string temp_str = type.substr(2);
        double temperature = 6500.0;
        try {
            temperature = std::stod(temp_str);
        } catch (const std::exception&) {
            // Fall back to default
            temperature = 6500.0;
        }
        spectrum = black_body_spectrum(temperature);
    }
    // Tungsten lamp with KG3 filter
    else if (type == "TH-KG3") {
        spectrum = black_body_spectrum(3200.0);
        // Ensure global filters are available
        if (!schott_kg3_heat_filter) {
            initialize_global_filters();
        }
        // Apply the heat filter with full strength (value = 1)
        spectrum = schott_kg3_heat_filter->apply(spectrum, 1.0f).flatten();
    }
    // Tungsten lamp with KG3 and lens transmission
    else if (type == "TH-KG3-L") {
        spectrum = black_body_spectrum(3200.0);
        if (!schott_kg3_heat_filter || !generic_lens_transmission) {
            initialize_global_filters();
        }
        spectrum = schott_kg3_heat_filter->apply(spectrum, 1.0f).flatten();
        spectrum = generic_lens_transmission->apply(spectrum, 1.0f).flatten();
    }
    // Unsupported or daylight approximations default to 6500 K black‑body
    else {
        spectrum = black_body_spectrum(6500.0);
    }
    // Normalise by the mean value of the spectrum
    const std::size_t n = spectrum.size();
    if (n == 0) {
        return spectrum;
    }
    double sum = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum += static_cast<double>(spectrum[i]);
    }
    const double mean = sum / static_cast<double>(n);
    if (mean > 0.0) {
        for (std::size_t i = 0; i < n; ++i) {
            double normalised = static_cast<double>(spectrum[i]) / mean;
            spectrum[i] = static_cast<float>(normalised);
        }
    }
    return spectrum;
}

} // namespace model
} // namespace agx