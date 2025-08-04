// SPDX-License-Identifier: MIT
//
// agx_emulsion/model/illuminants.cpp reimplementation in C++.
//
// This header declares functions to generate illuminant spectra on the
// project's wavelength grid.  It mirrors the behaviour of the original
// Python implementation found in ``agx_emulsion/model/illuminants.py``.
//
// The functions here rely on the NumCpp library for array handling and
// the existing global filter instances declared in ``color_filters.hpp``.
// A simple black‑body spectrum generator is provided along with a
// convenience wrapper ``standard_illuminant`` that supports a subset of
// the Python illuminant types ("BBxxxx" for arbitrary colour
// temperatures, "TH-KG3" and "TH-KG3-L" for tungsten lamps with
// heat‑absorbing and lens filters).  Unknown illuminant labels fall
// back to a 6500 K black‑body approximation.

#pragma once

#include <string>

#include "NumCpp.hpp"
#include "config.hpp"
#include "color_filters.hpp"

namespace agx {
namespace model {

/**
 * @brief Compute the spectral power distribution of a black body.
 *
 * This function evaluates Planck's law for the wavelengths defined by
 * ``agx::config::SPECTRAL_SHAPE`` and the supplied temperature.  The
 * resulting array is one‑dimensional and not normalised.
 *
 * @param temperature Colour temperature in Kelvin.
 * @return nc::NdArray containing the unnormalised spectral power
 *         distribution for each wavelength.
 */
nc::NdArray<float> black_body_spectrum(double temperature);

/**
 * @brief Generate a normalised illuminant spectrum.
 *
 * This function is a C++ analogue of the Python ``standard_illuminant``
 * function.  It supports a subset of illuminant labels:
 *
 *  - ``"BBXXXX"``: A black‑body illuminant at the given colour
 *    temperature (e.g. "BB3200" or "BB6500").
 *  - ``"TH-KG3"``: A 3200 K black‑body spectrum passed through the
 *    KG3 heat‑absorbing filter (see ``schott_kg3_heat_filter``).
 *  - ``"TH-KG3-L"``: A 3200 K black‑body spectrum with both the KG3
 *    heat filter and a generic lens transmission applied.
 *
 * Any other illuminant label will result in a 6500 K black‑body
 * approximation.  The returned spectrum is always normalised such
 * that the mean of the values equals 1.0.
 *
 * @param type Illuminant label (default "D65" falls back to a 6500 K
 *             black‑body).
 * @return nc::NdArray of length equal to the number of wavelengths in
 *         ``agx::config::SPECTRAL_SHAPE`` representing the normalised
 *         illuminant spectrum.
 */
nc::NdArray<float> standard_illuminant(const std::string& type = "D65");

} // namespace model
} // namespace agx