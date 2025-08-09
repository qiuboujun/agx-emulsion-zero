// SPDX-License-Identifier: MIT
// C++ reimplementation of agx_emulsion/profiles/balance.py

#pragma once

#include <array>
#include <string>
#include <vector>

#include "NumCpp.hpp"
#include "io.hpp"

namespace agx {
namespace profiles {

struct BalanceResultMetameric {
    std::array<float, 3> d_cmy_metameric;  // fitted absolute densities
    std::array<float, 3> d_cmy_scale;      // scaled to green = 1
    nc::NdArray<float> dye_density_out;    // updated dye_density (Nx5)
};

/**
 * @brief Balance the channel sensitivities so a neutral illuminant yields equal exposure.
 * Mirrors balance_sensitivity in Python. Optionally shifts density curves along log exposure axis.
 */
void balance_sensitivity(
    AgxEmulsionData &data,
    const std::string &reference_illuminant,
    bool correct_log_exposure = true);

/**
 * @brief Balance density curves so M and Y intersect G at density at log_exposure=0.
 * Mirrors balance_density in Python.
 */
void balance_density(AgxEmulsionData &data);

/**
 * @brief Find metameric neutral CMY densities so that resulting sRGB equals mid-gray.
 * Mirrors balance_metameric_neutral in Python. Returns scales and updated dye_density.
 */
BalanceResultMetameric balance_metameric_neutral(
    const nc::NdArray<float> &dye_density,
    const std::string &viewing_illuminant,
    float midgray_value = 0.184f);

// Variant that accepts an explicit illuminant spectrum aligned to the
// global spectral shape, to ensure exact parity with Python reference.
BalanceResultMetameric balance_metameric_neutral_with_illuminant(
    const nc::NdArray<float> &dye_density,
    const nc::NdArray<float> &illuminant,
    float midgray_value = 0.184f);

// Debug helper: compute sRGB from a given mid density built as
// mid = dye_density[:,:3] @ density_cmy + dye_density[:,3]
nc::NdArray<float> debug_rgb_from_density_params(
    const nc::NdArray<float> &dye_density,
    const std::array<float,3> &density_cmy,
    const std::string &viewing_illuminant);

} // namespace profiles
} // namespace agx


