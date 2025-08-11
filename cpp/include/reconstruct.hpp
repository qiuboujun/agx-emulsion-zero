// SPDX-License-Identifier: MIT

#pragma once

#include <array>
#include <string>
#include "NumCpp.hpp"
#include "profile_io.hpp"

namespace agx {
namespace profiles {

// Parameter set mirroring reconstruct.py defaults for model 'dmid_dmin'
struct ReconstructParams {
    // Dye parameters
    float dye_amp[3]   = {1.0f, 1.0f, 1.0f};
    float dye_width[3] = {1.0f, 1.0f, 1.0f};
    float dye_shift[3] = {0.0f, 0.0f, 0.0f};

    // Coupler Gaussian parameters (five profiles assigned to CMY channels)
    float cpl_amp[5]   = {0.1f, 0.1f, 0.1f, 0.03f, 0.1f};
    float cpl_width[5] = {20.f, 20.f, 20.f, 40.f, 20.f};
    float cpl_max[5]   = {435.f, 560.f, 475.f, 700.f, 510.f};

    // Other parameters
    float dmax[3]      = {2.3f, 2.3f, 2.3f};
    float fog[3]       = {0.07f, 0.07f, 0.07f};
    float scat400      = 0.65f;
    float base         = 0.05f;
};

// Core helpers (equivalents of the Python functions)
nc::NdArray<float> low_pass_filter(
    const nc::NdArray<float>& wl,
    float wl_max, float width, float amp = 1.0f);

nc::NdArray<float> high_pass_filter(
    const nc::NdArray<float>& wl,
    float wl_min, float width, float amp = 1.0f);

nc::NdArray<float> high_pass_gaussian(
    const nc::NdArray<float>& wl,
    float wl_max, float width, float amount);

nc::NdArray<float> low_pass_gaussian(
    const nc::NdArray<float>& wl,
    float wl_max, float width, float amount);

nc::NdArray<float> shift_stretch(
    const nc::NdArray<float>& wl,
    const nc::NdArray<float>& spectrum,
    float amp = 1.0f, float width = 1.0f, float shift = 0.0f);

nc::NdArray<float> shift_stretch_cmy(
    const nc::NdArray<float>& wl,
    const nc::NdArray<float>& cmy, // (N,3)
    float da0, float dw0, float ds0,
    float da1, float dw1, float ds1,
    float da2, float dw2, float ds2);

nc::NdArray<float> gaussian_profiles(
    const nc::NdArray<float>& wl,
    const std::array<std::array<float,3>,5>& p_couplers);

// Models
struct DensityMidMinResult {
    nc::NdArray<float> cmy;        // (N,3)
    nc::NdArray<float> dye;        // (N,3)
    nc::NdArray<float> filters;    // (N,3) or (N,5) depending on model; here (N,3)
    nc::NdArray<float> dmin;       // (N,)
};

DensityMidMinResult density_mid_min_model(
    const ReconstructParams& params,
    const nc::NdArray<float>& wl,
    const nc::NdArray<float>& cmy_model,
    const std::string& model = "dmid_dmin");

// Additional helpers
nc::NdArray<float> compute_densitometer_crosstalk_matrix(
    const nc::NdArray<float>& densitometer_intensity, // (N,3)
    const nc::NdArray<float>& dye_density             // (N,3)
);

nc::NdArray<float> slopes_of_concentrations(
    const nc::NdArray<float>& log_exposure,
    const nc::NdArray<float>& density_curves,
    const nc::NdArray<float>& dstm_cm);

// Entry point similar to reconstruct_dye_density() in Python (no optimiser; uses params as-is)
Profile reconstruct_dye_density(
    Profile profile,
    const ReconstructParams& params,
    bool print_params = false);

} // namespace profiles
} // namespace agx


