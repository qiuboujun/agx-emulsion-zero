#pragma once

#include "NumCpp.hpp"
#include <string>

namespace agx { namespace utils {

nc::NdArray<float> density_to_light(const nc::NdArray<float>& density, const nc::NdArray<float>& light);
float density_to_light(float density, float light);

nc::NdArray<float> compute_densitometer_correction(const nc::NdArray<float>& dye_density, const std::string& type = "status_A");

nc::NdArray<float> compute_aces_conversion_matrix(const nc::NdArray<float>& sensitivity, const nc::NdArray<float>& illuminant);

std::pair<nc::NdArray<float>, nc::NdArray<float>> rgb_to_raw_aces_idt(
    const nc::NdArray<float>& RGB,
    const nc::NdArray<float>& illuminant,
    const nc::NdArray<float>& sensitivity,
    nc::NdArray<float> midgray_rgb = nc::NdArray<float>(),
    const std::string& color_space = "sRGB",
    bool apply_cctf_decoding = true,
    nc::NdArray<float> aces_conversion_matrix = nc::NdArray<float>());

} } // namespace agx::utils
