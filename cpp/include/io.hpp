#pragma once

#include "NumCpp.hpp"
#include <string>
#include <vector>
#include <map>
#include <nlohmann/json.hpp>

// A structure to hold the loaded emulsion data, similar to a class or dict in Python
struct AgxEmulsionData {
    nc::NdArray<float> log_sensitivity;
    nc::NdArray<float> dye_density;
    nc::NdArray<float> wavelengths;
    nc::NdArray<float> density_curves;
    nc::NdArray<float> log_exposure;
};

namespace agx {
namespace utils {

//================================================================================
// Interpolation
//================================================================================

nc::NdArray<float> interpolate_to_common_axis(
    const nc::NdArray<float>& data,
    const nc::NdArray<float>& new_x,
    bool extrapolate = false,
    const std::string& method = "akima"
);

//================================================================================
// Data Loading
//================================================================================

nc::NdArray<float> load_csv(const std::string& datapkg, const std::string& filename);

AgxEmulsionData load_agx_emulsion_data(
    const std::string& stock = "kodak_portra_400",
    const std::string& log_sensitivity_donor = "",
    const std::string& density_curves_donor = "",
    const std::string& dye_density_cmy_donor = "",
    const std::string& dye_density_min_mid_donor = "",
    const std::string& type = "negative",
    bool color = true
    // spectral_shape and log_exposure will be handled via a global config or passed differently
);

nc::NdArray<float> load_densitometer_data(
    const std::string& type = "status_A"
);

//================================================================================
// YMC Filter Values & Profiles
//================================================================================

// Using nlohmann::json to represent the JSON object (matches Python behavior)
using FilterValues = nlohmann::json;

void save_ymc_filter_values(const FilterValues& ymc_filters);

FilterValues read_neutral_ymc_filter_values();

nc::NdArray<float> load_dichroic_filters(const nc::NdArray<float>& wavelengths, const std::string& brand = "thorlabs");

nc::NdArray<float> load_filter(
    const nc::NdArray<float>& wavelengths,
    const std::string& name = "KG3",
    const std::string& brand = "schott",
    const std::string& filter_type = "heat_absorbing",
    bool percent_transmittance = false
);


} // namespace utils
} // namespace agx
