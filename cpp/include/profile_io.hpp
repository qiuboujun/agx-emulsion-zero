// SPDX-License-Identifier: MIT

#pragma once

#include <string>
#include <vector>

#include "NumCpp.hpp"
#include <nlohmann/json.hpp>

namespace agx {
namespace profiles {

struct ProfileInfo {
    std::string stock;
    std::string name;
    std::string type;
    bool color;
    std::string densitometer;
    float log_sensitivity_density_over_min;
    std::string reference_illuminant;
    std::string viewing_illuminant;
    std::array<float, 3> density_midscale_neutral;
};

struct ProfileData {
    nc::NdArray<float> log_sensitivity;        // [N,3]
    nc::NdArray<float> density_curves;         // [M,3]
    nc::NdArray<float> density_curves_layers;  // [L,3]
    nc::NdArray<float> dye_density;            // [K,5]
    nc::NdArray<float> log_exposure;           // [M,1] or [M]
    nc::NdArray<float> wavelengths;            // [K,1] or [K]
    std::array<float,3> gamma_factor = {1.0f, 1.0f, 1.0f};
    float dye_density_min_factor = 1.0f;
};

struct Profile {
    ProfileInfo info;
    ProfileData data;
};

class ProfileIO {
public:
    // Load a profile JSON from an absolute or relative path
    static Profile load_from_file(const std::string& json_path);

    // Save a profile JSON to the given path (overwrites existing)
    static void save_to_file(const Profile& profile, const std::string& json_path);
};

// Utility: exact element-wise equality (shape and contents)
bool arrays_equal(const nc::NdArray<float>& a, const nc::NdArray<float>& b);

// Helper function for other parts of the codebase to use sanitized JSON parsing
nlohmann::json parse_json_with_specials(const std::string& json_path);

} // namespace profiles
} // namespace agx


