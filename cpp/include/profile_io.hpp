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
};

struct ProfileData {
    nc::NdArray<float> log_sensitivity;        // [N,3]
    nc::NdArray<float> density_curves;         // [M,3]
    nc::NdArray<float> density_curves_layers;  // [L,3]
    nc::NdArray<float> dye_density;            // [K,5]
    nc::NdArray<float> log_exposure;           // [M,1] or [M]
    nc::NdArray<float> wavelengths;            // [K,1] or [K]
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

} // namespace profiles
} // namespace agx


