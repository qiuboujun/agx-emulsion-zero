// SPDX-License-Identifier: MIT

#include "profile_io.hpp"

#include <fstream>
#include <stdexcept>
#include <limits>
#include <cmath>

namespace agx {
namespace profiles {

using json = nlohmann::json;

static nc::NdArray<float> json_to_ndarray_2d(const json& j) {
    if (!j.is_array()) throw std::runtime_error("Expected array for ndarray");
    const std::size_t rows = j.size();
    if (rows == 0) return nc::NdArray<float>(0); // empty
    if (!j[0].is_array()) {
        // treat as 1D
        nc::NdArray<float> arr(1, rows);
        for (std::size_t i=0;i<rows;++i) {
            if (j[i].is_null()) arr[i] = std::numeric_limits<float>::quiet_NaN();
            else                arr[i] = static_cast<float>(j[i].get<double>());
        }
        return arr.transpose(); // column vector
    }
    const std::size_t cols = j[0].size();
    nc::NdArray<float> arr(rows, cols);
    for (std::size_t i=0;i<rows;++i) {
        if (j[i].size() != cols) throw std::runtime_error("Jagged nested array");
        for (std::size_t k=0;k<cols;++k) {
            if (j[i][k].is_null()) arr(i,k) = std::numeric_limits<float>::quiet_NaN();
            else                   arr(i,k) = static_cast<float>(j[i][k].get<double>());
        }
    }
    return arr;
}

static json ndarray_to_json(const nc::NdArray<float>& arr) {
    json j = json::array();
    if (arr.size() == 0) return j;
    for (nc::uint32 i=0;i<arr.shape().rows;++i) {
        json row = json::array();
        for (nc::uint32 k=0;k<arr.shape().cols;++k) {
            float v = arr(i,k);
            if (std::isfinite(v)) row.push_back(v);
            else                  row.push_back(nullptr);
        }
        j.push_back(row);
    }
    return j;
}

static std::string slurp(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("Cannot open: " + path);
    return std::string((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
}

static std::string sanitize_json_specials(const std::string& raw) {
    std::string out; out.reserve(raw.size());
    bool in_string = false, escape = false;
    auto match = [&](const char* tok, std::size_t pos){ for (std::size_t i=0; tok[i]; ++i){ if (pos+i>=raw.size()||raw[pos+i]!=tok[i]) return false; } return true; };
    for (std::size_t i=0; i<raw.size(); ++i) {
        char c = raw[i];
        if (in_string) {
            out.push_back(c);
            if (escape) escape = false; else if (c=='\\') escape = true; else if (c=='"') in_string=false;
            continue;
        }
        if (c=='"') { in_string=true; out.push_back(c); continue; }
        if (match("NaN", i)) { out += "null"; i+=2; continue; }
        if (match("Infinity", i)) { out += "null"; i+=7; continue; }
        if (match("-Infinity", i)) { out += "null"; i+=8; continue; }
        out.push_back(c);
    }
    return out;
}

Profile ProfileIO::load_from_file(const std::string& json_path) {
    const std::string raw = slurp(json_path);
    const std::string sanitized = sanitize_json_specials(raw);
    json j = json::parse(sanitized);
    Profile p;
    p.info.stock = j["info"]["stock"].get<std::string>();
    const auto& d = j["data"];
    p.data.log_sensitivity       = json_to_ndarray_2d(d.at("log_sensitivity"));
    p.data.density_curves        = json_to_ndarray_2d(d.at("density_curves"));
    p.data.density_curves_layers = json_to_ndarray_2d(d.at("density_curves_layers"));
    p.data.dye_density           = json_to_ndarray_2d(d.at("dye_density"));
    p.data.log_exposure          = json_to_ndarray_2d(d.at("log_exposure"));
    p.data.wavelengths           = json_to_ndarray_2d(d.at("wavelengths"));
    return p;
}

void ProfileIO::save_to_file(const Profile& profile, const std::string& json_path) {
    json j;
    j["info"]["stock"] = profile.info.stock;
    j["data"]["log_sensitivity"]       = ndarray_to_json(profile.data.log_sensitivity);
    j["data"]["density_curves"]        = ndarray_to_json(profile.data.density_curves);
    j["data"]["density_curves_layers"] = ndarray_to_json(profile.data.density_curves_layers);
    j["data"]["dye_density"]           = ndarray_to_json(profile.data.dye_density);
    j["data"]["log_exposure"]          = ndarray_to_json(profile.data.log_exposure);
    j["data"]["wavelengths"]           = ndarray_to_json(profile.data.wavelengths);
    std::ofstream f(json_path);
    if (!f) throw std::runtime_error("Cannot open file for write: " + json_path);
    f << j.dump(4);
}

bool arrays_equal(const nc::NdArray<float>& a, const nc::NdArray<float>& b) {
    if (a.shape() != b.shape()) return false;
    for (nc::uint32 i=0;i<a.shape().rows;++i)
        for (nc::uint32 k=0;k<a.shape().cols;++k)
            if (a(i,k) != b(i,k)) return false;
    return true;
}

} // namespace profiles
} // namespace agx


