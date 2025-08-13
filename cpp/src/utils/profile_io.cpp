// SPDX-License-Identifier: MIT

#include "profile_io.hpp"

#include <fstream>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <regex>
#include <iostream> // Added for debug output

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
            else if (j[i].is_string()) {
                std::string s = j[i].get<std::string>();
                if (s == "NaN") arr[i] = std::numeric_limits<float>::quiet_NaN();
                else if (s == "Infinity") arr[i] = std::numeric_limits<float>::infinity();
                else if (s == "-Infinity") arr[i] = -std::numeric_limits<float>::infinity();
                else throw std::runtime_error("Unknown string value: " + s);
            }
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
            else if (j[i][k].is_string()) {
                std::string s = j[i][k].get<std::string>();
                if (s == "NaN") arr(i,k) = std::numeric_limits<float>::quiet_NaN();
                else if (s == "Infinity") arr(i,k) = std::numeric_limits<float>::infinity();
                else if (s == "-Infinity") arr(i,k) = -std::numeric_limits<float>::infinity();
                else throw std::runtime_error("Unknown string value: " + s);
            }
            else                   arr(i,k) = static_cast<float>(j[i][k].get<double>());
        }
    }
    return arr;
}

static nc::NdArray<float> json_to_ndarray_3d(const json& j) {
    // Convert JSON 3D array to NumCpp 2D array by flattening the first two dimensions
    if (!j.is_array()) {
        throw std::runtime_error("Expected array for 3D array conversion");
    }
    
    const size_t dim1 = j.size();
    if (dim1 == 0) {
        return nc::NdArray<float>(0, 0);
    }
    
    if (!j[0].is_array()) {
        throw std::runtime_error("Expected 3D array for 3D array conversion");
    }
    
    const size_t dim2 = j[0].size();
    if (dim2 == 0) {
        return nc::NdArray<float>(0, 0);
    }
    
    if (!j[0][0].is_array()) {
        throw std::runtime_error("Expected 3D array for 3D array conversion");
    }
    
    const size_t dim3 = j[0][0].size();
    if (dim3 == 0) {
        return nc::NdArray<float>(0, 0);
    }
    
    // Flatten first two dimensions: (dim1, dim2, dim3) -> (dim1*dim2, dim3)
    const size_t rows = dim1 * dim2;
    const size_t cols = dim3;
    nc::NdArray<float> arr(rows, cols);
    
    for (size_t i = 0; i < dim1; ++i) {
        if (!j[i].is_array() || j[i].size() != dim2) {
            throw std::runtime_error("Inconsistent second dimension in 3D array");
        }
        for (size_t k = 0; k < dim2; ++k) {
            if (!j[i][k].is_array() || j[i][k].size() != dim3) {
                throw std::runtime_error("Inconsistent third dimension in 3D array");
            }
            const size_t row_idx = i * dim2 + k;
            for (size_t l = 0; l < dim3; ++l) {
                if (j[i][k][l].is_null()) {
                    arr(row_idx, l) = std::numeric_limits<float>::quiet_NaN();
                } else if (j[i][k][l].is_string()) {
                    std::string s = j[i][k][l].get<std::string>();
                    if (s == "NaN") arr(row_idx, l) = std::numeric_limits<float>::quiet_NaN();
                    else if (s == "Infinity") arr(row_idx, l) = std::numeric_limits<float>::infinity();
                    else if (s == "-Infinity") arr(row_idx, l) = -std::numeric_limits<float>::infinity();
                    else throw std::runtime_error("Unknown string value: " + s);
                } else {
                    arr(row_idx, l) = static_cast<float>(j[i][k][l].get<double>());
                }
            }
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
            if (std::isnan(v)) row.push_back("NaN");
            else if (std::isinf(v)) {
                if (std::signbit(v)) row.push_back("-Infinity");
                else row.push_back("Infinity");
            }
            else row.push_back(v);
        }
        j.push_back(row);
    }
    return j;
}

static json ndarray_to_json_3d_layers(const nc::NdArray<float>& flattened_layers,
                                      const nc::NdArray<float>& density_curves_ref) {
    // Reconstruct a 3D array with shape [dim1, dim2, dim3] where
    // dim1 = number of rows in density_curves_ref
    // dim2 = number of layers (flattened_layers.rows / dim1)
    // dim3 = flattened_layers.cols
    if (flattened_layers.size() == 0) {
        return json::array();
    }
    const nc::uint32 dim1 = density_curves_ref.shape().rows;
    const nc::uint32 total_rows = flattened_layers.shape().rows;
    const nc::uint32 dim2 = (dim1 == 0) ? 0u : (total_rows / dim1);
    const nc::uint32 dim3 = flattened_layers.shape().cols;

    // Fallback: if shape inference fails, return as 2D array
    if (dim1 == 0 || dim2 == 0 || dim1 * dim2 != total_rows) {
        return ndarray_to_json(flattened_layers);
    }

    json top = json::array();
    for (nc::uint32 i = 0; i < dim1; ++i) {
        json mid = json::array();
        for (nc::uint32 k = 0; k < dim2; ++k) {
            const nc::uint32 row_idx = i * dim2 + k;
            json inner = json::array();
            for (nc::uint32 l = 0; l < dim3; ++l) {
                float v = flattened_layers(row_idx, l);
                if (std::isnan(v)) inner.push_back("NaN");
                else if (std::isinf(v)) inner.push_back(std::signbit(v) ? "-Infinity" : "Infinity");
                else inner.push_back(v);
            }
            mid.push_back(inner);
        }
        top.push_back(mid);
    }
    return top;
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
        if (match("NaN", i)) { out += "\"NaN\""; i+=2; continue; }
        if (match("Infinity", i)) { out += "\"Infinity\""; i+=7; continue; }
        if (match("-Infinity", i)) { out += "\"-Infinity\""; i+=8; continue; }
        out.push_back(c);
    }
    return out;
}

static std::string post_process_json_for_python(const std::string& json_str) {
    // Safely unquote exact string values "NaN", "Infinity", "-Infinity" while
    // leaving any other strings intact. We scan the JSON text and when we see a
    // string token, we peek its full content; if it is exactly one of the
    // special tokens, we emit the bare token, otherwise we emit the original
    // quoted string verbatim.
    std::string out;
    out.reserve(json_str.size());

    for (std::size_t i = 0; i < json_str.size(); ++i) {
        char c = json_str[i];
        if (c != '"') {
            out.push_back(c);
            continue;
        }

        // We are at the start of a JSON string; capture its content while
        // handling escape sequences to locate the matching closing quote.
        std::size_t j = i + 1;
        bool escape = false;
        bool closed = false;
        std::string content;
        content.reserve(8);
        for (; j < json_str.size(); ++j) {
            char cj = json_str[j];
            if (escape) {
                content.push_back(cj);
                escape = false;
                continue;
            }
            if (cj == '\\') {
                content.push_back(cj);
                escape = true;
                continue;
            }
            if (cj == '"') {
                closed = true;
                break;
            }
            content.push_back(cj);
        }

        if (!closed) {
            // Malformed JSON (should not happen after dump); just copy the char
            out.push_back(c);
            continue;
        }

        // If the content is exactly one of the special tokens, write it
        // without quotes; otherwise, copy the original quoted string.
        if (content == "NaN" || content == "Infinity" || content == "-Infinity") {
            out += content;
        } else {
            out.append(json_str, i, (j - i + 1));
        }
        i = j; // advance to the closing quote we just handled
    }

    return out;
}

Profile ProfileIO::load_from_file(const std::string& json_path) {
    std::cout << "ProfileIO::load_from_file: Reading from " << json_path << std::endl;
    
    const std::string raw = slurp(json_path);
    std::cout << "ProfileIO::load_from_file: Raw file size: " << raw.size() << " bytes" << std::endl;
    
    const std::string sanitized = sanitize_json_specials(raw);
    std::cout << "ProfileIO::load_from_file: Sanitized JSON size: " << sanitized.size() << " bytes" << std::endl;
    
    std::cout << "ProfileIO::load_from_file: About to parse JSON..." << std::endl;
    json j = json::parse(sanitized);
    std::cout << "ProfileIO::load_from_file: JSON parsed successfully" << std::endl;
    
    std::cout << "ProfileIO::load_from_file: JSON keys: ";
    for (auto it = j.begin(); it != j.end(); ++it) {
        std::cout << it.key() << " ";
    }
    std::cout << std::endl;
    
    Profile p;
    std::cout << "ProfileIO::load_from_file: Reading info section..." << std::endl;
    const auto& info = j["info"];
    p.info.stock = info["stock"].get<std::string>();
    p.info.name = info["name"].get<std::string>();
    p.info.type = info["type"].get<std::string>();
    p.info.color = info["color"].get<bool>();
    p.info.densitometer = info["densitometer"].get<std::string>();
    p.info.log_sensitivity_density_over_min = info["log_sensitivity_density_over_min"].get<float>();
    p.info.reference_illuminant = info["reference_illuminant"].get<std::string>();
    p.info.viewing_illuminant = info["viewing_illuminant"].get<std::string>();
    
    // Parse density_midscale_neutral which may be a scalar or a 3-array in profiles
    const auto& dmn = info["density_midscale_neutral"];
    if (dmn.is_array()) {
        for (size_t i = 0; i < 3; ++i) {
            p.info.density_midscale_neutral[i] = dmn[i].get<float>();
        }
    } else if (dmn.is_number_float() || dmn.is_number_integer()) {
        float v = dmn.get<float>();
        p.info.density_midscale_neutral = {v, v, v};
    } else if (dmn.is_string()) {
        // Allow string tokens like "NaN" although unexpected here; treat as 0
        p.info.density_midscale_neutral = {0.0f, 0.0f, 0.0f};
    } else {
        p.info.density_midscale_neutral = {0.0f, 0.0f, 0.0f};
    }
    
    std::cout << "ProfileIO::load_from_file: Stock: " << p.info.stock << std::endl;
    
    std::cout << "ProfileIO::load_from_file: Reading data section..." << std::endl;
    const auto& d = j["data"];
    std::cout << "ProfileIO::load_from_file: Data keys: ";
    for (auto it = d.begin(); it != d.end(); ++it) {
        std::cout << it.key() << " ";
    }
    std::cout << std::endl;
    
    std::cout << "ProfileIO::load_from_file: Converting log_sensitivity..." << std::endl;
    p.data.log_sensitivity       = json_to_ndarray_2d(d.at("log_sensitivity"));
    std::cout << "ProfileIO::load_from_file: Converting density_curves..." << std::endl;
    p.data.density_curves        = json_to_ndarray_2d(d.at("density_curves"));
    std::cout << "ProfileIO::load_from_file: Converting density_curves_layers..." << std::endl;
    p.data.density_curves_layers = json_to_ndarray_3d(d.at("density_curves_layers"));
    std::cout << "ProfileIO::load_from_file: Converting dye_density..." << std::endl;
    p.data.dye_density           = json_to_ndarray_2d(d.at("dye_density"));
    std::cout << "ProfileIO::load_from_file: Converting log_exposure..." << std::endl;
    p.data.log_exposure          = json_to_ndarray_2d(d.at("log_exposure"));
    std::cout << "ProfileIO::load_from_file: Converting wavelengths..." << std::endl;
    p.data.wavelengths           = json_to_ndarray_2d(d.at("wavelengths"));
    
    // Optional tune parameters
    try {
        const auto& tune = d.at("tune");
        if (tune.contains("gamma_factor")) {
            const auto& gf = tune.at("gamma_factor");
            if (gf.is_array() && gf.size()>=3) {
                p.data.gamma_factor[0] = gf[0].get<float>();
                p.data.gamma_factor[1] = gf[1].get<float>();
                p.data.gamma_factor[2] = gf[2].get<float>();
            } else if (gf.is_number_float() || gf.is_number_integer()) {
                float g = gf.get<float>();
                p.data.gamma_factor = {g,g,g};
            }
        }
        if (tune.contains("dye_density_min_factor")) {
            p.data.dye_density_min_factor = tune.at("dye_density_min_factor").get<float>();
        }
    } catch (...) {
        // keep defaults
    }
    std::cout << "ProfileIO::load_from_file: All conversions completed successfully" << std::endl;
    return p;
}

void ProfileIO::save_to_file(const Profile& profile, const std::string& json_path) {
    json j;
    j["info"]["stock"] = profile.info.stock;
    j["info"]["name"] = profile.info.name;
    j["info"]["type"] = profile.info.type;
    j["info"]["color"] = profile.info.color;
    j["info"]["densitometer"] = profile.info.densitometer;
    j["info"]["log_sensitivity_density_over_min"] = profile.info.log_sensitivity_density_over_min;
    j["info"]["reference_illuminant"] = profile.info.reference_illuminant;
    j["info"]["viewing_illuminant"] = profile.info.viewing_illuminant;
    
    // Save density_midscale_neutral array
    j["info"]["density_midscale_neutral"] = {
        profile.info.density_midscale_neutral[0],
        profile.info.density_midscale_neutral[1],
        profile.info.density_midscale_neutral[2]
    };
    j["data"]["log_sensitivity"]       = ndarray_to_json(profile.data.log_sensitivity);
    j["data"]["density_curves"]        = ndarray_to_json(profile.data.density_curves);
    j["data"]["density_curves_layers"] = ndarray_to_json_3d_layers(
        profile.data.density_curves_layers,
        profile.data.density_curves
    );
    j["data"]["dye_density"]           = ndarray_to_json(profile.data.dye_density);
    j["data"]["log_exposure"]          = ndarray_to_json(profile.data.log_exposure);
    j["data"]["wavelengths"]           = ndarray_to_json(profile.data.wavelengths);
    
    // Dump JSON and then post-process to match Python's format
    std::string json_str = j.dump(4);
    std::string python_compatible = post_process_json_for_python(json_str);
    
    std::ofstream f(json_path);
    if (!f) throw std::runtime_error("Cannot open file for write: " + json_path);
    f << python_compatible;
}

// Helper function for other parts of the codebase to use sanitized JSON parsing
json parse_json_with_specials(const std::string& json_path) {
    const std::string raw = slurp(json_path);
    const std::string sanitized = sanitize_json_specials(raw);
    return json::parse(sanitized);
}

bool arrays_equal(const nc::NdArray<float>& a, const nc::NdArray<float>& b) {
    if (a.shape() != b.shape()) return false;
    for (nc::uint32 i=0;i<a.shape().rows;++i)
        for (nc::uint32 k=0;k<a.shape().cols;++k) {
            float va = a(i,k), vb = b(i,k);
            if (std::isnan(va) && std::isnan(vb)) continue; // NaN == NaN
            if (std::isinf(va) && std::isinf(vb) && std::signbit(va) == std::signbit(vb)) continue; // Inf == Inf
            if (va != vb) return false;
        }
    return true;
}

} // namespace profiles
} // namespace agx


