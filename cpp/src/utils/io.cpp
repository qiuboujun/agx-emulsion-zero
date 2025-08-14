#include "io.hpp"
#include "config.hpp" // Assuming a config.hpp holds SPECTRAL_SHAPE, LOG_EXPOSURE, and data paths
#include "scipy.hpp"  // Your custom scipy wrappers
#include "nlohmann/json.hpp" // For JSON parsing
#include "profile_io.hpp" // For sanitized JSON parsing
#include <fstream>
#include <stdexcept>
#include <algorithm> // For std::replace
#include <vector>
#include <iostream> // For debug output
#include <cmath> // For std::min
#include <limits>
#include <filesystem>
#include <dlfcn.h>

// Helper function to get the root data path.
namespace agx {
namespace utils {
std::string get_data_path() {
    // Try to resolve relative to the plugin binary location if available
    // This lets the OFX bundle carry its own resources.
    Dl_info dl_info{};
    if (dladdr((void*)&get_data_path, &dl_info) != 0 && dl_info.dli_fname) {
        try {
            std::filesystem::path lib_path(dl_info.dli_fname);
            std::filesystem::path base_dir = lib_path.parent_path();
            // Prefer placing resources alongside the plugin binary under
            // base_dir/agx_emulsion/data/...
            std::filesystem::path probe = base_dir / "agx_emulsion" / "data";
            if (std::filesystem::exists(probe)) {
                return (base_dir.string() + "/");
            }
        } catch (...) {
            // Fall back below
        }
    }
    // Fallback: repository root (development)
    return std::string(AGX_SOURCE_DIR) + "/";
}
} // namespace utils
} // namespace agx

namespace agx {
namespace utils {

// Recursively convert non-finite numbers in JSON to string tokens "NaN"/"Infinity"/"-Infinity"
static nlohmann::json convert_nonfinite_to_strings(const nlohmann::json& j) {
    using json = nlohmann::json;
    if (j.is_array()) {
        json out = json::array();
        for (const auto& el : j) {
            out.push_back(convert_nonfinite_to_strings(el));
        }
        return out;
    } else if (j.is_object()) {
        json out = json::object();
        for (auto it = j.begin(); it != j.end(); ++it) {
            out[it.key()] = convert_nonfinite_to_strings(it.value());
        }
        return out;
    } else if (j.is_number_float()) {
        double v = j.get<double>();
        if (std::isnan(v)) return json("NaN");
        if (std::isinf(v)) return json(std::signbit(v) ? "-Infinity" : "Infinity");
        return j;
    } else {
        return j;
    }
}

// Unquote exact string values "NaN"/"Infinity"/"-Infinity" in a JSON string
static std::string unquote_python_specials(const std::string& json_str) {
    std::string out;
    out.reserve(json_str.size());
    for (std::size_t i = 0; i < json_str.size(); ++i) {
        char c = json_str[i];
        if (c != '"') { out.push_back(c); continue; }
        std::size_t j = i + 1; bool escape = false; bool closed = false; std::string content; content.reserve(8);
        for (; j < json_str.size(); ++j) {
            char cj = json_str[j];
            if (escape) { content.push_back(cj); escape = false; continue; }
            if (cj == '\\') { content.push_back(cj); escape = true; continue; }
            if (cj == '"') { closed = true; break; }
            content.push_back(cj);
        }
        if (!closed) { out.push_back(c); continue; }
        if (content == "NaN" || content == "Infinity" || content == "-Infinity") { out += content; }
        else { out.append(json_str, i, (j - i + 1)); }
        i = j;
    }
    return out;
}

nc::NdArray<float> interpolate_to_common_axis(const nc::NdArray<float>& data, const nc::NdArray<float>& new_x, bool extrapolate, const std::string& method) {
    // ---------------------------------------------------------------------
    // 1. Split first/second rows → x, y (flattened to 1D)
    // ---------------------------------------------------------------------
    auto x = data.row(0).flatten();
    auto y = data.row(1).flatten();

    // ---------------------------------------------------------------------
    // 2. Sort by x (NumCpp’s argsort returns indices like NumPy)
    // ---------------------------------------------------------------------
    auto idx_sorted = nc::argsort(x);
    x = x[idx_sorted];
    y = y[idx_sorted];

    // ---------------------------------------------------------------------
    // 3. Keep first occurrence of duplicate x’s (unique w/ return_index=True)
    // ---------------------------------------------------------------------
    std::vector<nc::uint32> uniq_idx;
    uniq_idx.reserve(x.size());
    uniq_idx.emplace_back(0);
    for (nc::uint32 i = 1; i < x.size(); ++i)
        if (x[i] != x[i - 1]) uniq_idx.emplace_back(i);

    auto idx_unique = nc::NdArray<nc::uint32>(uniq_idx);
    x = x[idx_unique];
    y = y[idx_unique];

    // ---------------------------------------------------------------------
    // 4. Promote to double for the spline classes (ensure new_x is flattened)
    // ---------------------------------------------------------------------
    auto xd = x.astype<double>();
    auto yd = y.astype<double>();
    
    // Force 1D array by taking only the first row if it's 2D
    nc::NdArray<float> new_x_1d;
    if (new_x.shape().rows > 1 && new_x.shape().cols > 1) {
        // It's a 2D matrix, take only the first row
        new_x_1d = new_x.row(0);
    } else {
        // It's already 1D or a single row/column
        new_x_1d = new_x.flatten();
    }
    
    auto new_xd = new_x_1d.astype<double>();

    // ---------------------------------------------------------------------
    // 5. Select interpolator based on method parameter
    // ---------------------------------------------------------------------
    std::function<nc::NdArray<double>(const nc::NdArray<double>&)> interp;

    if (method == "linear") {
        // Use linear interpolation matching Python's np.interp behavior
        interp = [xd, yd](const nc::NdArray<double>& q){
            nc::NdArray<double> result = nc::zeros<double>({1, q.size()}).flatten();
            for (std::size_t i = 0; i < q.size(); ++i) {
                double xq = q[i];
                
                // Handle edge cases like np.interp
                if (xq <= xd[0]) {
                    result[i] = yd[0];
                } else if (xq >= xd[xd.size() - 1]) {
                    result[i] = yd[yd.size() - 1];
                } else {
                    // Find the segment
                    std::size_t j = 0;
                    for (; j < xd.size() - 1; ++j) {
                        if (xq <= xd[j + 1]) break;
                    }
                    
                    // Linear interpolation
                    double x0 = xd[j], x1 = xd[j + 1];
                    double y0 = yd[j], y1 = yd[j + 1];
                    double t = (xq - x0) / (x1 - x0);
                    result[i] = y0 + t * (y1 - y0);
                }
            }
            return result;
        };
    } else if (method == "akima") {
        // Use Akima interpolation matching Python's scipy.interpolate.Akima1DInterpolator
        try {
            scipy::interpolate::Akima1DInterpolator akima_interp(xd, yd, extrapolate);
            interp = [akima_interp](const nc::NdArray<double>& q) {
                return akima_interp(q);
            };
        } catch (const std::exception& e) {
            // Fallback to linear if Akima fails
            std::cerr << "Warning: Akima interpolation failed, falling back to linear: " << e.what() << std::endl;
            interp = [xd, yd](const nc::NdArray<double>& q){
                nc::NdArray<double> result = nc::zeros<double>({1, q.size()}).flatten();
                for (std::size_t i = 0; i < q.size(); ++i) {
                    double xq = q[i];
                    
                    // Handle edge cases like np.interp
                    if (xq <= xd[0]) {
                        result[i] = yd[0];
                    } else if (xq >= xd[xd.size() - 1]) {
                        result[i] = yd[yd.size() - 1];
                    } else {
                        // Find the segment
                        std::size_t j = 0;
                        for (; j < xd.size() - 1; ++j) {
                            if (xq <= xd[j + 1]) break;
                        }
                        
                        // Linear interpolation
                        double x0 = xd[j], x1 = xd[j + 1];
                        double y0 = yd[j], y1 = yd[j + 1];
                        double t = (xq - x0) / (x1 - x0);
                        result[i] = y0 + t * (y1 - y0);
                    }
                }
                return result;
            };
        }
    } else if (method == "cubic") {
        // Use cubic spline interpolation matching Python's scipy.interpolate.CubicSpline
        try {
            scipy::interpolate::CubicSpline cubic_interp(xd, yd, 
                scipy::interpolate::CubicSpline::natural(), 
                scipy::interpolate::CubicSpline::natural(), 
                extrapolate);
            interp = [cubic_interp](const nc::NdArray<double>& q) {
                return cubic_interp(q);
            };
        } catch (const std::exception& e) {
            // Fallback to linear if cubic fails
            std::cerr << "Warning: Cubic spline interpolation failed, falling back to linear: " << e.what() << std::endl;
            interp = [xd, yd](const nc::NdArray<double>& q){
                nc::NdArray<double> result = nc::zeros<double>({1, q.size()}).flatten();
                for (std::size_t i = 0; i < q.size(); ++i) {
                    double xq = q[i];
                    
                    // Handle edge cases like np.interp
                    if (xq <= xd[0]) {
                        result[i] = yd[0];
                    } else if (xq >= xd[xd.size() - 1]) {
                        result[i] = yd[yd.size() - 1];
                    } else {
                        // Find the segment
                        std::size_t j = 0;
                        for (; j < xd.size() - 1; ++j) {
                            if (xq <= xd[j + 1]) break;
                        }
                        
                        // Linear interpolation
                        double x0 = xd[j], x1 = xd[j + 1];
                        double y0 = yd[j], y1 = yd[j + 1];
                        double t = (xq - x0) / (x1 - x0);
                        result[i] = y0 + t * (y1 - y0);
                    }
                }
                return result;
            };
        }
    } else {
        // For other methods, use linear interpolation as fallback
        std::cerr << "Warning: Unknown interpolation method '" << method << "', using linear" << std::endl;
        interp = [xd, yd](const nc::NdArray<double>& q){
            nc::NdArray<double> result = nc::zeros<double>({1, q.size()}).flatten();
            for (std::size_t i = 0; i < q.size(); ++i) {
                double xq = q[i];
                
                // Handle edge cases like np.interp
                if (xq <= xd[0]) {
                    result[i] = yd[0];
                } else if (xq >= xd[xd.size() - 1]) {
                    result[i] = yd[yd.size() - 1];
                } else {
                    // Find the segment
                    std::size_t j = 0;
                    for (; j < xd.size() - 1; ++j) {
                        if (xq <= xd[j + 1]) break;
                    }
                    
                    // Linear interpolation
                    double x0 = xd[j], x1 = xd[j + 1];
                    double y0 = yd[j], y1 = yd[j + 1];
                    double t = (xq - x0) / (x1 - x0);
                    result[i] = y0 + t * (y1 - y0);
                }
            }
            return result;
        };
    }

    // ---------------------------------------------------------------------
    // 6. Evaluate & cast back to float
    // ---------------------------------------------------------------------
    auto result = interp(new_xd);
    return result.astype<float>();
}

nc::NdArray<float> load_csv(const std::string& datapkg, const std::string& filename) {
    std::string base_path = get_data_path();
    std::string datapkg_path = datapkg;
    std::replace(datapkg_path.begin(), datapkg_path.end(), '.', '/');
    
    // Construct path properly, avoiding double slashes
    std::string full_path;
    if (base_path.back() == '/') {
        full_path = base_path + datapkg_path + "/" + filename;
    } else {
        full_path = base_path + "/" + datapkg_path + "/" + filename;
    }



    // Load CSV manually to handle the format correctly
    std::ifstream file(full_path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open CSV file: " + full_path);
    }

    std::vector<float> x_values, y_values;
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        
        // Parse comma-separated values
        size_t comma_pos = line.find(',');
        if (comma_pos != std::string::npos) {
            try {
                float x = std::stof(line.substr(0, comma_pos));
                float y = std::stof(line.substr(comma_pos + 1));
                x_values.push_back(x);
                y_values.push_back(y);
            } catch (const std::exception& e) {
                // Skip invalid lines
                continue;
            }
        }
    }

    // Create 2xN array with x values in first row, y values in second row
    auto result = nc::zeros<float>(2, x_values.size());
    for (size_t i = 0; i < x_values.size(); ++i) {
        result(0, i) = x_values[i];
        result(1, i) = y_values[i];
    }

    return result;
}

AgxEmulsionData load_agx_emulsion_data(
    const std::string& stock, const std::string& log_sensitivity_donor,
    const std::string& density_curves_donor, const std::string& dye_density_cmy_donor,
    const std::string& dye_density_min_mid_donor, const std::string& type, bool color)
{
    AgxEmulsionData result;
    result.wavelengths = agx::config::SPECTRAL_SHAPE.wavelengths;
    result.log_exposure = agx::config::LOG_EXPOSURE;

    std::string maindatapkg;
    if (color && type == "negative") maindatapkg = "agx_emulsion.data.film.negative";
    else if (color && type == "positive") maindatapkg = "agx_emulsion.data.film.positive";
    else if (color && type == "paper") maindatapkg = "agx_emulsion.data.paper";

    // --- Load log sensitivity ---
    std::string sens_datapkg = maindatapkg + "." + (log_sensitivity_donor.empty() ? stock : log_sensitivity_donor);
    result.log_sensitivity = nc::zeros<float>(result.wavelengths.size(), 3);
    const char* sens_channels[] = {"r", "g", "b"};
    for (int i = 0; i < 3; ++i) {
        auto data = load_csv(sens_datapkg, "log_sensitivity_" + std::string(sens_channels[i]) + ".csv");
        auto interpolated = interpolate_to_common_axis(data, result.wavelengths);
        // Assign each interpolated value to the correct row in column i
        for (size_t j = 0; j < result.wavelengths.size(); ++j) {
            result.log_sensitivity(j, i) = interpolated[j];
        }
    }

    // --- Load density curves ---
    std::string dens_datapkg = maindatapkg + "." + (density_curves_donor.empty() ? stock : density_curves_donor);
    auto dh_curve_r = load_csv(dens_datapkg, "density_curve_r.csv");
    auto dh_curve_g = load_csv(dens_datapkg, "density_curve_g.csv");
    auto dh_curve_b = load_csv(dens_datapkg, "density_curve_b.csv");
    float log_exposure_shift = (nc::max(dh_curve_g(0, nc::Slice()))[0] + nc::min(dh_curve_g(0, nc::Slice()))[0]) / 2.0f;
    
    auto p_denc_r = interpolate_to_common_axis(dh_curve_r, result.log_exposure + log_exposure_shift);
    auto p_denc_g = interpolate_to_common_axis(dh_curve_g, result.log_exposure + log_exposure_shift);
    auto p_denc_b = interpolate_to_common_axis(dh_curve_b, result.log_exposure + log_exposure_shift);
    
    // Create density_curves as (N, 3) array to match Python format
    // Python: np.array([p_denc_r, p_denc_g, p_denc_b]).transpose()
    result.density_curves = nc::zeros<float>(p_denc_r.size(), 3);
    for (size_t i = 0; i < p_denc_r.size(); ++i) {
        result.density_curves(i, 0) = p_denc_r[i];
        result.density_curves(i, 1) = p_denc_g[i];
        result.density_curves(i, 2) = p_denc_b[i];
    }

    // --- Load dye density ---
    std::string dye_cmy_datapkg = maindatapkg + "." + (dye_density_cmy_donor.empty() ? stock : dye_density_cmy_donor);
    result.dye_density = nc::zeros<float>(result.wavelengths.size(), 5);
    const char* dye_channels_cmy[] = {"c", "m", "y"};
    for (int i = 0; i < 3; ++i) {
        auto data = load_csv(dye_cmy_datapkg, "dye_density_" + std::string(dye_channels_cmy[i]) + ".csv");
        auto interpolated = interpolate_to_common_axis(data, result.wavelengths);
        // Assign each interpolated value to the correct row in column i
        for (size_t j = 0; j < result.wavelengths.size(); ++j) {
            result.dye_density(j, i) = interpolated[j];
        }
    }

    if (type == "negative") {
        std::string dye_min_mid_datapkg = maindatapkg + "." + (dye_density_min_mid_donor.empty() ? stock : dye_density_min_mid_donor);
        const char* dye_channels_min_mid[] = {"min", "mid"};
        for (int i = 0; i < 2; ++i) {
            auto data = load_csv(dye_min_mid_datapkg, "dye_density_" + std::string(dye_channels_min_mid[i]) + ".csv");
            auto interpolated = interpolate_to_common_axis(data, result.wavelengths);
            // Assign each interpolated value to the correct row in column i+3
            for (size_t j = 0; j < result.wavelengths.size(); ++j) {
                result.dye_density(j, i + 3) = interpolated[j];
            }
        }
    }

    return result;
}

nc::NdArray<float> load_densitometer_data(const std::string& type) {
    auto wavelengths = agx::config::SPECTRAL_SHAPE.wavelengths;
    auto responsivities = nc::zeros<float>(wavelengths.size(), 3);
    const char* channels[] = {"r", "g", "b"};
    for (int i = 0; i < 3; ++i) {
        std::string datapkg = "agx_emulsion.data.densitometer." + type;
        std::string filename = "responsivity_" + std::string(channels[i]) + ".csv";
        auto data = load_csv(datapkg, filename);
        
        auto interpolated = interpolate_to_common_axis(data, wavelengths, false, "linear");
        
        // Assign to the column using a different approach
        for (size_t j = 0; j < interpolated.size(); ++j) {
            responsivities(j, i) = interpolated[j];
        }
    }
    
    // FIX 1: Use 0.0f to avoid type mismatch with float array
    responsivities = nc::where(responsivities < 0.0f, 0.0f, responsivities);
    
    // Sum down each column to get one scalar per channel, then reshape to (1,3)
    auto channelSums = nc::nansum(responsivities, nc::Axis::ROW);
    responsivities /= channelSums.reshape(1, responsivities.shape().cols);

    return responsivities;
}

void save_ymc_filter_values(const FilterValues& ymc_filters) {
    std::string path = get_data_path() + "agx_emulsion/data/profiles/enlarger_neutral_ymc_filters.json";
    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open JSON file for writing: " + path);
    
    // Convert any non-finite numbers to Python-compatible special tokens and unquote them
    nlohmann::json j = convert_nonfinite_to_strings(ymc_filters);
    std::string dumped = j.dump(4);
    std::string python_compatible = unquote_python_specials(dumped);
    file << python_compatible;
}

FilterValues read_neutral_ymc_filter_values() {
    std::string path = get_data_path() + "agx_emulsion/data/profiles/enlarger_neutral_ymc_filters.json";
    // Use sanitized JSON parsing to handle NaN/Infinity/-Infinity tokens
    return agx::profiles::parse_json_with_specials(path);
}

nc::NdArray<float> load_dichroic_filters(const nc::NdArray<float>& wavelengths, const std::string& brand) {
    // Ensure wavelengths is 1D
    auto wl_flat = wavelengths.flatten();
    auto filters = nc::zeros<float>(wl_flat.size(), 3);
    const char* channels[] = {"y", "m", "c"};
    for (int i = 0; i < 3; ++i) {
        std::string datapkg = "agx_emulsion.data.filters.dichroics." + brand;
        std::string filename = "filter_" + std::string(channels[i]) + ".csv";
        auto data = load_csv(datapkg, filename);
        
        // Scale the data by dividing the second column (y-values) by 100, like Python does
        auto scaled_data = data.copy();
        for (size_t i = 0; i < scaled_data.shape().cols; ++i) {
            scaled_data(1, i) /= 100.0f;
        }
        
        auto interpolated = interpolate_to_common_axis(scaled_data, wl_flat, false, "akima");
        
        // Assign each interpolated value to the correct row in column i
        for (size_t j = 0; j < wl_flat.size(); ++j) {
            filters(j, i) = interpolated[j];
        }
    }
    return filters;
}

nc::NdArray<float> load_filter(
    const nc::NdArray<float>& wavelengths, const std::string& name,
    const std::string& brand, const std::string& filter_type, bool percent_transmittance)
{
    std::string datapkg = "agx_emulsion.data.filters." + filter_type;
    std::string filename = brand + "/" + name + ".csv";
    
    // FIX 2: Create a mutable copy of the data to allow modification
    auto data = load_csv(datapkg, filename).copy();
    
    float scale = percent_transmittance ? 100.0f : 1.0f;
    
    // This operation now modifies the copy, not a temporary object
    auto scaled_data = data;
    // Manually scale the second row (y-values)
    for (size_t i = 0; i < scaled_data.shape().cols; ++i) {
        scaled_data(1, i) /= scale;
    }

    // Ensure wavelengths is 1D
    auto wl_flat = wavelengths.flatten();
    
    auto result = interpolate_to_common_axis(scaled_data, wl_flat);
    
    return result;
}

} // namespace utils
} // namespace agx
