#include "io.hpp"
#include "config.hpp" // Assuming a config.hpp holds SPECTRAL_SHAPE, LOG_EXPOSURE, and data paths
#include "scipy.hpp"  // Your custom scipy wrappers
#include "nlohmann/json.hpp" // For JSON parsing
#include <fstream>
#include <stdexcept>
#include <algorithm> // For std::replace
#include <vector>

// Helper function to get the root data path.
std::string get_data_path() {
    // This path needs to be correct relative to where the final executable is run.
    // For tests in cpp/build/tests/..., this goes up to the project root.
    return "../../../";
}

namespace agx {
namespace utils {

nc::NdArray<float> interpolate_to_common_axis(const nc::NdArray<float>& data, const nc::NdArray<float>& new_x, bool extrapolate, const std::string& method) {
    // Extract first and second rows properly
    auto x = data.row(0);
    auto y = data.row(1);

    // Sorting and finding unique values to prevent interpolation errors
    auto sorted_indices = nc::argsort(x);
    x = x[sorted_indices];
    y = y[sorted_indices];

    // FIX 3 & 4: Manually find indices of unique elements as nc::unique API is different.
    if (x.isempty()) {
        return nc::NdArray<float>();
    }
    std::vector<nc::uint32> unique_indices_vec;
    unique_indices_vec.push_back(0); // First element is always unique
    for (nc::uint32 i = 1; i < x.size(); ++i) {
        if (x[i] != x[i-1]) {
            unique_indices_vec.push_back(i);
        }
    }
    auto unique_indices = nc::NdArray<nc::uint32>(unique_indices_vec);
    x = x[unique_indices];
    y = y[unique_indices];
    
    // Convert float arrays to double for scipy functions
    auto x_double = x.astype<double>();
    auto y_double = y.astype<double>();
    auto new_x_double = new_x.astype<double>();
    
    // Use your custom scipy wrapper for interpolation
    auto interpolator = scipy::interpolate::create_interpolator(
        x_double, y_double,
        method,
        extrapolate);

    // FIX 5: Dereference the unique_ptr to call the operator()
    auto result_double = (*interpolator)(new_x_double);
    
    // Convert back to float
    return result_double.astype<float>();
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

    // Debug: Print the constructed path (commented out for clean output)
    // std::cout << "Loading CSV from: " << full_path << std::endl;

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
    result.log_sensitivity = nc::NdArray<float>(result.wavelengths.size(), 3);
    const char* sens_channels[] = {"r", "g", "b"};
    for (int i = 0; i < 3; ++i) {
        auto data = load_csv(sens_datapkg, "log_sensitivity_" + std::string(sens_channels[i]) + ".csv");
        result.log_sensitivity(nc::Slice(), i) = interpolate_to_common_axis(data, result.wavelengths);
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
    result.density_curves = nc::stack({p_denc_r, p_denc_g, p_denc_b}, nc::Axis::COL);

    // --- Load dye density ---
    std::string dye_cmy_datapkg = maindatapkg + "." + (dye_density_cmy_donor.empty() ? stock : dye_density_cmy_donor);
    result.dye_density = nc::zeros<float>(result.wavelengths.size(), 5);
    const char* dye_channels_cmy[] = {"c", "m", "y"};
    for (int i = 0; i < 3; ++i) {
        auto data = load_csv(dye_cmy_datapkg, "dye_density_" + std::string(dye_channels_cmy[i]) + ".csv");
        result.dye_density(nc::Slice(), i) = interpolate_to_common_axis(data, result.wavelengths);
    }

    if (type == "negative") {
        std::string dye_min_mid_datapkg = maindatapkg + "." + (dye_density_min_mid_donor.empty() ? stock : dye_density_min_mid_donor);
        const char* dye_channels_min_mid[] = {"min", "mid"};
        for (int i = 0; i < 2; ++i) {
            auto data = load_csv(dye_min_mid_datapkg, "dye_density_" + std::string(dye_channels_min_mid[i]) + ".csv");
            result.dye_density(nc::Slice(), i + 3) = interpolate_to_common_axis(data, result.wavelengths);
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
    
    nlohmann::json j = ymc_filters;
    file << j.dump(4);
}

FilterValues read_neutral_ymc_filter_values() {
    std::string path = get_data_path() + "agx_emulsion/data/profiles/enlarger_neutral_ymc_filters.json";
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open JSON file: " + path);
    
    nlohmann::json j;
    file >> j;
    return j; // Return the JSON object directly (matches Python behavior)
}

nc::NdArray<float> load_dichroic_filters(const nc::NdArray<float>& wavelengths, const std::string& brand) {
    auto filters = nc::zeros<float>(wavelengths.size(), 3);
    const char* channels[] = {"y", "m", "c"};
    for (int i = 0; i < 3; ++i) {
        std::string datapkg = "agx_emulsion.data.filters.dichroics." + brand;
        std::string filename = "filter_" + std::string(channels[i]) + ".csv";
        auto data = load_csv(datapkg, filename).transpose();
        filters(nc::Slice(), i) = interpolate_to_common_axis(data, wavelengths);
    }
    return filters;
}

nc::NdArray<float> load_filter(
    const nc::NdArray<float>& wavelengths, const std::string& name,
    const std::string& brand, const std::string& filter_type, bool percent_transmittance)
{
    std::string datapkg = "agx_emulsion.data.filters." + filter_type + "." + brand;
    // FIX 2: Create a mutable copy of the data to allow modification
    auto data = load_csv(datapkg, name + ".csv").transpose().copy();
    float scale = percent_transmittance ? 100.0f : 1.0f;
    
    // This operation now modifies the copy, not a temporary object
    auto scaled_data = data;
    // Manually scale the second row (y-values)
    for (size_t i = 0; i < scaled_data.shape().cols; ++i) {
        scaled_data(1, i) /= scale;
    }

    return interpolate_to_common_axis(scaled_data, wavelengths);
}

} // namespace utils
} // namespace agx
