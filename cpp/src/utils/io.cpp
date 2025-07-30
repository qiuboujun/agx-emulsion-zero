#include "io.hpp"
#include "config.hpp" // tested
#include "scipy.hpp"  // Your custom scipy wrappers
#include "nlohmann/json.hpp" // For JSON parsing
#include <fstream>
#include <stdexcept>
#include <algorithm> // For std::replace

// Helper function to get the root data path.
// For testing, this assumes the executable is run from a build directory inside cpp/
// e.g., cpp/build/tests/io/
std::string get_data_path() {
    return "../../../";
}

namespace agx {
namespace utils {

nc::NdArray<float> interpolate_to_common_axis(const nc::NdArray<float>& data, const nc::NdArray<float>& new_x, bool extrapolate, const std::string& method) {
    auto x = data(0, nc::Slice());
    auto y = data(1, nc::Slice());

    // Sorting and finding unique values to prevent interpolation errors
    auto sorted_indices = nc::argsort(x);
    x = x[sorted_indices];
    y = y[sorted_indices];

    auto unique_results = nc::unique<float>(x, true);
    x = x[unique_results.second];
    y = y[unique_results.second];

    // Use your custom scipy wrapper for interpolation
    auto interpolator = scipy::interpolate::create_interpolator(x, y, method, extrapolate);
    return interpolator(new_x);
}

nc::NdArray<float> load_csv(const std::string& datapkg, const std::string& filename) {
    std::string full_path = get_data_path() + datapkg;
    std::replace(full_path.begin(), full_path.end(), '.', '/');
    full_path += "/" + filename;

    // This simplified version assumes clean CSVs without empty values.
    // For production, a more robust line-by-line parser might be needed.
    return nc::fromfile<float>(full_path, ',').transpose();
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
        responsivities(nc::Slice(), i) = interpolate_to_common_axis(data, wavelengths, false, "linear");
    }
    responsivities = nc::where(responsivities < 0.0f, 0.0f, responsivities);
    responsivities /= nc::nansum(responsivities, nc::Axis::ROW).reshape(-1, 1);
    return responsivities;
}

void save_ymc_filter_values(const FilterValues& ymc_filters) {
    std::string path = get_data_path() + "agx_emulsion/data/profiles/enlarger_neutral_ymc_filters.json";
    std::ofstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open JSON file for writing: " + path);
    
    nlohmann::json j = ymc_filters;
    file << j.dump(4); // dump with an indent of 4 for pretty printing
}

FilterValues read_neutral_ymc_filter_values() {
    std::string path = get_data_path() + "agx_emulsion/data/profiles/enlarger_neutral_ymc_filters.json";
    std::ifstream file(path);
    if (!file.is_open()) throw std::runtime_error("Cannot open JSON file: " + path);
    
    nlohmann::json j;
    file >> j;
    return j.get<FilterValues>();
}

nc::NdArray<float> load_dichroic_filters(const nc::NdArray<float>& wavelengths, const std::string& brand) {
    auto filters = nc::zeros<float>(wavelengths.size(), 3);
    const char* channels[] = {"y", "m", "c"};
    for (int i = 0; i < 3; ++i) {
        std::string datapkg = "agx_emulsion.data.filters.dichroics." + brand;
        std::string filename = "filter_" + std::string(channels[i]) + ".csv";
        auto data = load_csv(datapkg, filename).transpose(); // load_csv already transposes, so transpose back
        filters(nc::Slice(), i) = interpolate_to_common_axis(data, wavelengths);
    }
    return filters;
}

nc::NdArray<float> load_filter(
    const nc::NdArray<float>& wavelengths, const std::string& name,
    const std::string& brand, const std::string& filter_type, bool percent_transmittance)
{
    std::string datapkg = "agx_emulsion.data.filters." + filter_type + "." + brand;
    auto data = load_csv(datapkg, name + ".csv").transpose();
    float scale = percent_transmittance ? 100.0f : 1.0f;
    
    // Manually scale the y-values before interpolation
    auto scaled_data = data;
    scaled_data(1, nc::Slice()) /= scale;

    return interpolate_to_common_axis(scaled_data, wavelengths);
}

} // namespace utils
} // namespace agx
