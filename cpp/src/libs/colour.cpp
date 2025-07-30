#include "colour.hpp"
#include "io.hpp" // For interpolate_to_common_axis
#include <stdexcept>

namespace colour {

// A helper to get the data path, similar to the one in io.cpp
std::string get_colour_data_path() {
    return "../../../agx_emulsion/data/"; // Adjust if necessary
}

nc::NdArray<float> get_cie_1931_2_degree_cmfs() {
    std::string path = get_colour_data_path() + "cmfs/CIE_1931_2_Degree_CMFS.csv";
    // For this specific case, we can use nc::fromfile as it's a simple CSV
    return nc::fromfile<float>(path, ',');
}

nc::NdArray<float> align(const nc::NdArray<float>& spectral_data, const SpectralShape& shape) {
    if (spectral_data.shape().cols < 2) {
        throw std::invalid_argument("Spectral data must have at least 2 columns (wavelengths, values).");
    }

    const auto num_value_channels = spectral_data.shape().cols - 1;
    auto aligned_data = nc::NdArray<float>(shape.wavelengths.size(), num_value_channels);

    // Interpolate each data channel separately
    for (nc::uint32 i = 0; i < num_value_channels; ++i) {
        // Create a 2xN array for interpolation: [wavelengths; values]
        auto channel_data = nc::stack({spectral_data(nc::Slice(), 0), spectral_data(nc::Slice(), i + 1)}, nc::Axis::ROW);
        aligned_data(nc::Slice(), i) = agx::utils::interpolate_to_common_axis(channel_data, shape.wavelengths);
    }

    return aligned_data;
}

} // namespace colour
