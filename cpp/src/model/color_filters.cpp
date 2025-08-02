#include "color_filters.hpp"

#include <cmath>    // std::erf
#include <algorithm> // std::clamp

namespace agx {
namespace model {

// -----------------------------------------------------------------------------
// Free functions
// -----------------------------------------------------------------------------

nc::NdArray<float> create_combined_dichroic_filter(
    const nc::NdArray<float>& wavelength,
    const std::array<float, 3>& filtering_amount_percent,
    const std::array<float, 4>& transitions,
    const std::array<float, 4>& edges,
    float nd_filter)
{
    // Flatten wavelength to ensure 1D shape
    auto wl = wavelength.flatten();
    const std::size_t n = wl.size();

    // Allocate a 3 x N array to hold the three dichroic curves
    nc::NdArray<float> dichroics(3, n);

    // Precompute error functions for each channel
    for (std::size_t j = 0; j < n; ++j) {
        const float w = wl[j];
        // Yellow channel (index 0)
        dichroics(0, j) = std::erf((w - edges[0]) / transitions[0]);
        // Magenta channel (index 1) has a piecewise definition around 550 nm
        if (w <= 550.f) {
            dichroics(1, j) = -std::erf((w - edges[1]) / transitions[1]);
        } else {
            dichroics(1, j) = std::erf((w - edges[2]) / transitions[2]);
        }
        // Cyan channel (index 2)
        dichroics(2, j) = -std::erf((w - edges[3]) / transitions[3]);
    }

    // Normalise into [0, 1] by dividing by 2 and adding 0.5
    for (std::size_t i = 0; i < 3; ++i) {
        for (std::size_t j = 0; j < n; ++j) {
            dichroics(i, j) = dichroics(i, j) / 2.0f + 0.5f;
        }
    }

    // Convert filtering amounts from percent to fraction
    float filtering_amount[3];
    for (std::size_t i = 0; i < 3; ++i) {
        filtering_amount[i] = filtering_amount_percent[i] / 100.0f;
    }

    // Compute the product across the three channels for each wavelength
    nc::NdArray<float> total_filter(n);
    for (std::size_t j = 0; j < n; ++j) {
        float prod = 1.0f;
        for (std::size_t i = 0; i < 3; ++i) {
            // Equivalent to (1 - a_i) + d_ij * a_i
            const float a = filtering_amount[i];
            prod *= ((1.0f - a) + dichroics(i, j) * a);
        }
        total_filter[j] = prod;
    }

    // Apply ND filter percentage
    const float scale = (100.0f - nd_filter) / 100.0f;
    for (std::size_t j = 0; j < n; ++j) {
        total_filter[j] *= scale;
    }

    return total_filter;
}

nc::NdArray<float> filterset(
    const nc::NdArray<float>& illuminant,
    const std::array<float, 3>& values,
    const std::array<float, 4>& edges,
    const std::array<float, 4>& transitions)
{
    // Build the combined filter on the project's wavelength grid
    const auto& wl = agx::config::SPECTRAL_SHAPE.wavelengths;
    auto total_filter = create_combined_dichroic_filter(wl, values, transitions, edges);

    // Multiply illuminant element‑wise by the filter
    auto illum_flat = illuminant.flatten();
    nc::NdArray<float> result(illum_flat.shape());
    for (std::size_t i = 0; i < illum_flat.size(); ++i) {
        result[i] = illum_flat[i] * total_filter[i];
    }
    return result;
}

nc::NdArray<float> sigmoid_erf(const nc::NdArray<float>& x, float center, float width)
{
    // Handle width of zero: return step function around centre
    // to match Python's floating‑point behaviour.  Dividing by zero
    // will yield INF which when passed through erf tends to ±1.
    auto xf = x.flatten();
    nc::NdArray<float> result(xf.size());
    for (std::size_t i = 0; i < xf.size(); ++i) {
        const float t = (xf[i] - center) / width;
        result[i] = std::erf(t) * 0.5f + 0.5f;
    }
    return result;
}

nc::NdArray<float> compute_band_pass_filter(
    const std::array<float, 3>& filter_uv,
    const std::array<float, 3>& filter_ir)
{
    // Extract parameters and clip amplitudes
    float amp_uv  = std::clamp(filter_uv[0], 0.0f, 1.0f);
    float wl_uv   = filter_uv[1];
    float width_uv = filter_uv[2];

    float amp_ir  = std::clamp(filter_ir[0], 0.0f, 1.0f);
    float wl_ir   = filter_ir[1];
    float width_ir = filter_ir[2];

    // Wavelength grid
    const auto& wl = agx::config::SPECTRAL_SHAPE.wavelengths;
    
    // Precompute sigmoid for UV (positive width) and IR (negative width yields descending edge)
    auto sigmoid_uv = sigmoid_erf(wl, wl_uv, width_uv);
    auto sigmoid_ir = sigmoid_erf(wl, wl_ir, -width_ir);

    // Allocate result and compute each element manually to avoid NumCpp broadcasting issues
    nc::NdArray<float> band_pass(wl.size());
    for (std::size_t idx = 0; idx < wl.size(); ++idx) {
        float uv_val = (1.0f - amp_uv) + amp_uv * static_cast<float>(sigmoid_uv[idx]);
        float ir_val = (1.0f - amp_ir) + amp_ir * static_cast<float>(sigmoid_ir[idx]);
        band_pass[idx] = uv_val * ir_val;
    }
    return band_pass;
}

// -----------------------------------------------------------------------------
// DichroicFilters implementation
// -----------------------------------------------------------------------------

DichroicFilters::DichroicFilters(const std::string& brand)
{
    // Copy the global wavelength sampling
    wavelengths = agx::config::SPECTRAL_SHAPE.wavelengths;
    // Load the filters for the given brand
    filters = agx::utils::load_dichroic_filters(wavelengths, brand);
}

nc::NdArray<float> DichroicFilters::apply(
    const nc::NdArray<float>& illuminant,
    const std::array<float, 3>& values) const
{
    // Flatten illuminant to ensure indexing matches wavelengths
    auto illum_flat = illuminant.flatten();
    const std::size_t n = illum_flat.size();

    // Ensure there are exactly three channel values
    float vals[3] = {values[0], values[1], values[2]};

    // Compute dimmed filter matrix (N × 3)
    // and product across channels for each wavelength
    nc::NdArray<float> total_filter(n);
    for (std::size_t j = 0; j < n; ++j) {
        float prod = 1.0f;
        for (std::size_t i = 0; i < 3; ++i) {
            // Note: filters are stored shape (N,3)
            const float f_val = filters(j, i);
            // Equivalent to 1 - (1 - f_val) * vals[i]
            float dimmed = 1.0f - (1.0f - f_val) * vals[i];
            prod *= dimmed;
        }
        total_filter[j] = prod;
    }

    // Multiply by illuminant
    nc::NdArray<float> result(n);
    for (std::size_t j = 0; j < n; ++j) {
        result[j] = illum_flat[j] * total_filter[j];
    }
    return result;
}

void DichroicFilters::plot() const
{
    // Plotting is intentionally omitted in the C++ port.  Use a
    // preferred plotting library in your application if needed.
}

// -----------------------------------------------------------------------------
// GenericFilter implementation
// -----------------------------------------------------------------------------

GenericFilter::GenericFilter(
    const std::string& name,
    const std::string& filter_type,
    const std::string& brand_name,
    bool data_in_percentage,
    bool load_from_database)
    : type(filter_type)
    , brand(brand_name)
{
    wavelengths = agx::config::SPECTRAL_SHAPE.wavelengths;
    transmittance = nc::zeros_like<float>(wavelengths);
    if (load_from_database) {
        // Note: load_filter expects (wavelengths, name, brand, filter_type, percent_transmittance)
        transmittance = agx::utils::load_filter(wavelengths, name, brand_name, filter_type, data_in_percentage);
    }
}

nc::NdArray<float> GenericFilter::apply(const nc::NdArray<float>& illuminant, float value) const
{
    auto illum_flat = illuminant.flatten();
    const std::size_t n = illum_flat.size();
    nc::NdArray<float> result(n);
    for (std::size_t i = 0; i < n; ++i) {
        // Equivalent to 1 - (1 - transmittance[i]) * value
        float dimmed = 1.0f - (1.0f - transmittance[i]) * value;
        result[i] = illum_flat[i] * dimmed;
    }
    return result;
}

// -----------------------------------------------------------------------------
// color_enlarger implementation
// -----------------------------------------------------------------------------

nc::NdArray<float> color_enlarger(
    const nc::NdArray<float>& light_source,
    float y_filter_value,
    float m_filter_value,
    float c_filter_value,
    int enlarger_steps,
    const DichroicFilters* filters)
{
    // Compute per‑channel values divided by the step count
    std::array<float, 3> ymc = {
        y_filter_value / static_cast<float>(enlarger_steps),
        m_filter_value / static_cast<float>(enlarger_steps),
        c_filter_value / static_cast<float>(enlarger_steps)
    };
    // Select filter set: default to durst digital light if null
    const DichroicFilters* filt = filters ? filters : &durst_digital_light_dichroic_filters;
    return filt->apply(light_source, ymc);
}

// -----------------------------------------------------------------------------
// Global variables definitions
// -----------------------------------------------------------------------------

DichroicFilters dichroic_filters{};
DichroicFilters thorlabs_dichroic_filters{ "thorlabs" };
DichroicFilters edmund_optics_dichroic_filters{ "edmund_optics" };
DichroicFilters durst_digital_light_dichroic_filters{ "durst_digital_light" };
GenericFilter schott_kg1_heat_filter{ "KG1", "heat_absorbing", "schott" };
GenericFilter schott_kg3_heat_filter{ "KG3", "heat_absorbing", "schott" };
GenericFilter schott_kg5_heat_filter{ "KG5", "heat_absorbing", "schott" };
GenericFilter generic_lens_transmission{ "canon_24_f28_is", "lens_transmission", "canon", true };

} // namespace model
} // namespace agx