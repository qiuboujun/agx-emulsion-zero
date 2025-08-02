#include "color_filters.hpp"
#include <algorithm>
#include <cmath>
#include <iostream>

namespace agx {
namespace model {

//================================================================================
// Utility Functions
//================================================================================

nc::NdArray<float> sigmoid_erf(const nc::NdArray<float>& x, float center, float width) {
    // Python: scipy.special.erf((x-center)/width)*0.5+0.5
    nc::NdArray<float> result = x.copy();
    for (auto& val : result) {
        val = std::erf((val - center) / width) * 0.5f + 0.5f;
    }
    return result;
}

nc::NdArray<float> create_combined_dichroic_filter(
    const nc::NdArray<float>& wavelength,
    const std::vector<float>& filtering_amount_percent,
    const std::vector<float>& transitions,
    const std::vector<float>& edges,
    float nd_filter) {
    
    // Python: dichroics = np.zeros((3, np.size(wavelength)))
    nc::NdArray<float> dichroics(3, wavelength.size());
    
    // Python: dichroics[0] = scipy.special.erf((wavelength-edges[0])/transitions[0])
    for (size_t i = 0; i < wavelength.size(); ++i) {
        dichroics(0, i) = std::erf((wavelength[i] - edges[0]) / transitions[0]);
    }
    
    // Python: dichroics[1][wavelength<=550] = -scipy.special.erf((wavelength[wavelength<=550]-edges[1])/transitions[1])
    // Python: dichroics[1][wavelength>550] = scipy.special.erf((wavelength[wavelength>550]-edges[2])/transitions[2])
    for (size_t i = 0; i < wavelength.size(); ++i) {
        if (wavelength[i] <= 550.0f) {
            dichroics(1, i) = -std::erf((wavelength[i] - edges[1]) / transitions[1]);
        } else {
            dichroics(1, i) = std::erf((wavelength[i] - edges[2]) / transitions[2]);
        }
    }
    
    // Python: dichroics[2] = -scipy.special.erf((wavelength-edges[3])/transitions[3])
    for (size_t i = 0; i < wavelength.size(); ++i) {
        dichroics(2, i) = -std::erf((wavelength[i] - edges[3]) / transitions[3]);
    }
    
    // Python: dichroics = dichroics/2 + 0.5
    dichroics = dichroics / 2.0f + 0.5f;
    
    // Python: filtering_amount = np.array(filtering_amount_percent)/100.0
    std::vector<float> filtering_amount = filtering_amount_percent;
    for (auto& val : filtering_amount) {
        val /= 100.0f;
    }
    
    // Python: total_filter = np.prod(((1-filtering_amount[:,None]) + dichroics*filtering_amount[:, None]),axis = 0)
    nc::NdArray<float> total_filter(wavelength.size());
    for (size_t i = 0; i < wavelength.size(); ++i) {
        float prod = 1.0f;
        for (size_t j = 0; j < 3; ++j) {
            float term = (1.0f - filtering_amount[j]) + dichroics(j, i) * filtering_amount[j];
            prod *= term;
        }
        total_filter[i] = prod;
    }
    
    // Python: total_filter *=(100-nd_filter)/100
    total_filter *= (100.0f - nd_filter) / 100.0f;
    
    return total_filter;
}

nc::NdArray<float> filterset(
    const nc::NdArray<float>& illuminant,
    const std::vector<float>& values,
    const std::vector<float>& edges,
    const std::vector<float>& transitions) {
    
    // Python: total_filter = create_combined_dichroic_filter(illuminant.wavelengths, filtering_amount_percent=values, transitions=transitions, edges=edges)
    auto total_filter = create_combined_dichroic_filter(
        agx::config::SPECTRAL_SHAPE.wavelengths, values, transitions, edges);
    
    // Python: values = illuminant*total_filter
    // Python: filtered_illuminant = colour.SpectralDistribution(values, domain=SPECTRAL_SHAPE)
    // For C++, we just return the element-wise product
    return illuminant * total_filter;
}

nc::NdArray<float> compute_band_pass_filter(
    const std::vector<float>& filter_uv,
    const std::vector<float>& filter_ir) {
    
    // Extract parameters
    float amp_uv = filter_uv[0];
    float wl_uv = filter_uv[1];
    float width_uv = filter_uv[2];
    
    float amp_ir = filter_ir[0];
    float wl_ir = filter_ir[1];
    float width_ir = filter_ir[2];
    
    // Python: amp_uv = np.clip(amp_uv, 0, 1)
    // Python: amp_ir = np.clip(amp_ir, 0, 1)
    amp_uv = std::max(0.0f, std::min(1.0f, amp_uv));
    amp_ir = std::max(0.0f, std::min(1.0f, amp_ir));
    
    // Python: wl = SPECTRAL_SHAPE.wavelengths
    auto wl = agx::config::SPECTRAL_SHAPE.wavelengths;
    
    // Python: filter_uv = 1-amp_uv + amp_uv*sigmoid_erf(wl, wl_uv, width=width_uv)
    auto filter_uv_result = (1.0f - amp_uv) + amp_uv * sigmoid_erf(wl, wl_uv, width_uv);
    
    // Python: filter_ir = 1-amp_ir + amp_ir*sigmoid_erf(wl, wl_ir, width=-width_ir)
    auto filter_ir_result = (1.0f - amp_ir) + amp_ir * sigmoid_erf(wl, wl_ir, -width_ir);
    
    // Python: band_pass_filter = filter_uv * filter_ir
    return filter_uv_result * filter_ir_result;
}

nc::NdArray<float> color_enlarger(
    const nc::NdArray<float>& light_source,
    float y_filter_value,
    float m_filter_value,
    float c_filter_value,
    int enlarger_steps,
    const std::string& filters) {
    
    // Python: ymc_filter_values = np.array([y_filter_value, m_filter_value, c_filter_value]) / enlarger_steps
    std::vector<float> ymc_filter_values = {y_filter_value, m_filter_value, c_filter_value};
    for (auto& val : ymc_filter_values) {
        val /= static_cast<float>(enlarger_steps);
    }
    
    // Python: filtered_illuminant = filters.apply(light_source, values=ymc_filter_values)
    // We need to create a DichroicFilters instance with the specified brand
    DichroicFilters filter_instance(filters);
    return filter_instance.apply(light_source, ymc_filter_values);
}

//================================================================================
// DichroicFilters Class Implementation
//================================================================================

DichroicFilters::DichroicFilters(const std::string& brand) {
    // Python: self.wavelengths = SPECTRAL_SHAPE.wavelengths
    wavelengths = agx::config::SPECTRAL_SHAPE.wavelengths;
    
    // Python: self.filters = np.zeros((np.size(self.wavelengths), 3))
    // Python: self.filters = load_dichroic_filters(self.wavelengths, brand)
    filters = agx::utils::load_dichroic_filters(wavelengths, brand);
}

nc::NdArray<float> DichroicFilters::apply(
    const nc::NdArray<float>& illuminant, 
    const std::vector<float>& values) const {
    
    // Python: dimmed_filters = 1 - (1-self.filters)*np.array(values)
    nc::NdArray<float> dimmed_filters = filters.copy();
    for (size_t i = 0; i < filters.shape().rows; ++i) {
        for (size_t j = 0; j < filters.shape().cols; ++j) {
            dimmed_filters(i, j) = 1.0f - (1.0f - filters(i, j)) * values[j];
        }
    }
    
    // Python: total_filter = np.prod(dimmed_filters, axis=1)
    nc::NdArray<float> total_filter(dimmed_filters.shape().rows);
    for (size_t i = 0; i < dimmed_filters.shape().rows; ++i) {
        float prod = 1.0f;
        for (size_t j = 0; j < dimmed_filters.shape().cols; ++j) {
            prod *= dimmed_filters(i, j);
        }
        total_filter[i] = prod;
    }
    
    // Python: filtered_illuminant = illuminant*total_filter
    return illuminant * total_filter;
}

//================================================================================
// GenericFilter Class Implementation
//================================================================================

GenericFilter::GenericFilter(const std::string& name,
                           const std::string& type,
                           const std::string& brand,
                           bool data_in_percentage,
                           bool load_from_database) 
    : type(type), brand(brand) {
    
    // Python: self.wavelengths = SPECTRAL_SHAPE.wavelengths
    wavelengths = agx::config::SPECTRAL_SHAPE.wavelengths;
    
    // Python: self.transmittance = np.zeros_like(self.wavelengths)
    transmittance = nc::zeros<float>(wavelengths.shape());
    
    // Python: if load_from_database: self.transmittance = load_filter(...)
    if (load_from_database) {
        transmittance = agx::utils::load_filter(wavelengths, name, brand, type, data_in_percentage);
    }
}

nc::NdArray<float> GenericFilter::apply(const nc::NdArray<float>& illuminant, float value) const {
    // Python: dimmed_filter = 1 - (1-self.transmittance)*value
    nc::NdArray<float> dimmed_filter = 1.0f - (1.0f - transmittance) * value;
    
    // Python: filtered_illuminant = illuminant*dimmed_filter
    return illuminant * dimmed_filter;
}

//================================================================================
// Global Filter Instances
//================================================================================

std::unique_ptr<DichroicFilters> dichroic_filters;
std::unique_ptr<DichroicFilters> thorlabs_dichroic_filters;
std::unique_ptr<DichroicFilters> edmund_optics_dichroic_filters;
std::unique_ptr<DichroicFilters> durst_digital_light_dichroic_filters;
std::unique_ptr<GenericFilter> schott_kg1_heat_filter;
std::unique_ptr<GenericFilter> schott_kg3_heat_filter;
std::unique_ptr<GenericFilter> schott_kg5_heat_filter;
std::unique_ptr<GenericFilter> generic_lens_transmission;

void initialize_global_filters() {
    // Python: dichroic_filters = DichroicFilters()
    dichroic_filters = std::make_unique<DichroicFilters>();
    
    // Python: thorlabs_dichroic_filters = DichroicFilters(brand='thorlabs')
    thorlabs_dichroic_filters = std::make_unique<DichroicFilters>("thorlabs");
    
    // Python: edmund_optics_dichroic_filters = DichroicFilters(brand='edmund_optics')
    edmund_optics_dichroic_filters = std::make_unique<DichroicFilters>("edmund_optics");
    
    // Python: durst_digital_light_dicrhoic_filters = DichroicFilters(brand='durst_digital_light')
    durst_digital_light_dichroic_filters = std::make_unique<DichroicFilters>("durst_digital_light");
    
    // Python: schott_kg1_heat_filter = GenericFilter(name='KG1', type='heat_absorbing', brand='schott')
    schott_kg1_heat_filter = std::make_unique<GenericFilter>("KG1", "heat_absorbing", "schott");
    
    // Python: schott_kg3_heat_filter = GenericFilter(name='KG3', type='heat_absorbing', brand='schott')
    schott_kg3_heat_filter = std::make_unique<GenericFilter>("KG3", "heat_absorbing", "schott");
    
    // Python: schott_kg5_heat_filter = GenericFilter(name='KG5', type='heat_absorbing', brand='schott')
    schott_kg5_heat_filter = std::make_unique<GenericFilter>("KG5", "heat_absorbing", "schott");
    
    // Python: generic_lens_transmission = GenericFilter(name='canon_24_f28_is', type='lens_transmission', brand='canon', data_in_percentage=True)
    generic_lens_transmission = std::make_unique<GenericFilter>("canon_24_f28_is", "lens_transmission", "canon", true);
}

} // namespace model
} // namespace agx 