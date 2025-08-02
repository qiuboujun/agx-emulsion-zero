#pragma once

#include "NumCpp.hpp"
#include "config.hpp"
#include "io.hpp"
#include "scipy.hpp"
#include <string>
#include <vector>
#include <memory>

namespace agx {
namespace model {

//================================================================================
// Utility Functions
//================================================================================

/**
 * @brief Sigmoid error function for band pass filters
 * @param x Input values
 * @param center Center of the sigmoid
 * @param width Width parameter (default: 1)
 * @return Sigmoid values
 */
nc::NdArray<float> sigmoid_erf(const nc::NdArray<float>& x, float center, float width = 1.0f);

/**
 * @brief Create a combined dichroic filter
 * @param wavelength Wavelength array
 * @param filtering_amount_percent Filtering amount as percentage
 * @param transitions Transition parameters
 * @param edges Edge parameters
 * @param nd_filter ND filter value (default: 0)
 * @return Combined filter values
 */
nc::NdArray<float> create_combined_dichroic_filter(
    const nc::NdArray<float>& wavelength,
    const std::vector<float>& filtering_amount_percent,
    const std::vector<float>& transitions,
    const std::vector<float>& edges,
    float nd_filter = 0.0f
);

/**
 * @brief Apply filterset to illuminant
 * @param illuminant Input illuminant spectral distribution
 * @param values Filter values [Y, M, C] (default: [0, 0, 0])
 * @param edges Edge parameters (default: [510, 495, 605, 590])
 * @param transitions Transition parameters (default: [10, 10, 10, 10])
 * @return Filtered illuminant
 */
nc::NdArray<float> filterset(
    const nc::NdArray<float>& illuminant,
    const std::vector<float>& values = {0.0f, 0.0f, 0.0f},
    const std::vector<float>& edges = {510.0f, 495.0f, 605.0f, 590.0f},
    const std::vector<float>& transitions = {10.0f, 10.0f, 10.0f, 10.0f}
);

/**
 * @brief Compute band pass filter
 * @param filter_uv UV filter parameters [amplitude, wavelength, width] (default: [1, 410, 8])
 * @param filter_ir IR filter parameters [amplitude, wavelength, width] (default: [1, 675, 15])
 * @return Band pass filter values
 */
nc::NdArray<float> compute_band_pass_filter(
    const std::vector<float>& filter_uv = {1.0f, 410.0f, 8.0f},
    const std::vector<float>& filter_ir = {1.0f, 675.0f, 15.0f}
);

/**
 * @brief Color enlarger function
 * @param light_source Input light source
 * @param y_filter_value Y filter value
 * @param m_filter_value M filter value
 * @param c_filter_value C filter value (default: 0)
 * @param enlarger_steps Enlarger steps (default: ENLARGER_STEPS)
 * @param filters Filter brand to use (default: "durst_digital_light")
 * @return Filtered light source
 */
nc::NdArray<float> color_enlarger(
    const nc::NdArray<float>& light_source,
    float y_filter_value,
    float m_filter_value,
    float c_filter_value = 0.0f,
    int enlarger_steps = agx::config::ENLARGER_STEPS,
    const std::string& filters = "durst_digital_light"
);

//================================================================================
// DichroicFilters Class
//================================================================================

class DichroicFilters {
public:
    /**
     * @brief Constructor
     * @param brand Filter brand (default: "thorlabs")
     */
    explicit DichroicFilters(const std::string& brand = "thorlabs");
    
    /**
     * @brief Apply filters to illuminant
     * @param illuminant Input illuminant
     * @param values Filter values [Y, M, C] (default: [0, 0, 0])
     * @return Filtered illuminant
     */
    nc::NdArray<float> apply(const nc::NdArray<float>& illuminant, 
                            const std::vector<float>& values = {0.0f, 0.0f, 0.0f}) const;
    
    /**
     * @brief Get wavelengths
     * @return Wavelength array
     */
    const nc::NdArray<float>& get_wavelengths() const { return wavelengths; }
    
    /**
     * @brief Get filters
     * @return Filter array
     */
    const nc::NdArray<float>& get_filters() const { return filters; }

private:
    nc::NdArray<float> wavelengths;
    nc::NdArray<float> filters;
};

//================================================================================
// GenericFilter Class
//================================================================================

class GenericFilter {
public:
    /**
     * @brief Constructor
     * @param name Filter name (default: "KG3")
     * @param type Filter type (default: "heat_absorbing")
     * @param brand Filter brand (default: "schott")
     * @param data_in_percentage Whether data is in percentage (default: false)
     * @param load_from_database Whether to load from database (default: true)
     */
    explicit GenericFilter(const std::string& name = "KG3",
                          const std::string& type = "heat_absorbing",
                          const std::string& brand = "schott",
                          bool data_in_percentage = false,
                          bool load_from_database = true);
    
    /**
     * @brief Apply filter to illuminant
     * @param illuminant Input illuminant
     * @param value Filter value (default: 1.0)
     * @return Filtered illuminant
     */
    nc::NdArray<float> apply(const nc::NdArray<float>& illuminant, float value = 1.0f) const;
    
    /**
     * @brief Get wavelengths
     * @return Wavelength array
     */
    const nc::NdArray<float>& get_wavelengths() const { return wavelengths; }
    
    /**
     * @brief Get transmittance
     * @return Transmittance array
     */
    const nc::NdArray<float>& get_transmittance() const { return transmittance; }
    
    /**
     * @brief Get filter type
     * @return Filter type
     */
    const std::string& get_type() const { return type; }
    
    /**
     * @brief Get filter brand
     * @return Filter brand
     */
    const std::string& get_brand() const { return brand; }

private:
    nc::NdArray<float> wavelengths;
    std::string type;
    std::string brand;
    nc::NdArray<float> transmittance;
};

//================================================================================
// Global Filter Instances
//================================================================================

// These will be initialized in the implementation file
extern std::unique_ptr<DichroicFilters> dichroic_filters;
extern std::unique_ptr<DichroicFilters> thorlabs_dichroic_filters;
extern std::unique_ptr<DichroicFilters> edmund_optics_dichroic_filters;
extern std::unique_ptr<DichroicFilters> durst_digital_light_dichroic_filters;
extern std::unique_ptr<GenericFilter> schott_kg1_heat_filter;
extern std::unique_ptr<GenericFilter> schott_kg3_heat_filter;
extern std::unique_ptr<GenericFilter> schott_kg5_heat_filter;
extern std::unique_ptr<GenericFilter> generic_lens_transmission;

/**
 * @brief Initialize global filter instances
 * This function should be called once after config initialization
 */
void initialize_global_filters();

} // namespace model
} // namespace agx 