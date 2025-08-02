#pragma once

#include <array>
#include <string>
#include <NumCpp.hpp>

#include "config.hpp"
#include "io.hpp"

namespace agx {
namespace model {

/**
 * @brief Create a combined dichroic filter with Y/M/C channels.
 *
 * This function creates a combined dichroic filter by applying Y/M/C
 * filtering amounts to a wavelength array. The filtering is done using
 * sigmoid transitions at specified edges with given transition widths.
 *
 * @param wavelength Wavelength array (nm).
 * @param filtering_amount_percent Y/M/C filtering amounts in percent [0-100].
 * @param transitions Transition widths for Y/M/C channels (nm).
 * @param edges Edge wavelengths for Y/M/C channels (nm).
 * @param nd_filter Neutral density filter value (0-1).
 * @return Combined filter array.
 */
nc::NdArray<float> create_combined_dichroic_filter(
    const nc::NdArray<float>& wavelength,
    const std::array<float, 3>& filtering_amount_percent,
    const std::array<float, 4>& transitions,
    const std::array<float, 4>& edges,
    float nd_filter = 0.0f);

/**
 * @brief Apply Y/M/C filters to an illuminant using sigmoid transitions.
 *
 * This function applies Y/M/C filter values to an illuminant spectrum
 * using sigmoid transitions at specified edges. The function multiplies the
 * illuminant by the filter and returns the result.  In the Python
 * implementation the return value is a ``colour.SpectralDistribution``;
 * here we simply return the filtered array.
 *
 * @param illuminant The spectrum to be filtered (same length as
 *        ``agx::config::SPECTRAL_SHAPE.wavelengths``).
 * @param values Y/M/C filter values in percent (0–100).  These are
 *        divided by 100 internally before mixing.
 * @param edges Four edge centres (nm).  Defaults to {510, 495, 605, 590}.
 * @param transitions Four transition widths (nm).  Defaults to
 *        {10, 10, 10, 10}.
 * @return The filtered illuminant as an ``nc::NdArray<float>``.
 */
nc::NdArray<float> filterset(
    const nc::NdArray<float>& illuminant,
    const std::array<float, 3>& values = {0.f, 0.f, 0.f},
    const std::array<float, 4>& edges = {510.f, 495.f, 605.f, 590.f},
    const std::array<float, 4>& transitions = {10.f, 10.f, 10.f, 10.f});

/**
 * @brief Small helper returning an error‑function shaped sigmoid.
 *
 * This function matches the Python ``sigmoid_erf`` implementation and
 * computes ``erf((x − center) / width) * 0.5 + 0.5`` element‑wise.
 * It accepts an array input and returns an array of the same shape.
 *
 * @param x Input array.
 * @param center The centre of the sigmoid.
 * @param width The width of the transition.  Negative widths invert
 *        the transition (as in the Python implementation).
 * @return An array of the same shape as ``x`` containing values in
 *         the range [0, 1].
 */
nc::NdArray<float> sigmoid_erf(const nc::NdArray<float>& x, float center, float width = 1.0f);

/**
 * @brief Compute a band‑pass filter with UV and IR roll‑offs.
 *
 * Equivalent to Python's ``compute_band_pass_filter``.  Each filter
 * parameter comprises an amplitude (0–1), a centre wavelength and a
 * width.  Amplitudes are clipped into [0, 1] exactly as NumPy does.
 * Negative widths on the IR side invert the sigmoid, providing the
 * falling edge.  The product of the UV and IR filters is returned.
 *
 * @param filter_uv The UV filter parameters {amp, centre, width}.  Amp
 *        is clipped to [0, 1].
 * @param filter_ir The IR filter parameters {amp, centre, width}.  Amp
 *        is clipped to [0, 1] and width is negated internally to
 *        create the falling edge.
 * @return A one‑dimensional band‑pass filter array matching
 *         ``agx::config::SPECTRAL_SHAPE.wavelengths``.
 */
nc::NdArray<float> compute_band_pass_filter(
    const std::array<float, 3>& filter_uv = {1.f, 410.f, 8.f},
    const std::array<float, 3>& filter_ir = {1.f, 675.f, 15.f});

/**
 * @brief A simple wrapper around a set of three dichroic filters.
 *
 * The constructor loads dichroic Y/M/C filters for a given brand via
 * ``agx::utils::load_dichroic_filters``.  The filters are stored
 * internally as an ``nc::NdArray<float>`` of shape (N, 3) where N
 * equals the number of wavelengths in ``agx::config::SPECTRAL_SHAPE``.
 *
 * The ``apply`` member function multiplies the three channels by
 * per‑channel values and returns the resulting one‑dimensional
 * spectrum.  This mirrors the Python ``DichroicFilters.apply`` method.
 */
class DichroicFilters {
public:
    /**
     * Construct a new set of dichroic filters for the given brand.
     * The available brands correspond to subdirectories within
     * ``agx_emulsion.data.filters.dichroics`` (e.g. "thorlabs",
     * "edmund_optics", "durst_digital_light").
     *
     * @param brand Name of the manufacturer.
     */
    explicit DichroicFilters(const std::string& brand = "thorlabs");

    /**
     * Apply the loaded dichroic filters to an illuminant with Y/M/C
     * values.  This performs the elementwise operation
     * ``dimmed_filters = 1 − (1 − filters) * values`` and then
     * multiplies the three channels together along axis 1.
     *
     * @param illuminant One‑dimensional array of spectral intensities.
     * @param values Three values in the range [0, 1] specifying the
     *        relative contribution of each Y/M/C channel.  Defaults to
     *        {0, 0, 0}.
     * @return The filtered illuminant array.
     */
    nc::NdArray<float> apply(
        const nc::NdArray<float>& illuminant,
        const std::array<float, 3>& values = {0.f, 0.f, 0.f}) const;

    /**
     * Placeholder plot function.  In Python this method renders a
     * plot of the filter curves via Matplotlib.  For the C++ port the
     * function is provided for API completeness but does nothing.
     */
    void plot() const;

    /// Public member giving access to the wavelength sampling.
    nc::NdArray<float> wavelengths;

    /// Public member containing the filter matrix with shape (N, 3).
    nc::NdArray<float> filters;
};

/**
 * @brief Representation of a single generic filter (e.g. heat
 *        absorbing or lens transmission).
 *
 * On construction this class loads a transmittance curve via
 * ``agx::utils::load_filter`` unless ``load_from_database`` is set
 * false.  Calling ``apply`` multiplies an illuminant by the filter
 * attenuated by a scalar value ``value``.  This mirrors the Python
 * ``GenericFilter`` class.
 */
class GenericFilter {
public:
    /**
     * Construct a new GenericFilter.
     *
     * @param name The filter name within the data directory (e.g. "KG3").
     * @param type The filter type subdirectory (e.g. "heat_absorbing").
     * @param brand The manufacturer (e.g. "schott").
     * @param data_in_percentage When true, the loaded data are scaled
     *        assuming they are given in percentage.
     * @param load_from_database If false, the transmittance array
     *        remains initialised to zeros.
     */
    GenericFilter(
        const std::string& name = "KG3",
        const std::string& type = "heat_absorbing",
        const std::string& brand = "schott",
        bool data_in_percentage = false,
        bool load_from_database = true);

    /**
     * Apply the filter to an illuminant with a scalar attenuation.
     * Equivalent to the Python implementation ``illuminant * (1 − (1 − transmittance) * value)``.
     *
     * @param illuminant One‑dimensional array of spectral intensities.
     * @param value Scalar in [0, 1] controlling the strength of the filter.
     * @return The attenuated illuminant array.
     */
    nc::NdArray<float> apply(const nc::NdArray<float>& illuminant, float value = 1.0f) const;

    /// Public member giving access to the wavelength sampling.
    nc::NdArray<float> wavelengths;
    /// The filter type (e.g. "heat_absorbing").
    std::string type;
    /// The manufacturer.
    std::string brand;
    /// Transmittance curve (one‑dimensional).
    nc::NdArray<float> transmittance;
};

/**
 * @brief Apply Y/M/C filter values to a light source using a set of
 *        dichroic filters.
 *
 * This is a convenience wrapper around ``DichroicFilters::apply``
 * matching the Python ``color_enlarger`` function.  The input
 * filter values are divided by the ``enlarger_steps`` parameter to
 * match the behaviour of the original code (170 steps for Durst
 * enlargers by default).
 *
 * @param light_source One‑dimensional illuminant array.
 * @param y_filter_value Value for the yellow filter wheel.
 * @param m_filter_value Value for the magenta filter wheel.
 * @param c_filter_value Value for the cyan filter wheel (default 0).
 * @param enlarger_steps Total number of discrete steps in the filter
 *        wheel.  Defaults to ``agx::config::ENLARGER_STEPS``.
 * @param filters Pointer to the dichroic filter set to use.  If
 *        ``nullptr``, the global
 *        ``durst_digital_light_dichroic_filters`` instance is used.
 * @return The filtered illuminant array.
 */
nc::NdArray<float> color_enlarger(
    const nc::NdArray<float>& light_source,
    float y_filter_value,
    float m_filter_value,
    float c_filter_value = 0.f,
    int enlarger_steps = agx::config::ENLARGER_STEPS,
    const DichroicFilters* filters = nullptr);

// -----------------------------------------------------------------------------
// Global filter instances
//
// The following variables correspond to the module‑level instances in the
// original Python file.  They are defined in the translation unit
// ``color_filters.cpp`` and declared here as extern so that users may
// reference them.  Construction of these objects will load filter data
// from disk via agx::utils::load_dichroic_filters and load_filter.
// -----------------------------------------------------------------------------

extern DichroicFilters dichroic_filters;
extern DichroicFilters thorlabs_dichroic_filters;
extern DichroicFilters edmund_optics_dichroic_filters;
extern DichroicFilters durst_digital_light_dichroic_filters;
extern GenericFilter schott_kg1_heat_filter;
extern GenericFilter schott_kg3_heat_filter;
extern GenericFilter schott_kg5_heat_filter;
extern GenericFilter generic_lens_transmission;

} // namespace model
} // namespace agx