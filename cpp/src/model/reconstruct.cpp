// SPDX-License-Identifier: MIT

#include "reconstruct.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include "scipy.hpp"
#include "measure.hpp"
#include "io.hpp"

namespace agx {
namespace profiles {

using nc::NdArray;

static NdArray<float> erf_array(const NdArray<float>& x) {
    NdArray<float> out = x.copy();
    for (size_t i = 0; i < x.size(); ++i) {
        out[i] = std::erff(x[i]);
    }
    return out;
}

NdArray<float> low_pass_filter(const NdArray<float>& wl, float wl_max, float width, float amp) {
    auto t = (wl - wl_max) / width;
    auto e = erf_array(t);
    return 1.0f - amp * (e + 1.0f) * 0.5f;
}

NdArray<float> high_pass_filter(const NdArray<float>& wl, float wl_min, float width, float amp) {
    auto t = (wl - wl_min) / width;
    auto e = erf_array(t);
    return 1.0f - amp + amp * (e + 1.0f) * 0.5f;
}

NdArray<float> high_pass_gaussian(const NdArray<float>& wl, float wl_max, float width, float amount) {
    NdArray<float> out = wl.copy();
    for (size_t i = 0; i < wl.size(); ++i) {
        const float x = wl[i];
        out[i] = amount * std::exp(-(x - wl_max + width) * (x - wl_max + width) / (2.0f * width * width));
    }
    return out;
}

NdArray<float> low_pass_gaussian(const NdArray<float>& wl, float wl_max, float width, float amount) {
    NdArray<float> out = wl.copy();
    for (size_t i = 0; i < wl.size(); ++i) {
        const float x = wl[i];
        out[i] = amount * std::exp(-(x - wl_max - width) * (x - wl_max - width) / (2.0f * width * width));
    }
    return out;
}

NdArray<float> shift_stretch(const NdArray<float>& wl, const NdArray<float>& spectrum, float amp, float width, float shift) {
    // Fast path: default parameters imply identity mapping
    if (std::abs(amp - 1.0f) < 1e-6f && std::abs(width - 1.0f) < 1e-6f && std::abs(shift) < 1e-6f) {
        return spectrum.copy();
    }

    // Find center at argmax of spectrum ignoring NaNs
    size_t argmax = 0; float maxv = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < spectrum.size(); ++i) {
        const float v = spectrum[i];
        if (std::isnan(v)) continue;
        if (v > maxv) { maxv = v; argmax = i; }
    }
    const float center = wl[argmax];

    // Build mask of valid points
    std::vector<double> x_valid; x_valid.reserve(wl.size());
    std::vector<double> y_valid; y_valid.reserve(wl.size());
    for (size_t i = 0; i < wl.size(); ++i) {
        const float v = spectrum[i];
        if (!std::isnan(v)) {
            x_valid.push_back(static_cast<double>(wl[i]));
            y_valid.push_back(static_cast<double>(v));
        }
    }
    // Smoothing spline with lambda=100 to match Python smoothing intent
    auto sp = scipy::interpolate::make_smoothing_spline(
        nc::NdArray<double>(x_valid), nc::NdArray<double>(y_valid), 100.0);

    // Apply shift/stretch mapping
    NdArray<float> mapped = wl.copy();
    for (size_t i = 0; i < wl.size(); ++i) {
        const double x = (static_cast<double>(wl[i]) - center) / width + center + shift;
        mapped[i] = static_cast<float>(sp(x));
    }

    // Resmooth and clamp using same smoothing spline
    auto sp2 = scipy::interpolate::make_smoothing_spline(
        wl.astype<double>(), mapped.astype<double>(), 100.0);
    NdArray<float> out = wl.copy();
    for (size_t i = 0; i < wl.size(); ++i) {
        out[i] = static_cast<float>(sp2(static_cast<double>(wl[i])));
        if (out[i] < 0.f) out[i] = 0.f;
        if (std::isnan(spectrum[i])) out[i] = std::numeric_limits<float>::quiet_NaN();
    }
    for (size_t i = 0; i < out.size(); ++i) out[i] *= amp;
    return out;
}

NdArray<float> shift_stretch_cmy(const NdArray<float>& wl, const NdArray<float>& cmy,
                                 float da0, float dw0, float ds0,
                                 float da1, float dw1, float ds1,
                                 float da2, float dw2, float ds2) {
    NdArray<float> out(cmy.shape());
    auto c = shift_stretch(wl, cmy(nc::Slice(0, cmy.shape().rows), 0), da0, dw0, ds0);
    auto m = shift_stretch(wl, cmy(nc::Slice(0, cmy.shape().rows), 1), da1, dw1, ds1);
    auto y = shift_stretch(wl, cmy(nc::Slice(0, cmy.shape().rows), 2), da2, dw2, ds2);
    for (size_t i = 0; i < wl.size(); ++i) {
        out(i, 0) = c[i];
        out(i, 1) = m[i];
        out(i, 2) = y[i];
    }
    return out;
}

NdArray<float> gaussian_profiles(const NdArray<float>& wl, const std::array<std::array<float,3>,5>& p_couplers) {
    NdArray<float> density(wl.size(), p_couplers.size());
    for (size_t i = 0; i < p_couplers.size(); ++i) {
        const auto& ps = p_couplers[i];
        for (size_t j = 0; j < wl.size(); ++j) {
            float x = wl[j];
            density(j, i) = ps[0] * std::exp(-(x - ps[2]) * (x - ps[2]) / (2.0f * ps[1] * ps[1]));
        }
    }
    return density;
}

DensityMidMinResult density_mid_min_model(const ReconstructParams& params,
                                          const NdArray<float>& wl,
                                          const NdArray<float>& cmy_model,
                                          const std::string& model) {
    DensityMidMinResult res;
    auto dye = shift_stretch_cmy(wl, cmy_model,
        params.dye_amp[0], params.dye_width[0], params.dye_shift[0],
        params.dye_amp[1], params.dye_width[1], params.dye_shift[1],
        params.dye_amp[2], params.dye_width[2], params.dye_shift[2]);

    NdArray<float> filters(wl.size(), 3);
    filters.fill(1.0f);

    if (model == "dmid_dmin") {
        int ch[5] = {0, 0, 1, 1, 2};
        std::array<std::array<float,3>,5> p{};
        for (int i = 0; i < 5; ++i) p[i] = {params.cpl_amp[i], params.cpl_width[i], params.cpl_max[i]};
        auto cpl = gaussian_profiles(wl, p);
        NdArray<float> cpl_cmy(wl.size(), 3); cpl_cmy.zeros();
        for (int i = 0; i < 5; ++i) {
            for (size_t j = 0; j < wl.size(); ++j) cpl_cmy(j, ch[i]) += cpl(j, i);
        }
        res.cmy = dye - cpl_cmy;

        // dmin model
        NdArray<float> cpl_cmy_scaled = cpl.copy();
        for (int i = 0; i < 5; ++i) {
            int ch_i = ch[i];
            for (size_t j = 0; j < wl.size(); ++j) {
                cpl_cmy_scaled(j, i) = cpl(j, i) / params.dye_amp[ch_i] * params.dmax[ch_i];
            }
        }
        NdArray<float> cpl_sum(wl.size(), 1); cpl_sum.zeros();
        for (size_t j = 0; j < wl.size(); ++j) {
            float s = 0.f;
            for (int i = 0; i < 5; ++i) s += cpl_cmy_scaled(j, i);
            cpl_sum[j] = s;
        }
        NdArray<float> fog_cmy = res.cmy.copy();
        for (int k = 0; k < 3; ++k) {
            for (size_t j = 0; j < wl.size(); ++j) fog_cmy(j, k) *= params.fog[k];
        }
        NdArray<float> fog_sum(wl.size(), 1); fog_sum.zeros();
        for (size_t j = 0; j < wl.size(); ++j) {
            fog_sum[j] = fog_cmy(j,0) + fog_cmy(j,1) + fog_cmy(j,2);
        }
        NdArray<float> scattering = wl.copy();
        for (size_t j = 0; j < wl.size(); ++j) {
            float wl4 = wl[j] * wl[j] * wl[j] * wl[j];
            scattering[j] = -std::log10(1.0f - params.scat400 * 400.0f * 400.0f * 400.0f * 400.0f / wl4);
        }
        res.dmin = wl.copy();
        for (size_t j = 0; j < wl.size(); ++j) res.dmin[j] = cpl_sum[j] + fog_sum[j] + scattering[j] + params.base;
        res.filters = filters; // not used in this model for plotting
        res.dye = dye;
        return res;
    }

    // Default: just return dye
    res.cmy = dye;
    res.filters = filters;
    res.dye = dye;
    res.dmin = NdArray<float>(wl.size(), 1); res.dmin.zeros();
    return res;
}

NdArray<float> compute_densitometer_crosstalk_matrix(const NdArray<float>& densitometer_intensity,
                                                     const NdArray<float>& dye_density) {
    NdArray<float> cm(3, 3);
    // dye_transmittance = 10 ** (-dye_density)
    NdArray<float> dye_transmittance = dye_density.copy();
    for (size_t i = 0; i < dye_transmittance.size(); ++i) {
        dye_transmittance[i] = std::pow(10.0f, -dye_transmittance[i]);
    }
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double num = 0.0, den = 0.0;
            for (size_t k = 0; k < densitometer_intensity.shape().rows; ++k) {
                num += static_cast<double>(densitometer_intensity(k, i) * dye_transmittance(k, j));
                den += static_cast<double>(densitometer_intensity(k, i));
            }
            cm(i, j) = -std::log10(static_cast<float>(num / den));
        }
    }
    return cm;
}

NdArray<float> slopes_of_concentrations(const NdArray<float>& log_exposure,
                                        const NdArray<float>& density_curves,
                                        const NdArray<float>& dstm_cm) {
    (void)dstm_cm; // not used in simplified slope calculation
    return agx::utils::measure_slopes_at_exposure(log_exposure, density_curves);
}

Profile reconstruct_dye_density(Profile profile, const ReconstructParams& params, bool print_params) {
    const auto& cmy_model = profile.data.dye_density(nc::Slice(0, profile.data.dye_density.shape().rows), nc::Slice(0,3));
    const auto& wl = profile.data.wavelengths.flatten();
    const auto& le = profile.data.log_exposure.flatten();
    const auto& dc = profile.data.density_curves;
    (void)le; (void)dc; // used for slopes if needed

    auto result = density_mid_min_model(params, wl, cmy_model, "dmid_dmin");

    auto dstm = agx::utils::load_densitometer_data(profile.info.densitometer);
    auto dstm_cm = compute_densitometer_crosstalk_matrix(dstm, result.cmy);
    auto g = slopes_of_concentrations(le, dc, dstm_cm);

    if (print_params) {
        std::cout << "Reconstructed Dye Density Parameters (C++)" << std::endl;
        std::cout << "Slopes of conc. at le=0: " << g << std::endl;
        std::cout << "Crosstalk matrix:\n" << dstm_cm << std::endl;
    }

    // Normalize and assign back as in Python
    NdArray<float> max_cmy(1,3);
    for (int k = 0; k < 3; ++k) {
        float m = 0.f;
        for (size_t i = 0; i < wl.size(); ++i) m = std::max(m, result.cmy(i, k));
        max_cmy[k] = m;
    }
    profile.info.density_midscale_neutral[0] = max_cmy[0];
    profile.info.density_midscale_neutral[1] = max_cmy[1];
    profile.info.density_midscale_neutral[2] = max_cmy[2];
    for (int k = 0; k < 3; ++k) {
        for (size_t i = 0; i < wl.size(); ++i) {
            profile.data.dye_density(i, k) = result.cmy(i, k) / (max_cmy[k] == 0.f ? 1.f : max_cmy[k]);
        }
    }
    return profile;
}

} // namespace profiles
} // namespace agx


