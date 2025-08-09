// SPDX-License-Identifier: MIT

#include "balance.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>
#include <vector>

#include "NumCpp.hpp"
#include "colour.hpp"
#include "config.hpp"
#include "illuminants.hpp"
#include "scipy.hpp"

namespace agx {
namespace profiles {

// Helpers
static nc::NdArray<float> interp1(const nc::NdArray<float> &x,
                                   const nc::NdArray<float> &xp,
                                   const nc::NdArray<float> &fp) {
    // Linear interpolation of fp(xp) at x
    if (xp.size() != fp.size()) {
        throw std::invalid_argument("interp1: xp and fp must have same size");
    }
    // Ensure xp is increasing (copy and sort indices if needed)
    std::vector<std::pair<float, float>> pairs;
    pairs.reserve(xp.size());
    for (std::size_t i = 0; i < xp.size(); ++i) pairs.push_back({xp[i], fp[i]});
    std::sort(pairs.begin(), pairs.end(), [](auto &a, auto &b) { return a.first < b.first; });
    nc::NdArray<float> xp_sorted(1, xp.size());
    nc::NdArray<float> fp_sorted(1, fp.size());
    for (std::size_t i = 0; i < pairs.size(); ++i) {
        xp_sorted[i] = pairs[i].first;
        fp_sorted[i] = pairs[i].second;
    }

    nc::NdArray<float> y(1, x.size());
    for (std::size_t k = 0; k < x.size(); ++k) {
        float xv = x[k];
        // clamp to bounds
        if (xv <= xp_sorted.front()) {
            y[k] = fp_sorted.front();
            continue;
        }
        if (xv >= xp_sorted.back()) {
            y[k] = fp_sorted.back();
            continue;
        }
        // find interval
        auto it = std::upper_bound(xp_sorted.begin(), xp_sorted.end(), xv);
        std::size_t i = static_cast<std::size_t>(it - xp_sorted.begin()) - 1;
        float x0 = xp_sorted[i], x1 = xp_sorted[i + 1];
        float y0 = fp_sorted[i], y1 = fp_sorted[i + 1];
        float t = (xv - x0) / (x1 - x0 + 1e-20f);
        y[k] = y0 + t * (y1 - y0);
    }
    return y.reshape(x.shape());
}

void balance_sensitivity(AgxEmulsionData &data,
                         const std::string &reference_illuminant,
                         bool correct_log_exposure) {
    // Python: s = 10**ls; neutral_exposures = sum(ill[:,None]*s, axis=0);
    // corr = neutral_exposures[1]/neutral_exposures; ls += log10(corr)
    const auto ill = agx::model::standard_illuminant(reference_illuminant);

    // s = 10 ** ls (elementwise)
    nc::NdArray<float> s(data.log_sensitivity.shape());
    for (std::size_t i = 0; i < data.log_sensitivity.shape().rows; ++i) {
        for (int c = 0; c < 3; ++c) {
            s(i, c) = std::pow(10.0f, data.log_sensitivity(i, c));
        }
    }
    nc::NdArray<float> neutral_exposures(1, 3);
    for (int c = 0; c < 3; ++c) {
        float sum = 0.0f;
        for (std::size_t i = 0; i < ill.size(); ++i) {
            float sv = s(i, c);
            if (!std::isnan(sv)) {
                sum += ill[i] * sv;
            }
        }
        neutral_exposures[c] = sum;
    }
    std::array<float, 3> corr = {
        neutral_exposures[1] / (neutral_exposures[0] + 1e-20f),
        1.0f,
        neutral_exposures[1] / (neutral_exposures[2] + 1e-20f)};

    // apply to sensitivities
    for (std::size_t i = 0; i < s.shape().rows; ++i) {
        for (int c = 0; c < 3; ++c) s(i, c) *= corr[c];
    }
    // back to log10
    nc::NdArray<float> ls_out(s.shape().rows, s.shape().cols);
    for (std::size_t i = 0; i < s.shape().rows; ++i) {
        for (int c = 0; c < 3; ++c) {
            float sv = s(i, c);
            if (std::isnan(sv)) {
                ls_out(i, c) = std::numeric_limits<float>::quiet_NaN();
            } else {
                ls_out(i, c) = std::log10(std::max(1e-20f, sv));
            }
        }
    }
    data.log_sensitivity = ls_out;

    if (correct_log_exposure) {
        // shift density curves in log exposure by log10(corr[c]) per channel
        nc::NdArray<float> dc_out = data.density_curves.copy();
        for (int c = 0; c < 3; ++c) {
            // new_curve(le) = interp(original_le, original_curve)
            // Python used np.interp(le, le+shift, curve)
            float shift = std::log10(corr[c]);
            nc::NdArray<float> xp = data.log_exposure + shift;
            nc::NdArray<float> fp = data.density_curves(nc::Slice(0, data.density_curves.shape().rows), c);
            auto col = interp1(data.log_exposure, xp, fp);
            for (std::size_t i = 0; i < col.size(); ++i) dc_out(i, c) = col[i];
        }
        data.density_curves = dc_out;
    }
}

void balance_density(AgxEmulsionData &data) {
    // Align M and Y so that their density at le=0 equals green's density at le=0
    const auto &le = data.log_exposure;
    const auto &dc = data.density_curves;
    // density_0 = interp(0, le, G)
    nc::NdArray<float> xq = nc::zeros<float>(1, 1);
    xq[0] = 0.0f;
    auto density0_arr = interp1(xq, le, dc(nc::Slice(0, dc.shape().rows), 1));
    float density_0 = density0_arr[0];

    // find log exposure shifts for M and Y so their curves hit density_0
    // Invert curves (density->le) via monotonic cubic interp.
    auto invert_to_le = [&](int channel, float target_density) -> float {
        // Build (density, le) pairs; drop NaNs; sort by density; enforce strictly increasing
        std::vector<std::pair<double, double>> pairs;
        pairs.reserve(dc.shape().rows);
        for (std::size_t i = 0; i < dc.shape().rows; ++i) {
            double dens = static_cast<double>(dc(i, channel));
            double le_i = static_cast<double>(le[i]);
            if (!std::isnan(dens) && !std::isnan(le_i)) {
                pairs.emplace_back(dens, le_i);
            }
        }
        if (pairs.size() < 2) return 0.0f;
        std::sort(pairs.begin(), pairs.end(), [](auto &a, auto &b) { return a.first < b.first; });
        // Deduplicate and ensure strict monotonic increase on density
        std::vector<double> dens_u;
        std::vector<double> le_u;
        dens_u.reserve(pairs.size());
        le_u.reserve(pairs.size());
        const double eps = 1e-9;
        for (std::size_t i = 0; i < pairs.size(); ++i) {
            double xval = pairs[i].first;
            double yval = pairs[i].second;
            if (!dens_u.empty()) {
                if (xval <= dens_u.back()) {
                    xval = dens_u.back() + eps; // enforce strictly increasing
                }
            }
            dens_u.push_back(xval);
            le_u.push_back(yval);
        }
        nc::NdArray<double> x_dens(dens_u);
        nc::NdArray<double> y_le(le_u);
        scipy::interpolate::interp1d inv(x_dens, y_le, scipy::interpolate::interp1d::Kind::Cubic);
        return static_cast<float>(inv(static_cast<double>(target_density)));
    };
    float le_shift_m = invert_to_le(0, density_0);
    float le_shift_y = invert_to_le(2, density_0);

    // shift M and Y curves by interpolating at le - shift
    auto shift_channel = [&](int channel, float shift) {
        nc::NdArray<float> xp = le - shift;
        nc::NdArray<float> fp = dc(nc::Slice(0, dc.shape().rows), channel);
        auto col = interp1(le, xp, fp);
        for (std::size_t i = 0; i < col.size(); ++i) data.density_curves(i, channel) = col[i];
    };
    shift_channel(0, le_shift_m);
    shift_channel(2, le_shift_y);

    // update log_sensitivity offsets
    for (std::size_t i = 0; i < data.log_sensitivity.shape().rows; ++i) {
        data.log_sensitivity(i, 0) -= le_shift_m;
        data.log_sensitivity(i, 1) -= 0.0f;
        data.log_sensitivity(i, 2) -= le_shift_y;
    }
}

static nc::NdArray<float> compute_rgb_from_mid(const nc::NdArray<float> &mid,
                                               const nc::NdArray<float> &illuminant,
                                               const std::string &color_space = "sRGB") {
    // light = 10^(-mid) * illuminant; NaNs in light are set to 0 (match Python)
    nc::NdArray<float> light(mid.shape().rows, mid.shape().cols);
    for (std::size_t i = 0; i < mid.size(); ++i) {
        float m = mid[i];
        if (std::isnan(m)) {
            light[i] = 0.0f;
        } else {
            light[i] = std::pow(10.0f, -m) * illuminant[i];
        }
    }

    // XYZ via CMFs normalized by sum(ill * y_bar)
    const auto &cmfs = agx::config::STANDARD_OBSERVER_CMFS;
    float normalization = 0.0f;
    for (std::size_t i = 0; i < illuminant.size(); ++i) normalization += illuminant[i] * cmfs(i, 1);
    normalization = (normalization == 0.0f) ? 1.0f : normalization;

    // accumulate XYZ (1x3)
    nc::NdArray<float> XYZ(1, 3);
    XYZ.fill(0.0f);
    for (std::size_t i = 0; i < illuminant.size(); ++i) {
        float L = light[i];
        if (!std::isnan(L)) {
            XYZ(0, 0) += L * cmfs(i, 0);
            XYZ(0, 1) += L * cmfs(i, 1);
            XYZ(0, 2) += L * cmfs(i, 2);
        }
    }
    XYZ = XYZ / normalization;

    // compute illuminant xy for adaptation
    nc::NdArray<float> illum_XYZ(1, 3);
    illum_XYZ.fill(0.0f);
    for (std::size_t i = 0; i < illuminant.size(); ++i) {
        illum_XYZ(0, 0) += illuminant[i] * cmfs(i, 0);
        illum_XYZ(0, 1) += illuminant[i] * cmfs(i, 1);
        illum_XYZ(0, 2) += illuminant[i] * cmfs(i, 2);
    }
    illum_XYZ = illum_XYZ / normalization;
    auto illum_xy = colour::XYZ_to_xy(illum_XYZ);

    // XYZ -> RGB (no encoding)
    auto rgb = colour::XYZ_to_RGB(XYZ, color_space, /*apply_cctf_encoding*/ false, illum_xy, "Bradford");
    return rgb;
}

static BalanceResultMetameric balance_metameric_core(const nc::NdArray<float> &dye_density,
                                                     const nc::NdArray<float> &illuminant,
                                                     float midgray_value) {
    if (dye_density.shape().cols < 4) {
        throw std::invalid_argument("balance_metameric_neutral: dye_density must have >=4 columns");
    }

    auto midscale_neutral = [&](const std::array<float, 3> &density_cmy) {
        // mid = sum(dye_density[:,:3] * d_cmy) + dye_density[:,3]
        nc::NdArray<float> mid(dye_density.shape().rows, 1);
        for (std::size_t i = 0; i < dye_density.shape().rows; ++i) {
            float v = 0.0f;
            v += dye_density(i, 0) * density_cmy[0];
            v += dye_density(i, 1) * density_cmy[1];
            v += dye_density(i, 2) * density_cmy[2];
            v += dye_density(i, 3);
            mid[i] = v;
        }
        return mid;
    };

    const std::array<float, 3> target_rgb = {midgray_value, midgray_value, midgray_value};

    auto objective = [&](const std::array<float, 3> &params) -> std::array<float, 3> {
        auto mid = midscale_neutral(params);
        auto rgb = compute_rgb_from_mid(mid, illuminant, "sRGB");
        std::array<float, 3> res = {target_rgb[0] - rgb(0, 0), target_rgb[1] - rgb(0, 1), target_rgb[2] - rgb(0, 2)};
        return res;
    };

    // Levenberg-Marquardt style least-squares with small bounds (x>0)
    std::array<float, 3> x = {1.0f, 1.0f, 1.0f};
    float lambda = 1e-2f;
    for (int iter = 0; iter < 50; ++iter) {
        auto r = objective(x);
        float err = std::abs(r[0]) + std::abs(r[1]) + std::abs(r[2]);
        if (err < 1e-8f) break;
        // finite-difference Jacobian J (3x3): rows residuals, cols params
        float h = 1e-3f;
        float J[3][3] = {};
        for (int j = 0; j < 3; ++j) {
            std::array<float, 3> xh = x; xh[j] += h;
            auto rh = objective(xh);
            J[0][j] = (rh[0] - r[0]) / h;
            J[1][j] = (rh[1] - r[1]) / h;
            J[2][j] = (rh[2] - r[2]) / h;
        }
        float JTJ[3][3] = {};
        float JTr[3] = {};
        for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
                for (int k = 0; k < 3; ++k) JTJ[a][b] += J[k][a] * J[k][b];
            }
            for (int k = 0; k < 3; ++k) JTr[a] += J[k][a] * r[k];
        }
        // Solve (JTJ + lambda*I) dx = -JTr  (LM damping: descend along negative gradient)
        for (int i = 0; i < 3; ++i) JTJ[i][i] += lambda;
        // 3x3 solve via Cramer's rule
        auto det3 = [&](float M[3][3]) {
            return M[0][0]*(M[1][1]*M[2][2]-M[1][2]*M[2][1]) - M[0][1]*(M[1][0]*M[2][2]-M[1][2]*M[2][0]) + M[0][2]*(M[1][0]*M[2][1]-M[1][1]*M[2][0]);
        };
        float A[3][3]; for (int i=0;i<3;++i)for(int j=0;j<3;++j)A[i][j]=JTJ[i][j];
        float detA = det3(A);
        if (std::abs(detA) < 1e-12f) break;
        float dx[3];
        for (int k = 0; k < 3; ++k) {
            float Ak[3][3]; for(int i=0;i<3;++i)for(int j=0;j<3;++j)Ak[i][j]=A[i][j];
            Ak[0][k] = -JTr[0]; Ak[1][k] = -JTr[1]; Ak[2][k] = -JTr[2];
            dx[k] = det3(Ak) / detA;
        }
        std::array<float,3> x_new = {x[0] + dx[0], x[1] + dx[1], x[2] + dx[2]};
        auto r_new = objective(x_new);
        float err_new = std::abs(r_new[0]) + std::abs(r_new[1]) + std::abs(r_new[2]);
        if (err_new < err) {
            x = x_new;
            lambda = std::max(1e-6f, lambda * 0.5f);
        } else {
            lambda = std::min(1e3f, lambda * 2.0f);
        }
    }

    BalanceResultMetameric out;
    out.d_cmy_metameric = x;
    out.d_cmy_scale = {x[0] / x[1], 1.0f, x[2] / x[1]};

    // build updated dye_density: column 4 is mid, columns 0..2 scaled
    out.dye_density_out = dye_density.copy();
    // compute mid with absolute d_cmy
    auto mid = nc::NdArray<float>(dye_density.shape().rows, 1);
    for (std::size_t i = 0; i < dye_density.shape().rows; ++i) {
        mid[i] = dye_density(i, 0) * x[0] + dye_density(i, 1) * x[1] + dye_density(i, 2) * x[2] + dye_density(i, 3);
    }
    // ensure column 4 exists; if not, expand to 5 columns
    if (out.dye_density_out.shape().cols < 5) {
        nc::NdArray<float> dd_new(dye_density.shape().rows, 5);
        for (std::size_t i = 0; i < dye_density.shape().rows; ++i) {
            dd_new(i, 0) = dye_density(i, 0);
            dd_new(i, 1) = dye_density(i, 1);
            dd_new(i, 2) = dye_density(i, 2);
            dd_new(i, 3) = dye_density(i, 3);
            dd_new(i, 4) = mid[i];
        }
        out.dye_density_out = dd_new;
    } else {
        for (std::size_t i = 0; i < out.dye_density_out.shape().rows; ++i) out.dye_density_out(i, 4) = mid[i];
    }
    // scale first three columns
    for (std::size_t i = 0; i < out.dye_density_out.shape().rows; ++i) {
        out.dye_density_out(i, 0) *= out.d_cmy_scale[0];
        out.dye_density_out(i, 1) *= out.d_cmy_scale[1];
        out.dye_density_out(i, 2) *= out.d_cmy_scale[2];
    }

    return out;
}

BalanceResultMetameric balance_metameric_neutral(const nc::NdArray<float> &dye_density,
                                                 const std::string &viewing_illuminant,
                                                 float midgray_value) {
    const auto illuminant = agx::model::standard_illuminant(viewing_illuminant);
    return balance_metameric_core(dye_density, illuminant, midgray_value);
}

BalanceResultMetameric balance_metameric_neutral_with_illuminant(
    const nc::NdArray<float> &dye_density,
    const nc::NdArray<float> &illuminant,
    float midgray_value) {
    return balance_metameric_core(dye_density, illuminant, midgray_value);
}

nc::NdArray<float> debug_rgb_from_density_params(
    const nc::NdArray<float> &dye_density,
    const std::array<float,3> &density_cmy,
    const std::string &viewing_illuminant) {
    const auto illuminant = agx::model::standard_illuminant(viewing_illuminant);
    nc::NdArray<float> mid(dye_density.shape().rows, 1);
    for (std::size_t i = 0; i < dye_density.shape().rows; ++i) {
        mid[i] = dye_density(i, 0) * density_cmy[0] + dye_density(i, 1) * density_cmy[1] + dye_density(i, 2) * density_cmy[2] + dye_density(i, 3);
    }
    return compute_rgb_from_mid(mid, illuminant, "sRGB");
}

} // namespace profiles
} // namespace agx


