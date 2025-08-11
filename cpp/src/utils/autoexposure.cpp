// SPDX-License-Identifier: MIT

#include "autoexposure.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace agx {
namespace utils {

static inline float srgb_inverse_eotf(float v) {
    // sRGB decoding (approximate)
    if (v <= 0.04045f) return v / 12.92f;
    return std::pow((v + 0.055f) / 1.055f, 2.4f);
}

static nc::NdArray<float> rgb_to_xyz(const nc::NdArray<float>& image_hwc, bool decode) {
    // sRGB D65 to XYZ matrix
    const float M[3][3] = {
        {0.4124564f, 0.3575761f, 0.1804375f},
        {0.2126729f, 0.7151522f, 0.0721750f},
        {0.0193339f, 0.1191920f, 0.9503041f}
    };
    const int H = static_cast<int>(image_hwc.shape().rows);
    const int W = static_cast<int>(image_hwc.shape().cols);
    nc::NdArray<float> xyz(H, W * 3);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) {
            float R = image_hwc(y, x * 3 + 0);
            float G = image_hwc(y, x * 3 + 1);
            float B = image_hwc(y, x * 3 + 2);
            if (decode) {
                R = srgb_inverse_eotf(R);
                G = srgb_inverse_eotf(G);
                B = srgb_inverse_eotf(B);
            }
            float X = M[0][0] * R + M[0][1] * G + M[0][2] * B;
            float Y = M[1][0] * R + M[1][1] * G + M[1][2] * B;
            float Z = M[2][0] * R + M[2][1] * G + M[2][2] * B;
            xyz(y, x * 3 + 0) = X;
            xyz(y, x * 3 + 1) = Y;
            xyz(y, x * 3 + 2) = Z;
        }
    }
    return xyz;
}

static float compute_center_weighted_Y(const nc::NdArray<float>& image_Y) {
    const int H = static_cast<int>(image_Y.shape().rows);
    const int W = static_cast<int>(image_Y.shape().cols);
    // Normalized shape relative to max(H, W)
    const float smax = static_cast<float>(std::max(H, W));
    const float sy = static_cast<float>(H) / smax;
    const float sx = static_cast<float>(W) / smax;
    const float sigma = 0.2f;

    double sum_w = 0.0;
    double sum_yw = 0.0;
    for (int y = 0; y < H; ++y) {
        float yy = (static_cast<float>(y) / W) - 0.5f; // base on width? Match Python: y / H
        yy = (static_cast<float>(y) / H) - 0.5f;
        yy *= sy;
        for (int x = 0; x < W; ++x) {
            float xx = (static_cast<float>(x) / W) - 0.5f;
            xx *= sx;
            float w = std::exp(-(xx * xx + yy * yy) / (2.0f * sigma * sigma));
            sum_w += w;
            sum_yw += static_cast<double>(image_Y(y, x)) * w;
        }
    }
    if (sum_w <= 0.0) return 0.0f;
    return static_cast<float>(sum_yw / sum_w);
}

float measure_autoexposure_ev(
    const nc::NdArray<float>& image_hwc, bool apply_cctf_decoding, const std::string& method) {
    // reshape to (H, W, 3)
    const int H = static_cast<int>(image_hwc.shape().rows);
    const int W3 = static_cast<int>(image_hwc.shape().cols);
    if (W3 % 3 != 0) throw std::runtime_error("image_hwc must have 3 channels packed in last dim");
    const int W = W3 / 3;
    auto xyz = rgb_to_xyz(image_hwc, apply_cctf_decoding);
    // Extract Y
    nc::NdArray<float> Y(H, W);
    for (int y = 0; y < H; ++y) {
        for (int x = 0; x < W; ++x) Y(y, x) = xyz(y, x * 3 + 1);
    }

    float Y_exposure = 0.0f;
    if (method == "median") {
        // Flatten and median (nth_element)
        std::vector<float> vals; vals.reserve(H * W);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) vals.push_back(Y(y, x));
        auto mid = vals.begin() + vals.size() / 2;
        std::nth_element(vals.begin(), mid, vals.end());
        Y_exposure = *mid;
    } else {
        // center_weighted
        Y_exposure = compute_center_weighted_Y(Y);
    }

    float exposure = Y_exposure / 0.184f;
    float ev = 0.0f;
    if (exposure > 0.0f) ev = -std::log2(exposure);
    return ev;
}

bool measure_autoexposure_ev_cuda_center_weighted(
    const nc::NdArray<float>& image_hwc, bool apply_cctf_decoding, float& out_ev) {
    // CUDA kernel is implemented in autoexposure.cu; default to CPU for now.
    (void)image_hwc; (void)apply_cctf_decoding; (void)out_ev;
    return false;
}

float measure_autoexposure_ev_auto(
    const nc::NdArray<float>& image_hwc,
    bool apply_cctf_decoding,
    const std::string& method) {
    if (method == "center_weighted") {
        float ev;
        if (measure_autoexposure_ev_cuda_center_weighted(image_hwc, apply_cctf_decoding, ev)) return ev;
    }
    return measure_autoexposure_ev(image_hwc, apply_cctf_decoding, method);
}

} // namespace utils
} // namespace agx


