// SPDX-License-Identifier: MIT

#include "crop_resize.hpp"
#include <algorithm>
#include <cmath>

namespace agx {
namespace utils {

nc::NdArray<float> crop_image(
    const nc::NdArray<float>& image_hwc,
    const std::array<float, 2>& center,
    const std::array<float, 2>& size_fraction) {
    const int H = (int)image_hwc.shape().rows;
    const int W3 = (int)image_hwc.shape().cols;
    if (W3 % 3 != 0) throw std::runtime_error("image_hwc last dim not multiple of 3");
    const int W = W3 / 3;

    // Python flips center to (y, x)
    float cy = center[1];
    float cx = center[0];

    // shape array
    float shape0 = (float)H, shape1 = (float)W;
    float cn0 = std::round(shape0 * cy);
    float cn1 = std::round(shape1 * cx);

    // sz fraction based on long side, flip size vector to (y, x)
    float longside = std::max(shape0, shape1);
    float sz0 = std::round(longside * size_fraction[1]);
    float sz1 = std::round(longside * size_fraction[0]);

    float x0_0 = std::round(cn0 - sz0 / 2.0f);
    float x0_1 = std::round(cn1 - sz1 / 2.0f);

    int sz0_i = (int)sz0, sz1_i = (int)sz1;
    int x0_0_i = (int)x0_0, x0_1_i = (int)x0_1;
    if (x0_0_i < 0) x0_0_i = 0;
    if (x0_1_i < 0) x0_1_i = 0;
    if (x0_0_i + sz0_i > H) x0_0_i = H - sz0_i;
    if (x0_1_i + sz1_i > W) x0_1_i = W - sz1_i;
    if (x0_0_i < 0) { x0_0_i = 0; sz0_i = 0; }
    if (x0_1_i < 0) { x0_1_i = 0; sz1_i = 0; }

    nc::NdArray<float> out(sz0_i, sz1_i * 3);
    for (int y = 0; y < sz0_i; ++y) {
        for (int x = 0; x < sz1_i; ++x) {
            for (int c = 0; c < 3; ++c) out(y, x * 3 + c) = image_hwc(x0_0_i + y, (x0_1_i + x) * 3 + c);
        }
    }
    return out;
}

static inline float lerp(float a, float b, float t) { return a + (b - a) * t; }

nc::NdArray<float> resize_image_bilinear(
    const nc::NdArray<float>& image_hwc,
    int newH, int newW) {
    const int H = (int)image_hwc.shape().rows;
    const int W3 = (int)image_hwc.shape().cols;
    if (W3 % 3 != 0) throw std::runtime_error("image_hwc last dim not multiple of 3");
    const int W = W3 / 3;
    if (newH <= 0 || newW <= 0) return nc::NdArray<float>(0);

    nc::NdArray<float> out(newH, newW * 3);
    const float scaleY = (float)H / (float)newH;
    const float scaleX = (float)W / (float)newW;
    for (int y = 0; y < newH; ++y) {
        float srcY = (y + 0.5f) * scaleY - 0.5f;
        int y0 = (int)std::floor(srcY);
        int y1 = std::min(y0 + 1, H - 1);
        float ty = srcY - y0;
        if (y0 < 0) { y0 = 0; ty = 0.0f; }
        for (int x = 0; x < newW; ++x) {
            float srcX = (x + 0.5f) * scaleX - 0.5f;
            int x0 = (int)std::floor(srcX);
            int x1 = std::min(x0 + 1, W - 1);
            float tx = srcX - x0;
            if (x0 < 0) { x0 = 0; tx = 0.0f; }
            for (int c = 0; c < 3; ++c) {
                float p00 = image_hwc(y0, x0 * 3 + c);
                float p01 = image_hwc(y0, x1 * 3 + c);
                float p10 = image_hwc(y1, x0 * 3 + c);
                float p11 = image_hwc(y1, x1 * 3 + c);
                float a = lerp(p00, p01, tx);
                float b = lerp(p10, p11, tx);
                out(y, x * 3 + c) = lerp(a, b, ty);
            }
        }
    }
    return out;
}

bool resize_image_bilinear_cuda(
    const nc::NdArray<float>& image_hwc, int newH, int newW, nc::NdArray<float>& out_hwc) {
    (void)image_hwc; (void)newH; (void)newW; (void)out_hwc;
    return false;
}

nc::NdArray<float> resize_image_bilinear_auto(
    const nc::NdArray<float>& image_hwc, int newH, int newW, bool use_cuda) {
    if (use_cuda) {
        nc::NdArray<float> out;
        if (resize_image_bilinear_cuda(image_hwc, newH, newW, out)) return out;
    }
    return resize_image_bilinear(image_hwc, newH, newW);
}

} // namespace utils
} // namespace agx


