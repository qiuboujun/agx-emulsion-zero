// SPDX-License-Identifier: MIT

#pragma once

#include "NumCpp.hpp"
#include <array>

namespace agx {
namespace utils {

// CPU crop: image (H, W, 3) in row-major packed channels in last dim.
nc::NdArray<float> crop_image(
    const nc::NdArray<float>& image_hwc,
    const std::array<float, 2>& center = {0.5f, 0.5f},
    const std::array<float, 2>& size_fraction = {0.1f, 0.1f});

// CPU resize (bilinear): output size (newH, newW)
nc::NdArray<float> resize_image_bilinear(
    const nc::NdArray<float>& image_hwc,
    int newH, int newW);

// CUDA resize version; returns true on success and fills out
bool resize_image_bilinear_cuda(
    const nc::NdArray<float>& image_hwc,
    int newH, int newW,
    nc::NdArray<float>& out_hwc);

// Wrapper: try CUDA then CPU
nc::NdArray<float> resize_image_bilinear_auto(
    const nc::NdArray<float>& image_hwc,
    int newH, int newW,
    bool use_cuda = true);

} // namespace utils
} // namespace agx


