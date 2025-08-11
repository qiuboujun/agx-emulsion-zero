// SPDX-License-Identifier: MIT

#pragma once

#include "NumCpp.hpp"
#include <string>

namespace agx {
namespace utils {

// CPU implementation
float measure_autoexposure_ev(
    const nc::NdArray<float>& image_hwc,                  // shape (H, W, 3)
    bool apply_cctf_decoding = true,
    const std::string& method = "center_weighted");

// CUDA accelerated center-weighted path (returns true on success and fills ev)
bool measure_autoexposure_ev_cuda_center_weighted(
    const nc::NdArray<float>& image_hwc,                  // shape (H, W, 3)
    bool apply_cctf_decoding,
    float& out_ev);

// Wrapper: uses CUDA for center-weighted when available, otherwise CPU
float measure_autoexposure_ev_auto(
    const nc::NdArray<float>& image_hwc,
    bool apply_cctf_decoding = true,
    const std::string& method = "center_weighted");

} // namespace utils
} // namespace agx


