#include "NumCpp.hpp"
#include "fast_interp_lut.hpp"
#include "lut.hpp"
#include "spectral_upsampling.hpp"
#include <cmath>
// pybind headers removed to use pure C++/CUDA path only

namespace agx { namespace utils {

static inline float srgb_decode(float v){
    if (v <= 0.04045f) return v / 12.92f;
    return std::pow((v + 0.055f) / 1.055f, 2.4f);
}

nc::NdArray<float> create_lut_3d_camera_rgb_to_raw(
    int steps,
    const nc::NdArray<float>& sensitivity,
    const std::string& color_space,
    bool apply_cctf_decoding,
    const std::string& reference_illuminant,
    const nc::NdArray<float>& spectra_lut) {
    const int L = steps;
    nc::NdArray<float> lut(L*L*L, 3);
    // Pre-project spectra
    // We reuse rgb_to_raw_hanatos2025 for accuracy by sampling grid points
    for (int r=0; r<L; ++r){
        for (int g=0; g<L; ++g){
            for (int b=0; b<L; ++b){
                int idx = (r*L + g)*L + b;
                nc::NdArray<float> rgb(1,3);
                rgb(0,0) = (float)r / (L-1);
                rgb(0,1) = (float)g / (L-1);
                rgb(0,2) = (float)b / (L-1);
                auto raw = rgb_to_raw_hanatos2025(rgb, sensitivity, color_space, apply_cctf_decoding, reference_illuminant, spectra_lut);
                lut(idx,0) = raw(0,0);
                lut(idx,1) = raw(0,1);
                lut(idx,2) = raw(0,2);
            }
        }
    }
    return lut;
}

nc::NdArray<float> apply_lut_3d(const nc::NdArray<float>& lut_flat,
                                const nc::NdArray<float>& rgb_hw_by3,
                                int height,
                                int width) {
    return agx::apply_lut_cubic_3d(lut_flat, rgb_hw_by3, height, width);
}

std::pair<nc::NdArray<float>, nc::NdArray<float>> compute_camera_with_lut(
    const nc::NdArray<float>& rgb_hw_by3,
    int height,
    int width,
    const nc::NdArray<float>& sensitivity,
    const std::string& color_space,
    bool apply_cctf_decoding,
    const std::string& reference_illuminant,
    const nc::NdArray<float>& spectra_lut,
    int steps) {
    auto lut = create_lut_3d_camera_rgb_to_raw(steps, sensitivity, color_space, apply_cctf_decoding, reference_illuminant, spectra_lut);
    auto out = apply_lut_3d(lut, rgb_hw_by3, height, width);
    return {out, lut};
}

// ---------------------------------------------------------------------------------
// Generic 3D LUT helpers (parity with agx_emulsion/utils/lut.py)
// ---------------------------------------------------------------------------------

nc::NdArray<float> _create_lut_3d(
    const std::function<nc::NdArray<float>(const nc::NdArray<float>&)>& function,
    float xmin,
    float xmax,
    int steps) {
    const int L = steps;
    const int N = L * L * L;
    // Build grid of inputs in [xmin, xmax]
    nc::NdArray<float> inputs(N, 3);
    for (int r = 0; r < L; ++r) {
        for (int g = 0; g < L; ++g) {
            for (int b = 0; b < L; ++b) {
                const int idx = (r * L + g) * L + b;
                const float rf = (L == 1) ? 0.0f : (float)r / (float)(L - 1);
                const float gf = (L == 1) ? 0.0f : (float)g / (float)(L - 1);
                const float bf = (L == 1) ? 0.0f : (float)b / (float)(L - 1);
                inputs(idx, 0) = xmin + (xmax - xmin) * rf;
                inputs(idx, 1) = xmin + (xmax - xmin) * gf;
                inputs(idx, 2) = xmin + (xmax - xmin) * bf;
            }
        }
    }
    auto outputs = function(inputs); // Expect shape (N, C)
    if (outputs.shape().rows != (uint32_t)N) {
        throw std::runtime_error("_create_lut_3d: function returned unexpected number of rows");
    }
    // Flattened LUT is simply outputs (N x C)
    return outputs;
}

std::pair<nc::NdArray<float>, nc::NdArray<float>> compute_with_lut(
    const nc::NdArray<float>& data,
    const std::function<nc::NdArray<float>(const nc::NdArray<float>&)>& function,
    float xmin,
    float xmax,
    int steps) {
    // Create LUT by sampling provided function
    auto lut = _create_lut_3d(function, xmin, xmax, steps);
    // Infer height/width from data shaped (H, W*3) or (N,3)
    int height = (int)data.shape().rows;
    int width;
    nc::NdArray<float> image_hw_by3;
    if ((int)data.shape().cols == 3) {
        width = 1;
        image_hw_by3 = data;
    } else {
        width = (int)data.shape().cols / 3;
        // reshape Hx(W*3) -> (H*W) x 3
        image_hw_by3 = nc::NdArray<float>(height * width, 3);
        for (int i = 0; i < height; ++i) {
            for (int w = 0; w < width; ++w) {
                for (int c = 0; c < 3; ++c) {
                    image_hw_by3(i*width + w, c) = data(i, w*3 + c);
                }
            }
        }
    }
    auto out_hw_by3 = agx::apply_lut_cubic_3d(lut, image_hw_by3, height, width);
    return {out_hw_by3, lut};
}

void warmup_luts() {
    // Build a tiny synthetic LUT and run a tiny application pass to warm up GPU JITs
    int steps = 8;
    auto lut = _create_lut_3d(
        [](const nc::NdArray<float>& X){
            nc::NdArray<float> Y(X.shape().rows, 3);
            for (uint32_t i=0;i<X.shape().rows;++i){
                float r=X(i,0), g=X(i,1), b=X(i,2);
                Y(i,0) = 3*g + r;
                Y(i,1) = 3*b + g;
                Y(i,2) = 3*r + b;
            }
            return Y;
        }, 0.0f, 1.0f, steps);
    int H=16,W=16;
    nc::NdArray<float> img(H*W,3);
    for(int i=0;i<H*W;++i){ img(i,0)=(float)(i%W)/(W-1); img(i,1)=(float)(i/W)/(H-1); img(i,2)=0.5f; }
    (void)agx::apply_lut_cubic_3d(lut, img, H, W);
}

} } // namespace agx::utils
