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
    const int N = L * L * L;
    std::cout << "[LUT] Building 3D camera LUT: steps=" << L << ", samples=" << N << std::endl;

    // Build the full RGB grid once (N x 3)
    nc::NdArray<float> inputs(N, 3);
    int idx = 0;
    for (int r = 0; r < L; ++r) {
        for (int g = 0; g < L; ++g) {
            for (int b = 0; b < L; ++b) {
                const float rf = (L == 1) ? 0.0f : (float)r / (float)(L - 1);
                const float gf = (L == 1) ? 0.0f : (float)g / (float)(L - 1);
                const float bf = (L == 1) ? 0.0f : (float)b / (float)(L - 1);
                inputs(idx, 0) = rf;
                inputs(idx, 1) = gf;
                inputs(idx, 2) = bf;
                ++idx;
            }
        }
    }

    // Vectorized evaluation using grid path (preprojects spectra once)
    auto outputs = rgb_to_raw_hanatos2025_grid(
        inputs,
        /*height=*/N,
        /*width=*/1,
        sensitivity,
        color_space,
        apply_cctf_decoding,
        reference_illuminant,
        spectra_lut
    );

    std::cout << "[LUT] 3D camera LUT built." << std::endl;
    // outputs is (N x 3), already the flattened LUT layout we need
    return outputs;
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
    int steps,
    int height,
    int width) {
    std::cout << "[LUT] compute_with_lut: data shape " << data.shape().rows << "x" << data.shape().cols 
              << ", steps=" << steps << ", range=[" << xmin << "," << xmax << "]" << std::endl;
    
    try {
        // Create LUT by sampling provided function
        std::cout << "[LUT] Creating 3D LUT..." << std::endl;
        auto lut = _create_lut_3d(function, xmin, xmax, steps);
        std::cout << "[LUT] 3D LUT created, shape: " << lut.shape().rows << "x" << lut.shape().cols << std::endl;
        
        // Normalize input to (H*W, 3)
        nc::NdArray<float> image_hw_by3;
        if ((int)data.shape().cols == 3) {
            // Data is already (H*W, 3). Validate provided height/width
            const long long expected = static_cast<long long>(height) * static_cast<long long>(width);
            if ((long long)data.shape().rows != expected) {
                std::cout << "[LUT] WARNING: Provided height*width does not equal data rows. height="
                          << height << ", width=" << width << ", rows=" << data.shape().rows << std::endl;
                // Try to infer width if possible
                if (width <= 0 && height > 0 && (int)data.shape().rows % height == 0) {
                    width = (int)data.shape().rows / height;
                } else if (height <= 0 && width > 0 && (int)data.shape().rows % width == 0) {
                    height = (int)data.shape().rows / width;
                } else {
                    // Fallback: assume image is contiguous row-major with unknown H/W. Treat as 1 x N
                    height = (int)data.shape().rows;
                    width = 1;
                }
            }
            image_hw_by3 = data;
        } else {
            // Data is (H, W*3) format - reshape to (H*W, 3)
            height = (int)data.shape().rows;
            width = (int)data.shape().cols / 3;
            std::cout << "[LUT] Reshaping from " << height << "x" << (width*3) << " to " << (height*width) << "x3" << std::endl;
            image_hw_by3 = nc::NdArray<float>(height * width, 3);
            for (int i = 0; i < height; ++i) {
                for (int w = 0; w < width; ++w) {
                    for (int c = 0; c < 3; ++c) {
                        image_hw_by3(i*width + w, c) = data(i, w*3 + c);
                    }
                }
            }
        }

        std::cout << "[LUT] Applying LUT to image " << height << "x" << width << " (reshaped to " << (height*width) << "x3)..." << std::endl;
        
        // Use try-catch around the CUDA LUT application
        nc::NdArray<float> out_hw_by3;
        out_hw_by3 = agx::apply_lut_cubic_3d(lut, image_hw_by3, height, width);
        std::cout << "[LUT] CUDA LUT application successful" << std::endl;
        
        std::cout << "[LUT] compute_with_lut completed successfully" << std::endl;
        return {out_hw_by3, lut};
        
    } catch (const std::exception& e) {
        std::cout << "[LUT] ERROR in compute_with_lut: " << e.what() << std::endl;
        // Return empty arrays on error
        return {nc::NdArray<float>(), nc::NdArray<float>()};
    } catch (...) {
        std::cout << "[LUT] ERROR in compute_with_lut: unknown exception" << std::endl;
        // Return empty arrays on error
        return {nc::NdArray<float>(), nc::NdArray<float>()};
    }
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
