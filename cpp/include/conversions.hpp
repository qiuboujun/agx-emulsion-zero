#pragma once

#include "NumCpp.hpp"
#include <string>

namespace agx { namespace utils {

nc::NdArray<float> density_to_light(const nc::NdArray<float>& density, const nc::NdArray<float>& light);
// GPU-accelerated version with CPU fallback. Returns true if GPU path executed, else false.
bool density_to_light_gpu(const nc::NdArray<float>& density, const nc::NdArray<float>& light, nc::NdArray<float>& out);

// Enforced CUDA variant (no CPU fallback)
bool density_to_light_cuda(const nc::NdArray<float>& density, const nc::NdArray<float>& light, nc::NdArray<float>& out);

// GPU-accelerated blocked dot: A is H x (W*K), B is K x 3, out is H x (W*3)
// Returns true if GPU path executed; else false and 'out' is untouched.
bool dot_blocks_K3_gpu(const nc::NdArray<float>& A,
                       const nc::NdArray<float>& B,
                       int W,
                       nc::NdArray<float>& out);
float density_to_light(float density, float light);

nc::NdArray<float> compute_densitometer_correction(const nc::NdArray<float>& dye_density, const std::string& type = "status_A");

nc::NdArray<float> compute_aces_conversion_matrix(const nc::NdArray<float>& sensitivity, const nc::NdArray<float>& illuminant);

std::pair<nc::NdArray<float>, nc::NdArray<float>> rgb_to_raw_aces_idt(
    const nc::NdArray<float>& RGB,
    const nc::NdArray<float>& illuminant,
    const nc::NdArray<float>& sensitivity,
    nc::NdArray<float> midgray_rgb = nc::NdArray<float>(),
    const std::string& color_space = "sRGB",
    bool apply_cctf_decoding = true,
    nc::NdArray<float> aces_conversion_matrix = nc::NdArray<float>());

// Add deterministic glare to XYZ by adding a fixed percentage of illuminant XYZ.
// xyz: H x (W*3) image in XYZ, illuminant_xyz: 1x3 XYZ of the illuminant
// percent: [0..1] fraction of illuminant added uniformly (not random), for parity unit tests
nc::NdArray<float> add_glare(const nc::NdArray<float>& xyz,
                             const nc::NdArray<float>& illuminant_xyz,
                             float percent);

// Add stochastic glare: generates a lognormal random glare field with given mean `percent`
// and std `roughness*percent` (percent in [0..1]), blurs it with Gaussian sigma (pixels),
// then adds glare_amount * illuminant_xyz per pixel to XYZ.
// Uses a fixed RNG seed for determinism in tests.
nc::NdArray<float> add_random_glare(const nc::NdArray<float>& xyz,
                                    const nc::NdArray<float>& illuminant_xyz,
                                    float percent,
                                    float roughness,
                                    float blur_sigma_px,
                                    int height,
                                    int width,
                                    unsigned int seed = 12345u);

} } // namespace agx::utils
