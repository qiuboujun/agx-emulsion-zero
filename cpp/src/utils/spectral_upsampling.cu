#include "spectral_upsampling.hpp"
#include "fast_interp_lut.hpp" // Assumed header for apply_lut_cubic_2d from your project

#include <fstream>
#include <stdexcept>
#include <iostream>

// CUDA runtime
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// Global constants corresponding to Python globals from config.py
// These are assumed to be defined and initialized elsewhere in your application.
namespace agx {
    extern const nc::NdArray<float> SPECTRAL_SHAPE;
    extern const nc::NdArray<float> STANDARD_OBSERVER_CMFS;
    extern const nc::NdArray<float> MALLETT2019_BASIS;
    nc::NdArray<float> HANATOS2025_SPECTRA_LUT; // Non-const, loaded at runtime
}

//================================================================================
// CUDA Kernels
//================================================================================

__global__ void tri2quad_kernel(float* out, const float* in, int n_coords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_coords) {
        const int i = idx * 2;
        float tx = in[i];
        float ty = in[i + 1];
        float y = ty / fmaxf(1.0f - tx, 1e-10f);
        float x = (1.0f - tx) * (1.0f - tx);
        out[i] = fminf(fmaxf(x, 0.0f), 1.0f);
        out[i + 1] = fminf(fmaxf(y, 0.0f), 1.0f);
    }
}

__global__ void quad2tri_kernel(float* out, const float* in, int n_coords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_coords) {
        const int i = idx * 2;
        float x = in[i];
        float y = in[i + 1];
        float sqrt_x = sqrtf(x);
        out[i] = 1.0f - sqrt_x;
        out[i + 1] = y * sqrt_x;
    }
}


namespace agx {
namespace utils {

/**
 * @brief A generic wrapper to simplify calling simple, element-wise CUDA kernels.
 * @tparam KernelFunc The type of the CUDA kernel function to launch.
 * @param input The input nc::NdArray.
 * @param kernel The CUDA kernel to execute.
 * @return A new nc::NdArray containing the results.
 */
template<typename KernelFunc>
nc::NdArray<float> cuda_elementwise_wrapper(const nc::NdArray<float>& input, KernelFunc kernel) {
    auto output = nc::NdArray<float>(input.shape());
    const auto num_elements = input.size();
    if (num_elements == 0) return output;
    const auto bytes = num_elements * sizeof(float);

    float *dev_in, *dev_out;
    cudaMalloc(&dev_in, bytes);
    cudaMalloc(&dev_out, bytes);

    cudaMemcpy(dev_in, input.data(), bytes, cudaMemcpyHostToDevice);

    const int threadsPerBlock = 256;
    const int n_coords = num_elements / input.shape().back(); // Number of (x,y) or (x,y,z) vectors
    const int blocksPerGrid = (n_coords + threadsPerBlock - 1) / threadsPerBlock;
    
    // Launch the provided kernel
    kernel<<<blocksPerGrid, threadsPerBlock>>>(dev_out, dev_in, n_coords);
    
    // Check for kernel launch errors
    cudaDeviceSynchronize();
    auto err = cudaGetLastError();
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string("CUDA Kernel Launch Error: ") + cudaGetErrorString(err));
    }

    cudaMemcpy(output.data(), dev_out, bytes, cudaMemcpyDeviceToHost);

    cudaFree(dev_in);
    cudaFree(dev_out);
    return output;
}


//================================================================================
// Function Implementations
//================================================================================

nc::NdArray<float> load_coeffs_lut(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }
    struct Header { int32_t magic, version, width, height; } h;
    file.read(reinterpret_cast<char*>(&h), sizeof(Header));
    
    auto lut_data = nc::NdArray<float>(h.height, h.width * 4);
    file.read(reinterpret_cast<char*>(lut_data.data()), lut_data.nbytes());
    
    return lut_data.reshape(h.height, h.width, 4);
}

nc::NdArray<float> tri2quad(const nc::NdArray<float>& tc) {
    return cuda_elementwise_wrapper(tc, tri2quad_kernel);
}

nc::NdArray<float> quad2tri(const nc::NdArray<float>& xy) {
    return cuda_elementwise_wrapper(xy, quad2tri_kernel);
}

nc::NdArray<float> fetch_coeffs(const nc::NdArray<float>& tc, const nc::NdArray<float>& lut_coeffs) {
    auto x_coords = nc::linspace<float>(0.0, 1.0, lut_coeffs.shape().rows);
    auto y_coords = nc::linspace<float>(0.0, 1.0, lut_coeffs.shape().cols);

    auto final_coeffs = nc::NdArray<float>(tc.shape().rows, tc.shape().cols, 4);

    for (nc::uint32 i = 0; i < 4; ++i) {
        auto lut_channel = lut_coeffs(nc::Slice(), nc::Slice(), i);
        // This assumes your scipy.hpp provides a C++ equivalent of RegularGridInterpolator
        auto channel_interpolator = scipy::interpolate::RegularGridInterpolator({x_coords, y_coords}, lut_channel, "cubic");
        final_coeffs(nc::Slice(), nc::Slice(), i) = channel_interpolator(tc);
    }
    return final_coeffs;
}

nc::NdArray<float> compute_spectra_from_coeffs(const nc::NdArray<float>& coeffs, int smooth_steps) {
    auto wl_up = nc::linspace<float>(360, 800, 441);
    
    // Extract coefficient channels
    auto c0 = coeffs(nc::Slice(), nc::Slice(), 0).reshape(coeffs.shape().rows, coeffs.shape().cols, 1);
    auto c1 = coeffs(nc::Slice(), nc::Slice(), 1).reshape(coeffs.shape().rows, coeffs.shape().cols, 1);
    auto c2 = coeffs(nc::Slice(), nc::Slice(), 2).reshape(coeffs.shape().rows, coeffs.shape().cols, 1);
    auto c3 = coeffs(nc::Slice(), nc::Slice(), 3).reshape(coeffs.shape().rows, coeffs.shape().cols, 1);

    // Perform calculations using broadcasting
    auto x = (c0 * wl_up + c1) * wl_up + c2;
    auto y = 1.0f / nc::sqrt(nc::square(x) + 1.0f);
    auto spectra = 0.5f * x * y + 0.5f;
    spectra /= c3;

    // Apply Gaussian filter
    auto step = nc::mean(nc::diff(agx::SPECTRAL_SHAPE.wavelengths)).item();
    spectra = scipy::ndimage::gaussian_filter(spectra, step * smooth_steps, -1);
    
    // Resample to final wavelength shape
    // This assumes a C++ equivalent of np.apply_along_axis with interp
    auto final_spectra = scipy::interpolate::interp(agx::SPECTRAL_SHAPE.wavelengths, wl_up, spectra, -1);
    return final_spectra;
}

nc::NdArray<float> compute_lut_spectra(int lut_size, int smooth_steps, const std::string& lut_coeffs_filename) {
    auto v = nc::linspace<float>(0.0, 1.0, lut_size);
    auto [tx, ty] = nc::meshgrid(v, v);
    auto tc = nc::stack({tx, ty}, nc::Axis::COL);
    
    auto lut_coeffs = load_coeffs_lut(lut_coeffs_filename);
    auto coeffs = fetch_coeffs(tc, lut_coeffs);
    auto lut_spectra = compute_spectra_from_coeffs(coeffs, smooth_steps);
    
    return lut_spectra.astype<nc::uint16>(); // Equivalent of np.half
}

nc::NdArray<float> load_spectra_lut(const std::string& filename) {
    return nc::load<double>(filename).astype<float>();
}

nc::NdArray<float> illuminant_to_xy(const std::string& illuminant_label) {
    auto illu = agx::model::standard_illuminant(illuminant_label);
    auto xyz = nc::dot(agx::STANDARD_OBSERVER_CMFS.transpose(), illu);
    return xyz(nc::Slice(0, 2)) / nc::sum(xyz).item();
}

std::pair<nc::NdArray<float>, nc::NdArray<float>> rgb_to_tc_b(
        const nc::NdArray<float>& rgb, const std::string& color_space,
        bool apply_cctf_decoding, const std::string& reference_illuminant) {

    auto illu_xy = illuminant_to_xy(reference_illuminant);
    auto xyz = colour::RGB_to_XYZ(rgb, color_space, apply_cctf_decoding, illu_xy, "CAT02");

    auto b = nc::sum<float>(xyz, nc::Axis::COL);
    auto xy = xyz(nc::Slice(), nc::Slice(0, 2)) / nc::maximum(b.reshape(-1, 1), 1e-10f);
    xy = nc::clip(xy, 0.0f, 1.0f);
    
    auto tc = quad2tri(xy); // Use the CUDA-accelerated version
    auto b_final = nc::nan_to_num(b);

    return {tc, b_final};
}

nc::NdArray<float> compute_band_pass_filter(const nc::NdArray<float>& filter_uv_params, const nc::NdArray<float>& filter_ir_params) {
    float amp_uv = nc::clip(filter_uv_params[0], 0.f, 1.f);
    float wl_uv = filter_uv_params[1];
    float width_uv = filter_uv_params[2];

    float amp_ir = nc::clip(filter_ir_params[0], 0.f, 1.f);
    float wl_ir = filter_ir_params[1];
    float width_ir = filter_ir_params[2];

    const auto& wl = agx::SPECTRAL_SHAPE.wavelengths;
    auto sigmoid = [](const auto& x, float center, float width) {
        return scipy::special::erf((x - center) / width) * 0.5f + 0.5f;
    };
    
    auto filter_uv  = 1.0f - amp_uv + amp_uv * sigmoid(wl, wl_uv, width_uv);
    auto filter_ir  = 1.0f - amp_ir + amp_ir * sigmoid(wl, wl_ir, -width_ir);

    return filter_uv * filter_ir;
}

nc::NdArray<float> rgb_to_raw_mallett2019(
        const nc::NdArray<float>& RGB, const nc::NdArray<float>& sensitivity,
        const std::string& color_space, bool apply_cctf_decoding, const std::string& reference_illuminant) {

    auto illuminant = agx::model::standard_illuminant(reference_illuminant);
    auto basis_set_with_illuminant = agx::MALLETT2019_BASIS * illuminant.reshape(illuminant.size(), 1);

    auto lrgb = colour::RGB_to_RGB(RGB, color_space, "sRGB", apply_cctf_decoding, false);
    lrgb = nc::clip(lrgb, 0.0f, nc::constants::inf);

    // contract('ijk,lk,lm->ijm', lrgb, basis, sens) is complex.
    // This requires a batched matrix multiplication.
    // lrgb [h, w, 3], basis [n_wl, 3], sens [n_wl, 3] -> raw [h, w, 3]
    // Simplified, for each pixel: reconstructed_spectrum = basis @ lrgb_pixel
    // raw_pixel = sens.T @ reconstructed_spectrum
    // This is a placeholder for that complex operation.
    auto raw = colour::contract_mallett2019(lrgb, basis_set_with_illuminant, sensitivity);
    raw = nc::nan_to_num(raw);
    
    auto midgray_spectrum = illuminant * 0.184f;
    auto raw_midgray  = nc::dot(midgray_spectrum, sensitivity); // einsum('k,km->m')
    return raw / raw_midgray[1];
}

nc::NdArray<float> rgb_to_raw_hanatos2025(
        const nc::NdArray<float>& rgb, const nc::NdArray<float>& sensitivity,
        const std::string& color_space, bool apply_cctf_decoding, const std::string& reference_illuminant) {

    nc::NdArray<float> raw;
    if (rgb.shape().cols == 1 && rgb.shape().rows == 1) { // Single pixel case
        auto spectrum = rgb_to_spectrum(rgb, color_space, apply_cctf_decoding, reference_illuminant);
        raw = nc::dot(spectrum, sensitivity); // einsum('l,lm->m')
        raw.reshape(1, 1, raw.size());
    } else { // Image case
        auto [tc_raw, b] = rgb_to_tc_b(rgb, color_space, apply_cctf_decoding, reference_illuminant);
        
        // einsum('ijl,lm->ijm') -> dot(lut.reshape(-1, n_wl), sens).reshape(lut.shape)
        auto reshaped_lut = agx::HANATOS2025_SPECTRA_LUT.reshape(-1, agx::HANATOS2025_SPECTRA_LUT.shape().back());
        auto tc_lut = nc::dot(reshaped_lut, sensitivity).reshape(
            agx::HANATOS2025_SPECTRA_LUT.shape().rows,
            agx::HANATOS2025_SPECTRA_LUT.shape().cols,
            sensitivity.shape().cols);
        
        raw = apply_lut_cubic_2d(tc_lut, tc_raw); // Your custom interpolation
        raw *= b.reshape(b.size(), 1); // Scale back with brightness
    }
    
    auto midgray_rgb = nc::full<float>({1, 1, 3}, 0.184f);
    auto illuminant_midgray = rgb_to_spectrum(midgray_rgb, color_space, false, reference_illuminant);
    auto raw_midgray = nc::dot(illuminant_midgray, sensitivity); // einsum('k,km->m')

    return raw / raw_midgray[1];
}

nc::NdArray<float> rgb_to_spectrum(
        const nc::NdArray<float>& rgb, const std::string& color_space,
        bool apply_cctf_decoding, const std::string& reference_illuminant) {

    auto [tc_w, b_w] = rgb_to_tc_b(rgb, color_space, apply_cctf_decoding, reference_illuminant);

    auto v = nc::linspace<float>(0.0, 1.0, agx::HANATOS2025_SPECTRA_LUT.shape().rows);
    auto interp = scipy::interpolate::RegularGridInterpolator({v, v}, agx::HANATOS2025_SPECTRA_LUT, "cubic");

    auto spectrum_w = interp(tc_w);
    spectrum_w *= b_w.reshape(b_w.size(), 1); // Ensure b_w is broadcastable

    return spectrum_w.flatten();
}

void run_spectral_upsampling_example() {
    std::cout << "\n--- Running Spectral Upsampling Example ---" << std::endl;
    try {
        agx::HANATOS2025_SPECTRA_LUT = compute_lut_spectra(128);
        std::cout << "Successfully computed main spectral LUT." << std::endl;

        auto tc_test = nc::full<float>({1, 1, 2}, 0.5f); // Test with a middle gray
        auto lut_coeffs = load_coeffs_lut();
        auto coeffs = fetch_coeffs(tc_test, lut_coeffs);
        auto spectra = compute_spectra_from_coeffs(coeffs);
        std::cout << "Successfully computed a test spectrum from coefficients. Shape: " << spectra.shape() << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "An error occurred during the example run: " << e.what() << std::endl;
    }
}

} // namespace utils
} // namespace agx