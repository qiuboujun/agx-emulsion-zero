// Copyright (c) 2025
//
// This file provides the CPU implementation of the SpectralUpsampling
// routines declared in `spectral_upsampling.hpp`.  These functions
// operate on scalar inputs and simple STL containers.  They are
// intended as a teaching example for how one might translate Python
// code into C++ while maintaining readability and without relying on
// external dependencies.  Should more performance or functionality be
// required, consider replacing the hand‑rolled loops with calls into
// libraries such as Eigen, xtensor or a custom CUDA implementation.

#include "spectral_upsampling.hpp"
#include "colour.hpp"
#include "fast_interp_lut.hpp"
#include "illuminants.hpp"
#include <fstream>
#include <stdexcept>

#include <algorithm> // for std::clamp
#include <cmath>
#include <cstddef>

std::pair<float, float>
SpectralUpsampling::tri2quad(float tx, float ty)
{
    // Guard against division by zero by ensuring the denominator is
    // never smaller than a small epsilon.  This mirrors the
    // Python implementation which uses `np.fmax(1.0 - tx, 1e-10)`.
    const float denom = std::max(1.0f - tx, 1e-10f);
    float y = ty / denom;
    // The x coordinate is simply the squared complement of tx.
    float x = (1.0f - tx) * (1.0f - tx);
    // Clamp both coordinates into the [0,1] range just as the
    // reference implementation does via np.clip.
    x = std::clamp(x, 0.0f, 1.0f);
    y = std::clamp(y, 0.0f, 1.0f);
    return {x, y};
}

std::pair<float, float>
SpectralUpsampling::quad2tri(float x, float y)
{
    // Convert from square back into triangular coordinates.  The
    // inverse transform uses the square root of the x coordinate.
    float sqrt_x = std::sqrt(x);
    float tx = 1.0f - sqrt_x;
    float ty = y * sqrt_x;
    return {tx, ty};
}

std::vector<float>
SpectralUpsampling::computeSpectraFromCoeffs(const std::array<float, 4> &coeffs)
{
    // We generate a wavelength axis from 360 to 800 inclusive.  The
    // Python implementation uses 441 points to achieve half‑nm
    // resolution; here we match that granularity for consistency.  The
    // step size will be (800 - 360) / (441 - 1) ≈ 1.0.
    constexpr std::size_t n_wavelengths = 441;
    const float wl_start = 360.0f;
    const float wl_end   = 800.0f;
    const float step = (wl_end - wl_start) / static_cast<float>(n_wavelengths - 1);

    std::vector<float> spectra;
    spectra.reserve(n_wavelengths);
    for (std::size_t i = 0; i < n_wavelengths; ++i) {
        const float wl = wl_start + static_cast<float>(i) * step;
        // Evaluate the polynomial x = (c0 * wl + c1) * wl + c2
        const float x = (coeffs[0] * wl + coeffs[1]) * wl + coeffs[2];
        const float y = 1.0f / std::sqrt(x * x + 1.0f);
        float value = 0.5f * x * y + 0.5f;
        // Divide by c3, guarding against zero to avoid division by
        // zero; if c3 is zero we simply leave the value unchanged.
        if (coeffs[3] != 0.0f) {
            value /= coeffs[3];
        }
        spectra.push_back(value);
    }
    return spectra;
}

// ================= Consolidated extras: npy loader and rgb_to_raw =================

namespace agx { namespace utils {

// Minimal NPY (v1.0/2.0) loader for float32 arrays; returns row-major NdArray
static void parse_npy_header(std::ifstream& f, size_t& rows, size_t& cols, bool& fortran_order, int& word_size, char& type_code) {
    char magic[6]; f.read(magic, 6);
    if (std::string(magic,6) != "\x93NUMPY") throw std::runtime_error("Invalid NPY magic");
    unsigned char v_major = 0, v_minor = 0; f.read((char*)&v_major,1); f.read((char*)&v_minor,1);
    uint16_t header_len_u16 = 0; uint32_t header_len_u32 = 0; size_t header_len = 0;
    if (v_major == 1) { f.read((char*)&header_len_u16,2); header_len = header_len_u16; }
    else { f.read((char*)&header_len_u32,4); header_len = header_len_u32; }
    std::string header(header_len, '\0'); f.read(header.data(), header_len);
    // Parse dtype: support f2 (float16), f4 (float32), f8 (float64)
    if      (header.find("'descr': '<f2'") != std::string::npos || header.find("'descr': '|f2'") != std::string::npos) { word_size = 2; type_code='f'; }
    else if (header.find("'descr': '<f4'") != std::string::npos || header.find("'descr': '|f4'") != std::string::npos) { word_size = 4; type_code='f'; }
    else if (header.find("'descr': '<f8'") != std::string::npos || header.find("'descr': '|f8'") != std::string::npos) { word_size = 8; type_code='f'; }
    else {
        throw std::runtime_error("NPY dtype not supported (expect f2/f4/f8)");
    }
    fortran_order = header.find("'fortran_order': True") != std::string::npos;
    auto p0 = header.find("shape");
    if (p0 == std::string::npos) throw std::runtime_error("NPY missing shape");
    auto p1 = header.find('(', p0);
    auto p2 = header.find(')', p1);
    std::string shape_str = header.substr(p1+1, p2-p1-1);
    // Expect 3D (L, L, K)
    int L1=0,L2=0,K=0;
    if (sscanf(shape_str.c_str(), "%d, %d, %d", &L1, &L2, &K) != 3)
        throw std::runtime_error("NPY shape parse failed");
    if (L1 != L2) throw std::runtime_error("NPY expected square LUT (L,L,K)");
    rows = static_cast<size_t>(L1*L2);
    cols = static_cast<size_t>(K);
}

// float16 -> float converter
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (h & 0x8000u) << 16;
    uint32_t exp  = (h & 0x7C00u) >> 10;
    uint32_t frac = (h & 0x03FFu);
    uint32_t f;
    if (exp == 0) {
        if (frac == 0) { f = sign; }
        else {
            // subnormal
            while ((frac & 0x0400u) == 0) { frac <<= 1; exp--; }
            exp++;
            frac &= ~0x0400u;
            f = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
        }
    } else if (exp == 0x1Fu) {
        // Inf/NaN
        f = sign | 0x7F800000u | (frac << 13);
    } else {
        // normal
        f = sign | ((exp + (127 - 15)) << 23) | (frac << 13);
    }
    float out;
    std::memcpy(&out, &f, sizeof(out));
    return out;
}

nc::NdArray<float> load_hanatos_spectra_lut_npy(const std::string& npy_path) {
    std::ifstream f(npy_path, std::ios::binary);
    if (!f.is_open()) throw std::runtime_error("Cannot open NPY: " + npy_path);
    size_t rows=0, cols=0; bool fortran=false; int word_size=0; char type_code='\0';
    parse_npy_header(f, rows, cols, fortran, word_size, type_code);
    if (fortran) throw std::runtime_error("NPY Fortran order not supported");
    // Read raw data (L,L,K)
    nc::NdArray<float> lut(rows, cols);
    const size_t count = rows*cols;

    if (word_size == 4) {
        std::vector<float> buf(count);
        f.read(reinterpret_cast<char*>(buf.data()), count*sizeof(float));
        for (size_t i=0;i<count;++i) lut[i] = buf[i];
    } else if (word_size == 8) {
        std::vector<double> buf(count);
        f.read(reinterpret_cast<char*>(buf.data()), count*sizeof(double));
        for (size_t i=0;i<count;++i) lut[i] = static_cast<float>(buf[i]);
    } else if (word_size == 2) {
        std::vector<uint16_t> buf(count);
        f.read(reinterpret_cast<char*>(buf.data()), count*sizeof(uint16_t));
        for (size_t i=0;i<count;++i) lut[i] = f16_to_f32(buf[i]);
    } else {
        throw std::runtime_error("Unsupported word size in NPY");
    }
    // Map to (rows=L*L, cols=K) NumCpp
    return lut;
}

std::pair<nc::NdArray<float>, nc::NdArray<float>> rgb_to_tc_b_cpp(
    const nc::NdArray<float>& rgb,
    const std::string& color_space,
    bool apply_cctf_decoding,
    const std::string& reference_illuminant) {
    // Compute illuminant xy matching Python path via SPD and CMFs
    auto illuminant_spd = agx::model::standard_illuminant(reference_illuminant).flatten();
    auto cmfs = colour::get_cie_1931_2_degree_cmfs(); // columns: wl,x,y,z
    // Ensure compatible lengths
    const size_t n = std::min(static_cast<size_t>(illuminant_spd.size()), static_cast<size_t>(cmfs.shape().rows));
    double X = 0.0, Y = 0.0, Z = 0.0;
    for (size_t i = 0; i < n; ++i) {
        double e = static_cast<double>(illuminant_spd[i]);
        X += e * static_cast<double>(cmfs(i, 1 - 1 + 0)); // cmfs(:,0) is x
        Y += e * static_cast<double>(cmfs(i, 1));         // cmfs(:,1) is y
        Z += e * static_cast<double>(cmfs(i, 2));         // cmfs(:,2) is z
    }
    double S = std::max(1e-20, X + Y + Z);
    nc::NdArray<float> illum_xy(1,2);
    illum_xy(0,0) = static_cast<float>(X / S);
    illum_xy(0,1) = static_cast<float>(Y / S);

    auto xyz = colour::RGB_to_XYZ(rgb, color_space, apply_cctf_decoding, illum_xy, "CAT02");
    nc::NdArray<float> b(xyz.shape().rows,1);
    nc::NdArray<float> xy(xyz.shape().rows,2);
    for (uint32_t i=0;i<xyz.shape().rows;++i){
        b(i,0)=xyz(i,0)+xyz(i,1)+xyz(i,2);
        float denom = std::max(b(i,0), 1e-10f);
        xy(i,0)=xyz(i,0)/denom; xy(i,1)=xyz(i,1)/denom;
        if (xy(i,0)<0) xy(i,0)=0; if (xy(i,0)>1) xy(i,0)=1;
        if (xy(i,1)<0) xy(i,1)=0; if (xy(i,1)>1) xy(i,1)=1;
    }
    // tri2quad
    nc::NdArray<float> tc(xy.shape().rows,2);
    for (uint32_t i=0;i<xy.shape().rows;++i){
        float sqrt_x = std::sqrt(xy(i,0));
        float tx = 1.0f - sqrt_x;
        float ty = xy(i,1) * sqrt_x;
        float y = ty / std::max(1.0f - tx, 1e-10f);
        float x = (1.0f - tx) * (1.0f - tx);
        x = std::min(1.0f, std::max(0.0f, x));
        y = std::min(1.0f, std::max(0.0f, y));
        tc(i,0)=x; tc(i,1)=y;
    }
    return {tc, b};
}

// Bilinear interpolation for a single (x,y) over a LUT of shape (L*L, K)
static std::vector<float> bilinear_interp_lut_at_2d_channels(const nc::NdArray<float>& lut, float x, float y) {
    const int L = static_cast<int>(std::round(std::sqrt(lut.shape().rows)));
    const int K = lut.shape().cols;
    // Clamp coordinates to [0, L-1]
    x = std::max(0.0f, std::min(x, static_cast<float>(L-1)));
    y = std::max(0.0f, std::min(y, static_cast<float>(L-1)));
    const int x0 = static_cast<int>(std::floor(x));
    const int y0 = static_cast<int>(std::floor(y));
    const int x1 = std::min(x0 + 1, L - 1);
    const int y1 = std::min(y0 + 1, L - 1);
    const float fx = x - static_cast<float>(x0);
    const float fy = y - static_cast<float>(y0);
    const float w00 = (1.0f - fx) * (1.0f - fy);
    const float w10 = fx * (1.0f - fy);
    const float w01 = (1.0f - fx) * fy;
    const float w11 = fx * fy;

    std::vector<float> out(K, 0.0f);
    const int idx00 = x0 * L + y0;
    const int idx10 = x1 * L + y0;
    const int idx01 = x0 * L + y1;
    const int idx11 = x1 * L + y1;
    for (int k = 0; k < K; ++k) {
        float v00 = lut(idx00, k);
        float v10 = lut(idx10, k);
        float v01 = lut(idx01, k);
        float v11 = lut(idx11, k);
        out[k] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
    }
    return out;
}

nc::NdArray<float> rgb_to_raw_hanatos2025(
    const nc::NdArray<float>& rgb,
    const nc::NdArray<float>& sensitivity,
    const std::string& color_space,
    bool apply_cctf_decoding,
    const std::string& reference_illuminant,
    const nc::NdArray<float>& spectra_lut) {
    // Preproject spectra LUT by sensitivity: (L*L,K) dot (K,3) -> (L*L,3)
    const int rows = spectra_lut.shape().rows;
    const int K = spectra_lut.shape().cols;
    if ((int)sensitivity.shape().rows != K || sensitivity.shape().cols != 3) {
        throw std::runtime_error("rgb_to_raw_hanatos2025: sensitivity shape must be (K,3)");
    }
    nc::NdArray<float> lut_proj(rows, 3);
    for (int r=0;r<rows;++r){
        for (int c=0;c<3;++c){
            float acc = 0.0f;
            for (int k=0;k<K;++k) acc += spectra_lut(r,k) * sensitivity(k,c);
            lut_proj(r,c) = acc;
        }
    }
    const int L = static_cast<int>(std::round(std::sqrt(rows)));

    auto [tc, b] = rgb_to_tc_b_cpp(rgb, color_space, apply_cctf_decoding, reference_illuminant);
    const int H = rgb.shape().rows; const int W = 1; // treat as Hx1 grid
    auto raw = agx::apply_lut_cubic_2d(lut_proj, tc, H, W); // (H, 3)
    // Scale by b
    for (int i=0;i<H;++i){ for (int c=0;c<3;++c) raw(i,c) *= b(i,0); }

    // Midgray normalization using spectra bilinear interpolation (match Python RegularGridInterpolator default linear)
    nc::NdArray<float> mid(1,3); mid(0,0)=0.184f; mid(0,1)=0.184f; mid(0,2)=0.184f;
    auto [tc_m, b_m] = rgb_to_tc_b_cpp(mid, color_space, false, reference_illuminant);
    const float x_m = tc_m(0,0) * (L - 1);
    const float y_m = tc_m(0,1) * (L - 1);
    std::vector<float> spectrum_mid = bilinear_interp_lut_at_2d_channels(spectra_lut, x_m, y_m);
    // Scale spectrum by b_m
    for (auto& v : spectrum_mid) v *= b_m(0,0);
    // Project with sensitivity
    float raw_mid[3] = {0.0f, 0.0f, 0.0f};
    for (int c=0;c<3;++c){
        float acc = 0.0f;
        for (int k=0;k<K;++k) acc += spectrum_mid[k] * sensitivity(k,c);
        raw_mid[c] = acc;
    }
    float scale = 1.0f / std::max(1e-10f, raw_mid[1]);
    for (int i=0;i<H;++i){ for(int c=0;c<3;++c) raw(i,c)*=scale; }
    return raw;
}

}} // namespace agx::utils
