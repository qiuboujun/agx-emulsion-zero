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