// Copyright (c) 2025
//
// This header declares a simple C++ interface for a spectral upsampling
// utility.  The goal of this exercise is not to perfectly reproduce the
// entire Python implementation found in `agx_emulsion/utils/spectral_upsampling.py`,
// but rather to demonstrate how one might begin porting selected
// functionality into a modern C++ codebase with clear separation of
// interface (this header), CPU implementation (.cpp) and GPU
// implementation (.cu).
//
// The functions provided here focus on coordinate transforms used by
// the spectral upsampling code.  Additional routines such as the
// spectrum synthesis from a set of coefficients are provided in a
// simplified form in the CPU implementation.  These can serve as a
// starting point for a fuller port as more of the Python functionality
// is needed.

#ifndef SPECTRAL_UPSAMPLING_HPP
#define SPECTRAL_UPSAMPLING_HPP

#include <array>
#include <utility>
#include <vector>

// The SpectralUpsampling class exposes a handful of static helper
// functions.  In this basic port we avoid any dependencies on third
// party linear algebra or interpolation libraries; all operations are
// implemented using standard C++ containers and simple loops.  For
// production use one would likely prefer to integrate a library such
// as Eigen or xtensor for array handling and interpolation.

class SpectralUpsampling {
public:
    /// Convert a point expressed in triangular barycentric coordinates
    /// `(tx, ty)` into square coordinates.  In the Python reference
    /// implementation this function operates on array inputs.  Here we
    /// provide a scalar version returning a pair of floats.  See
    /// `tri2quad` in `spectral_upsampling.py` for the original algorithm.
    static std::pair<float, float> tri2quad(float tx, float ty);

    /// Convert a point expressed in square coordinates `(x, y)` back into
    /// triangular barycentric coordinates.  This is the inverse of
    /// `tri2quad`.  See `quad2tri` in the Python reference for details.
    static std::pair<float, float> quad2tri(float x, float y);

    /// Given four spectral synthesis coefficients `coeffs` this routine
    /// computes a coarse spectral distribution.  The implementation
    /// directly follows the core of `compute_spectra_from_coeffs` from
    /// the Python version, but omits Gaussian smoothing and down
    /// sampling.  The returned vector will contain 441 samples over
    /// wavelengths 360–800 nm at 1 nm intervals.  The order of the
    /// coefficients is assumed to be `(c0, c1, c2, c3)`.
    static std::vector<float> computeSpectraFromCoeffs(const std::array<float, 4> &coeffs);
};

#endif // SPECTRAL_UPSAMPLING_HPP