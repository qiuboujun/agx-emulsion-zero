// couplers.hpp
//
// This header defines a small utility class used to model the DIR couplers
// used in analogue film simulation.  The implementation lives in
// ``couplers.cpp`` for host (CPU) execution and ``couplers.cu`` for GPU
// accelerated paths.  The API mirrors the behaviour of the Python module
// ``agx_emulsion/model/couplers.py`` found in the AGX Emulsion Zero project.
// Each function attempts to faithfully reproduce the original algorithms
// implemented in that file, including the same normalisation schemes and
// interpolation routines.

#pragma once

#include <array>
#include <vector>

namespace agx_emulsion {

// The ``Couplers`` class contains only static methods.  It is not
// intended to be instantiated.  The functions here operate on simple
// containers (std::array and std::vector) rather than heavy weight
// matrices to reduce dependencies on third‑party libraries.  Should you
// wish to integrate with Eigen or another linear algebra library you can
// easily convert the inputs and outputs accordingly.
class Couplers {
public:
    /// Compute the DIR couplers matrix given per‑channel amount and a
    /// diffusion sigma.  The returned matrix M has its first index
    /// corresponding to the input (source) layer and the second index
    /// corresponding to the output (target) layer.  This is equivalent
    /// to the result returned by the Python function
    /// ``compute_dir_couplers_matrix``.
    ///
    /// \param amount_rgb  Amount of inhibitors per colour channel in the
    ///                    [cyan, magenta, yellow] ordering.  Values are
    ///                    typically in the range [0,1].
    /// \param layer_diffusion  Standard deviation (sigma) of the
    ///                         Gaussian diffusion along the layer axis.
    /// \return A 3×3 matrix representing the diffused inhibition
    ///         coefficients.
    static std::array<std::array<double, 3>, 3>
    compute_dir_couplers_matrix(const std::array<double, 3> &amount_rgb,
                                double layer_diffusion);

    /// Compute density curves before the effect of DIR couplers.  This
    /// function inverts the effect of the inhibitors by performing a
    /// per‑channel linear interpolation.  The API is deliberately
    /// conservative: all inputs are passed by constant reference and a
    /// freshly allocated output is returned.
    ///
    /// \param density_curves  Characteristic density curves after
    ///                        couplers have been applied.  The first
    ///                        dimension indexes exposure samples, the second
    ///                        indexes channels (0=cyan, 1=magenta, 2=yellow).
    /// \param log_exposure    Logarithmic exposure values corresponding to
    ///                        each sample in the first dimension of
    ///                        ``density_curves``.
    /// \param dir_couplers_matrix  The matrix returned by
    ///                             ``compute_dir_couplers_matrix``.
    /// \param high_exposure_couplers_shift  Optional coefficient that
    ///                                      increases inhibitors at
    ///                                      high exposures.  Defaults to 0.
    /// \return A new 2D array of shape (N×3) containing the corrected
    ///         density curves.
    static std::vector<std::vector<double>>
    compute_density_curves_before_dir_couplers(
        const std::vector<std::vector<double>> &density_curves,
        const std::vector<double> &log_exposure,
        const std::array<std::array<double, 3>, 3> &dir_couplers_matrix,
        double high_exposure_couplers_shift = 0.0);

    /// Apply coupler inhibitors to a raw log exposure volume.  The
    /// inhibitors act on the density and spatially diffuse according to
    /// ``diffusion_size_pixel``.  The result is subtracted from the input
    /// to produce a corrected log exposure volume.  This mirrors the
    /// behaviour of ``compute_exposure_correction_dir_couplers`` in the
    /// original Python module.
    ///
    /// \param log_raw    A two dimensional grid of raw log exposure values
    ///                    where the innermost array holds three channel
    ///                    components (C,M,Y).  Dimensions: [height][width][3].
    /// \param density_cmy  The per‑pixel density values in the same shape
    ///                     as ``log_raw``.
    /// \param density_max  The maximum achievable density per channel.
    ///                     Typically a three component array.  If all
    ///                     channels share the same maximum a uniform array
    ///                     may be supplied.
    /// \param dir_couplers_matrix  Matrix as returned by
    ///                             ``compute_dir_couplers_matrix``.
    /// \param diffusion_size_pixel  Spatial diffusion sigma in pixels for
    ///                              the Gaussian blur.  When zero or
    ///                              negative no blur is applied.
    /// \param high_exposure_couplers_shift  Optional coefficient as above.
    /// \return A corrected log exposure volume of the same dimensions as
    ///         ``log_raw``.
    static std::vector<std::vector<std::array<double, 3>>>
    compute_exposure_correction_dir_couplers(
        const std::vector<std::vector<std::array<double, 3>>> &log_raw,
        const std::vector<std::vector<std::array<double, 3>>> &density_cmy,
        const std::array<double, 3> &density_max,
        const std::array<std::array<double, 3>, 3> &dir_couplers_matrix,
        int diffusion_size_pixel,
        double high_exposure_couplers_shift = 0.0);
};

} // namespace agx_emulsion