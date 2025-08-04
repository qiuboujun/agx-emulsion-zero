"""
Unit tests for the compiled C++ implementation of the film emulsion coupler algorithms.

This script tests the actual compiled C++ code by importing the pybind11 module
and comparing it against the Python reference implementation.
"""

import unittest
import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from opt_einsum import contract  # pragma: no cover
except ImportError:
    # Fallback to numpy.einsum if opt_einsum is not available.  The
    # signatures used in these tests are simple and can be handled by
    # numpy directly.
    def contract(subscripts, *operands):
        return np.einsum(subscripts, *operands)

# Try to import the compiled C++ module
try:
    import couplers_cpp_tests as cpp_couplers
    CPP_AVAILABLE = True
except ImportError:
    print("Warning: C++ couplers module not available. Skipping C++ tests.")
    CPP_AVAILABLE = False


def compute_dir_couplers_matrix_py(amount_rgb=[0.7, 0.7, 0.5], layer_diffusion=1.0):
    """Reference implementation lifted verbatim from couplers.py."""
    M = np.eye(3)
    M_diffused = gaussian_filter(M, layer_diffusion, mode="constant", cval=0, axes=1)
    M_diffused /= np.sum(M_diffused, axis=1)[:, None]
    M = M_diffused * np.array(amount_rgb)[:, None]
    return M


def compute_density_curves_before_dir_couplers_py(
    density_curves, log_exposure, dir_couplers_matrix, high_exposure_couplers_shift=0.0
):
    d_max = np.nanmax(density_curves, axis=0)
    dc_norm = density_curves / d_max
    dc_norm_shift = dc_norm + high_exposure_couplers_shift * dc_norm**2
    couplers_amount_curves = contract("jk, km->jm", dc_norm_shift, dir_couplers_matrix)
    x0 = log_exposure[:, None] - couplers_amount_curves
    density_curves_corrected = np.zeros_like(density_curves)
    for i in np.arange(3):
        # np.interp expects 1D arrays; np.interp performs linear interpolation
        density_curves_corrected[:, i] = np.interp(log_exposure, x0[:, i], density_curves[:, i])
    return density_curves_corrected


def compute_exposure_correction_dir_couplers_py(
    log_raw,
    density_cmy,
    density_max,
    dir_couplers_matrix,
    diffusion_size_pixel,
    high_exposure_couplers_shift=0.0,
):
    norm_density = density_cmy / density_max
    norm_density = norm_density + high_exposure_couplers_shift * norm_density**2
    log_raw_correction = contract("ijk, km->ijm", norm_density, dir_couplers_matrix)
    if diffusion_size_pixel > 0:
        sigma_tuple = (diffusion_size_pixel, diffusion_size_pixel, 0)
        log_raw_correction = gaussian_filter(log_raw_correction, sigma_tuple)
    log_raw_corrected = log_raw - log_raw_correction
    return log_raw_corrected


class TestCouplersCppImplementation(unittest.TestCase):
    @unittest.skipUnless(CPP_AVAILABLE, "C++ module not available")
    def test_compute_dir_couplers_matrix_cpp(self):
        """Test the C++ implementation of compute_dir_couplers_matrix."""
        amount_rgb = np.array([0.7, 0.7, 0.5])
        for sigma in [0.5, 1.0, 2.0]:
            m_py = compute_dir_couplers_matrix_py(amount_rgb, sigma)
            m_cpp = cpp_couplers.compute_dir_couplers_matrix(amount_rgb, sigma)
            self.assertTrue(np.allclose(m_py, m_cpp, atol=1e-6),
                           f"Mismatch with sigma={sigma}")

    @unittest.skipUnless(CPP_AVAILABLE, "C++ module not available")
    def test_compute_density_curves_before_cpp(self):
        """Test the C++ implementation of compute_density_curves_before_dir_couplers."""
        # Generate monotonic exposure and strictly increasing density curves.
        N = 50
        log_exposure = np.linspace(-2.0, 1.0, N)
        density_curves = np.zeros((N, 3))
        # Define unique slopes and intercepts per channel
        slopes = [0.6, 0.8, 0.7]
        intercepts = [0.1, 0.2, 0.3]
        for c in range(3):
            density_curves[:, c] = intercepts[c] + slopes[c] * np.linspace(0, 1, N)
        
        dir_matrix = compute_dir_couplers_matrix_py([0.7, 0.7, 0.5], 1.0)
        
        corrected_py = compute_density_curves_before_dir_couplers_py(
            density_curves, log_exposure, dir_matrix
        )
        corrected_cpp = cpp_couplers.compute_density_curves_before_dir_couplers(
            density_curves, log_exposure, dir_matrix
        )
        self.assertTrue(np.allclose(corrected_py, corrected_cpp, atol=1e-6))

    @unittest.skipUnless(CPP_AVAILABLE, "C++ module not available")
    def test_compute_exposure_correction_cpp(self):
        """Test the C++ implementation of compute_exposure_correction_dir_couplers."""
        np.random.seed(1)
        H, W = 6, 5
        log_raw = np.random.rand(H, W, 3).astype(float)
        density_cmy = np.random.rand(H, W, 3).astype(float)
        density_max = np.array([2.0, 2.2, 2.4])
        dir_matrix = compute_dir_couplers_matrix_py([0.9, 0.7, 0.5], 1.5)
        
        for diffusion in [0, 1, 2]:
            corrected_py = compute_exposure_correction_dir_couplers_py(
                log_raw, density_cmy, density_max, dir_matrix, diffusion, 0.1
            )
            corrected_cpp = cpp_couplers.compute_exposure_correction_dir_couplers(
                log_raw, density_cmy, density_max, dir_matrix, diffusion, 0.1
            )
            self.assertTrue(
                np.allclose(corrected_py, corrected_cpp, atol=1e-5),
                msg=f"Mismatch with diffusion={diffusion}",
            )

    @unittest.skipUnless(CPP_AVAILABLE, "C++ module not available")
    def test_edge_cases_cpp(self):
        """Test edge cases for the C++ implementation."""
        # Test with zero diffusion
        amount_rgb = np.array([0.0, 0.0, 0.0])
        m_cpp = cpp_couplers.compute_dir_couplers_matrix(amount_rgb, 0.0)
        self.assertTrue(np.allclose(m_cpp, np.zeros((3, 3)), atol=1e-10))
        
        # Test with very small diffusion
        m_cpp = cpp_couplers.compute_dir_couplers_matrix(amount_rgb, 1e-10)
        self.assertTrue(np.allclose(m_cpp, np.zeros((3, 3)), atol=1e-10))
        
        # Test with very large diffusion
        amount_rgb = np.array([1.0, 1.0, 1.0])
        m_cpp = cpp_couplers.compute_dir_couplers_matrix(amount_rgb, 100.0)
        # Should be approximately uniform
        expected = np.ones((3, 3)) / 3
        self.assertTrue(np.allclose(m_cpp, expected, atol=1e-2))


if __name__ == "__main__":
    unittest.main() 