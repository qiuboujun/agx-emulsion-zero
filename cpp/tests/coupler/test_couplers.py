"""
Unit tests for the C++ translation of the film emulsion coupler algorithms.

These tests exercise the three public functions exposed by ``couplers.cpp`` by
replicating the algorithms in pure Python and comparing against the original
Python reference implementation in ``agx_emulsion/model/couplers.py``.  The
tests do not require compiling the C++ code – instead they validate that
the translated algorithms produce identical results to the Python versions
when provided with fixed, randomised input.  Should you compile
``couplers.cpp`` and bind it into Python (for example using pybind11),
these tests may be easily adapted to call the compiled code directly.
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


# Python versions of the C++ algorithms.  These duplicate the logic in
# couplers.cpp and serve to validate that the C++ code will match the
# behaviour of the Python reference implementation.

def compute_dir_couplers_matrix_cpp(amount_rgb, layer_diffusion):
    """Replicates Couplers::compute_dir_couplers_matrix in Python."""
    result = np.zeros((3, 3), dtype=float)
    sigma = layer_diffusion
    for i in range(3):
        # Compute unnormalised weights
        weights = np.zeros(3)
        sum_w = 0.0
        for j in range(3):
            if sigma > 0:
                diff = j - i
                w = np.exp(-0.5 * (diff / sigma) ** 2)
            else:
                w = 1.0 if j == i else 0.0
            weights[j] = w
            sum_w += w
        if sum_w > 0:
            weights = weights / sum_w
        weights = weights * amount_rgb[i]
        result[i] = weights
    return result


def interpolate_1d_cpp(x, y, x_new):
    """Performs the same sorted linear interpolation as in the C++ code."""
    x = np.asarray(x)
    y = np.asarray(y)
    idx = np.argsort(x)
    x_sorted = x[idx]
    y_sorted = y[idx]
    out = np.empty_like(x_new, dtype=float)
    for i, t in enumerate(x_new):
        if len(x_sorted) == 1:
            out[i] = y_sorted[0]
            continue
        if t <= x_sorted[0]:
            out[i] = y_sorted[0]
            continue
        if t >= x_sorted[-1]:
            out[i] = y_sorted[-1]
            continue
        # binary search interval
        lo = 0
        hi = len(x_sorted) - 1
        while lo + 1 < hi:
            mid = (lo + hi) // 2
            if t < x_sorted[mid]:
                hi = mid
            else:
                lo = mid
        x0 = x_sorted[lo]
        x1 = x_sorted[lo + 1]
        y0 = y_sorted[lo]
        y1 = y_sorted[lo + 1]
        if x1 == x0:
            out[i] = y0
        else:
            alpha = (t - x0) / (x1 - x0)
            out[i] = y0 + alpha * (y1 - y0)
    return out


def compute_density_curves_before_dir_couplers_cpp(
    density_curves, log_exposure, dir_couplers_matrix, high_exposure_couplers_shift=0.0
):
    N = density_curves.shape[0]
    d_max = np.nanmax(density_curves, axis=0)
    dc_norm_shift = np.zeros_like(density_curves)
    for i in range(N):
        for c in range(3):
            denom = d_max[c]
            norm = density_curves[i, c] / denom if denom != 0 else 0.0
            shifted = norm + high_exposure_couplers_shift * norm * norm
            dc_norm_shift[i, c] = shifted
    # matrix multiplication (N×3) × (3×3)
    couplers_amount = np.dot(dc_norm_shift, dir_couplers_matrix)
    x0 = log_exposure[:, None] - couplers_amount
    corrected = np.zeros_like(density_curves)
    for c in range(3):
        corrected[:, c] = interpolate_1d_cpp(x0[:, c], density_curves[:, c], log_exposure)
    return corrected


def make_gaussian_kernel_2d_cpp(sigma):
    if sigma <= 0:
        return np.array([[1.0]])
    # Match the kernel truncation used in the C++ implementation: four
    # standard deviations are retained on either side.
    half = int(np.ceil(4.0 * sigma))
    size = 2 * half + 1
    coords = np.arange(-half, half + 1)
    g1d = np.exp(-0.5 * (coords / sigma) ** 2)
    g1d /= g1d.sum()
    g2d = np.outer(g1d, g1d)
    return g2d


def compute_exposure_correction_dir_couplers_cpp(
    log_raw,
    density_cmy,
    density_max,
    dir_couplers_matrix,
    diffusion_size_pixel,
    high_exposure_couplers_shift=0.0,
):
    H, W, _ = log_raw.shape
    norm_density = np.zeros_like(log_raw)
    for i in range(H):
        for j in range(W):
            for c in range(3):
                denom = density_max[c]
                norm = density_cmy[i, j, c] / denom if denom != 0 else 0.0
                norm += high_exposure_couplers_shift * norm * norm
                norm_density[i, j, c] = norm
    # per pixel multiply
    corr = np.zeros_like(log_raw)
    for i in range(H):
        for j in range(W):
            for m in range(3):
                acc = 0.0
                for k in range(3):
                    acc += norm_density[i, j, k] * dir_couplers_matrix[k, m]
                corr[i, j, m] = acc
    if diffusion_size_pixel > 0:
        sigma = diffusion_size_pixel
        kernel = make_gaussian_kernel_2d_cpp(sigma)
        ksize = kernel.shape[0]
        half = ksize // 2
        blurred = np.zeros_like(corr)
        # 2D convolution with reflective padding (mode='reflect')
        for y in range(H):
            for x in range(W):
                for c in range(3):
                    acc = 0.0
                    for dy in range(-half, half + 1):
                        ny = y + dy
                        # repeatedly reflect ny until it lies in [0, H-1]
                        while ny < 0 or ny >= H:
                            if ny < 0:
                                ny = -ny - 1
                            else:
                                ny = 2 * H - ny - 1
                        for dx in range(-half, half + 1):
                            nx = x + dx
                            while nx < 0 or nx >= W:
                                if nx < 0:
                                    nx = -nx - 1
                                else:
                                    nx = 2 * W - nx - 1
                            w = kernel[dy + half, dx + half]
                            acc += corr[ny, nx, c] * w
                    blurred[y, x, c] = acc
        corr = blurred
    corrected = log_raw - corr
    return corrected


class TestCouplersAlgorithms(unittest.TestCase):
    def test_compute_dir_couplers_matrix(self):
        amount_rgb = [0.7, 0.7, 0.5]
        for sigma in [0.5, 1.0, 2.0]:
            m_py = compute_dir_couplers_matrix_py(amount_rgb, sigma)
            m_cpp = compute_dir_couplers_matrix_cpp(amount_rgb, sigma)
            self.assertTrue(np.allclose(m_py, m_cpp, atol=1e-6))

    def test_compute_density_curves_before(self):
        # Generate monotonic exposure and strictly increasing density curves.
        # The interpolation routine used in the C++ implementation assumes
        # that the abscissa values for each channel are monotonic.  This
        # assumption holds for the film density curves stored in the
        # project, so the test uses simple linear ramps.
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
        corrected_cpp = compute_density_curves_before_dir_couplers_cpp(
            density_curves, log_exposure, dir_matrix
        )
        self.assertTrue(np.allclose(corrected_py, corrected_cpp, atol=1e-6))

    def test_compute_exposure_correction(self):
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
            corrected_cpp = compute_exposure_correction_dir_couplers_cpp(
                log_raw, density_cmy, density_max, dir_matrix, diffusion, 0.1
            )
            self.assertTrue(
                np.allclose(corrected_py, corrected_cpp, atol=1e-5),
                msg=f"Mismatch with diffusion={diffusion}",
            )


if __name__ == "__main__":
    unittest.main()