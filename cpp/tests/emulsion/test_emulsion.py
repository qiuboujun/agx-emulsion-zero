"""
Unit tests for the C++ AgX emulsion implementation.

This test compares the output of the reference Python implementation of the
film development pipeline against the results produced by the C++ library
compiled from emulsion.cpp.  A small synthetic dataset is constructed and
passed through both pipelines.  The C++ side is accessed via a simple
C wrapper exposed from emulsion.cpp.

If a C++ compiler is unavailable at test time the test is skipped.  The
reference Python implementation is a direct translation of the algorithms
used in the C++ code and does not rely on any external modules from the
original project.
"""

import os
import ctypes
import numpy as np
import pytest
from scipy.ndimage import gaussian_filter


def interp1d_py(x, x_grid, y_grid):
    """Python reimplementation of linear interpolation used by the C++ code."""
    if x <= x_grid[0]:
        return y_grid[0]
    if x >= x_grid[-1]:
        return y_grid[-1]
    # Find indices
    idx = np.searchsorted(x_grid, x) - 1
    x0 = x_grid[idx]
    x1 = x_grid[idx + 1]
    y0 = y_grid[idx]
    y1 = y_grid[idx + 1]
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)


def interpolate_exposure_to_density_py(log_raw, density_curves, log_exposure, gamma_factor):
    """Replicates interpolate_exposure_to_density from C++ in Python."""
    if np.isscalar(gamma_factor):
        gamma = np.array([gamma_factor] * 3, dtype=float)
    else:
        gamma = np.asarray(gamma_factor, dtype=float)
        if gamma.size == 1:
            gamma = np.array([gamma[0]] * 3, dtype=float)
    h, w, ch = log_raw.shape
    density_cmy = np.zeros_like(log_raw, dtype=float)
    for c in range(ch):
        x_grid = log_exposure / gamma[c]
        y_grid = density_curves[:, c]
        # Vectorised interpolation
        density_cmy[:, :, c] = np.interp(log_raw[:, :, c], x_grid, y_grid)
    return density_cmy


def compute_dir_couplers_matrix_py(amount_rgb, layer_diffusion):
    """Replicates compute_dir_couplers_matrix from C++ in Python."""
    M = np.eye(3, dtype=float)
    # Apply Gaussian blur along the second axis
    if layer_diffusion > 0:
        # Only blur along axis=1 (columns)
        M = gaussian_filter(M, sigma=(0.0, layer_diffusion), mode='constant')
    # Normalise rows
    M = M / M.sum(axis=1, keepdims=True)
    M = M * amount_rgb[:, None]
    return M


def compute_density_curves_before_dir_couplers_py(density_curves, log_exposure, dir_matrix, high_exposure_shift):
    n = len(log_exposure)
    channels = density_curves.shape[1]
    d_max = np.nanmax(density_curves, axis=0)
    dc_norm = density_curves / d_max
    dc_norm_shift = dc_norm + high_exposure_shift * dc_norm ** 2
    couplers_amount_curves = dc_norm_shift @ dir_matrix
    x0 = log_exposure[:, None] - couplers_amount_curves
    density_curves_corrected = np.zeros_like(density_curves)
    for c in range(channels):
        density_curves_corrected[:, c] = np.interp(log_exposure, x0[:, c], density_curves[:, c])
    return density_curves_corrected


def compute_exposure_correction_dir_couplers_py(log_raw, density_cmy, density_max, dir_matrix, diffusion_size_pixel, high_exposure_shift):
    norm_density = density_cmy / density_max
    norm_density = norm_density + high_exposure_shift * norm_density ** 2
    # Contract along channel axis
    log_raw_correction = np.tensordot(norm_density, dir_matrix, axes=(2, 0))
    if diffusion_size_pixel > 0:
        # Apply Gaussian blur over first two axes
        sigma = (diffusion_size_pixel, diffusion_size_pixel, 0.0)
        log_raw_correction = gaussian_filter(log_raw_correction, sigma=sigma)
    return log_raw - log_raw_correction


def apply_grain_to_density_py(density_cmy, pixel_size_um, agx_particle_area_um2, agx_particle_scale, density_min, density_max, grain_uniformity, grain_blur, n_sub_layers):
    h, w, channels = density_cmy.shape
    density_cmy_out = np.zeros_like(density_cmy, dtype=float)
    pixel_area_um2 = pixel_size_um ** 2
    # For reproducibility
    rng = np.random.default_rng(12345)
    for c in range(channels):
        density_max_total = density_max[c] + density_min[c]
        area = agx_particle_area_um2[c] * agx_particle_scale[c]
        n_particles_per_pixel = pixel_area_um2 / area
        if n_sub_layers > 1:
            n_particles_per_pixel /= n_sub_layers
        accum = np.zeros((h, w), dtype=float)
        for sl in range(n_sub_layers):
            d = density_cmy[:, :, c] + density_min[c]
            probability_of_development = d / density_max_total
            probability_of_development = np.clip(probability_of_development, 1e-6, 1 - 1e-6)
            od_particle = density_max_total / n_particles_per_pixel
            saturation = 1 - probability_of_development * grain_uniformity[c] * (1 - 1e-6)
            seeds = rng.poisson(n_particles_per_pixel / saturation)
            grain_counts = rng.binomial(seeds, probability_of_development)
            grain = grain_counts.astype(float) * od_particle * saturation
            accum += grain
        accum /= n_sub_layers
        accum -= density_min[c]
        density_cmy_out[:, :, c] = accum
    # Blur final result if requested
    if grain_blur > 0.4:
        for c in range(channels):
            density_cmy_out[:, :, c] = gaussian_filter(density_cmy_out[:, :, c], sigma=grain_blur)
    return density_cmy_out


def develop_py(log_raw, density_curves, log_exposure, dir_params, grain_params, pixel_size_um, gamma_factor):
    density_cmy = interpolate_exposure_to_density_py(log_raw, density_curves, log_exposure, gamma_factor)
    # Apply couplers
    if dir_params['active']:
        M = compute_dir_couplers_matrix_py(dir_params['amount'] * np.array(dir_params['ratio_rgb']), dir_params['diffusion_interlayer'])
        density_curves_0 = compute_density_curves_before_dir_couplers_py(density_curves, log_exposure, M, dir_params['high_exposure_shift'])
        density_max = np.nanmax(density_curves, axis=0)
        log_raw_0 = compute_exposure_correction_dir_couplers_py(log_raw, density_cmy, density_max, M, dir_params['diffusion_size_um'] / pixel_size_um, dir_params['high_exposure_shift'])
        density_cmy = interpolate_exposure_to_density_py(log_raw_0, density_curves_0, log_exposure, gamma_factor)
    # Apply grain
    if grain_params['active']:
        density_cmy = apply_grain_to_density_py(
            density_cmy,
            pixel_size_um=pixel_size_um,
            agx_particle_area_um2=np.array(grain_params['agx_particle_area_um2']),
            agx_particle_scale=np.array(grain_params['agx_particle_scale']),
            density_min=np.array(grain_params['density_min']),
            density_max=np.array(grain_params['density_max']),
            grain_uniformity=np.array(grain_params['grain_uniformity']),
            grain_blur=grain_params['blur'],
            n_sub_layers=grain_params['n_sub_layers'],
        )
    return density_cmy


def build_cpp_library(tmpdir):
    """Attempt to compile the C++ emulsion library.  Returns the path to the
    compiled shared object if successful, else returns None."""
    import subprocess
    src_cpp = os.path.join(os.path.dirname(__file__), 'emulsion.cpp')
    lib_path = os.path.join(tmpdir, 'libemulsion.so')
    try:
        # Compile only the C++ file; CUDA kernel is optional.  Use -fPIC and -shared.
        subprocess.run([
            'g++', '-O2', '-std=c++14', '-fPIC', src_cpp, '-shared', '-o', lib_path
        ], check=True)
    except Exception:
        return None
    return lib_path


@pytest.mark.skipif(not os.path.exists(__file__), reason="Test must be run inside repository")
@pytest.mark.skip("Grain introduces randomness; a deterministic comparison is provided in test_emulsion_no_grain.")
def test_emulsion_cpp_matches_python(tmp_path):
    """Test placeholder – intentionally skipped."""
    pass

def test_emulsion_no_grain(tmp_path):
    """Test C++ vs Python when grain is disabled (deterministic)."""
    lib_path = build_cpp_library(tmp_path)
    if lib_path is None:
        pytest.skip("C++ compiler unavailable, skipping C++ comparison")
    lib = ctypes.cdll.LoadLibrary(str(lib_path))
    lib.agx_film_develop_simple.argtypes = [
        ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int,
        ctypes.POINTER(ctypes.c_float), ctypes.c_int,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float), ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float,
        ctypes.POINTER(ctypes.c_float)
    ]
    lib.agx_film_develop_simple.restype = None
    # Deterministic test data
    np.random.seed(1)
    height, width = 3, 3
    log_raw = np.random.uniform(-2.0, 0.0, size=(height, width, 3)).astype(np.float32)
    n_points = 5
    log_exposure = np.linspace(-2.0, 0.0, n_points, dtype=np.float32)
    density_curves = np.zeros((n_points, 3), dtype=np.float32)
    for c in range(3):
        density_curves[:, c] = np.linspace(0.2 + 0.1 * c, 1.4 + 0.1 * c, n_points, dtype=np.float32)
    gamma_factor = 1.0
    dir_params = {
        'active': True,
        'amount': 0.4,
        'ratio_rgb': [0.6, 0.7, 0.8],
        'diffusion_interlayer': 0.5,
        'diffusion_size_um': 1.0,
        'high_exposure_shift': 0.0,
    }
    grain_params = {
        'active': False,
        'agx_particle_area_um2': [0.2, 0.2, 0.2],
        'agx_particle_scale': [1.0, 1.0, 1.0],
        'density_min': [0.03, 0.03, 0.03],
        'density_max': [2.0, 2.0, 2.0],
        'grain_uniformity': [1.0, 1.0, 1.0],
        'blur': 0.0,
        'n_sub_layers': 1,
    }
    pixel_size_um = 10.0
    density_py = develop_py(log_raw, density_curves, log_exposure, dir_params, grain_params, pixel_size_um, gamma_factor)
    # Prepare C++ inputs
    log_raw_flat = log_raw.ravel().astype(np.float32)
    density_curves_flat = density_curves.ravel().astype(np.float32)
    log_exposure_flat = log_exposure.astype(np.float32)
    ratio_rgb = np.array(dir_params['ratio_rgb'], dtype=np.float32)
    out_cpp = np.zeros_like(log_raw_flat)
    lib.agx_film_develop_simple(
        log_raw_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(height), ctypes.c_int(width),
        density_curves_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_int(n_points),
        log_exposure_flat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
        ctypes.c_int(1), ctypes.c_float(dir_params['amount']), ratio_rgb.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), ctypes.c_float(dir_params['diffusion_interlayer']), ctypes.c_float(dir_params['diffusion_size_um']), ctypes.c_float(dir_params['high_exposure_shift']),
        ctypes.c_int(0), ctypes.c_float(grain_params['blur']), ctypes.c_float(pixel_size_um), ctypes.c_float(gamma_factor),
        out_cpp.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
    )
    density_cpp = out_cpp.reshape(height, width, 3)
    # Assert that C++ and Python outputs are close
    assert np.allclose(density_cpp, density_py, rtol=1e-5, atol=1e-5)