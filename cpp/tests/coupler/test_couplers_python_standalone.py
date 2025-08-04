"""
Python standalone test for couplers with fixed input data.
This script uses the same input data as the C++ test for comparison.
"""

import numpy as np
from scipy.ndimage import gaussian_filter

try:
    from opt_einsum import contract
except ImportError:
    def contract(subscripts, *operands):
        return np.einsum(subscripts, *operands)


def compute_dir_couplers_matrix_py(amount_rgb, layer_diffusion):
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


def print_matrix(matrix, name):
    """Print a 3x3 matrix in the same format as C++."""
    print(f"{name}:")
    for i in range(3):
        for j in range(3):
            print(f"{matrix[i, j]:.10f}", end="")
            if j < 2:
                print(", ", end="")
        print()
    print()


def print_2d_array(arr, name):
    """Print a 2D array in the same format as C++."""
    print(f"{name}:")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            print(f"{arr[i, j]:.10f}", end="")
            if j < arr.shape[1] - 1:
                print(", ", end="")
        print()
    print()


def print_3d_array(arr, name):
    """Print a 3D array in the same format as C++."""
    print(f"{name}:")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            print(f"[{i},{j}]: ", end="")
            for k in range(3):
                print(f"{arr[i, j, k]:.10f}", end="")
                if k < 2:
                    print(", ", end="")
            print()
    print()


if __name__ == "__main__":
    print("=== Python Couplers Test Results ===")
    print()
    
    # Test 1: compute_dir_couplers_matrix
    print("Test 1: compute_dir_couplers_matrix")
    print("==================================")
    
    amount_rgb = np.array([0.7, 0.7, 0.5])
    layer_diffusion = 1.0
    
    print(f"Input amount_rgb: [{amount_rgb[0]}, {amount_rgb[1]}, {amount_rgb[2]}]")
    print(f"Input layer_diffusion: {layer_diffusion}")
    print()
    
    matrix = compute_dir_couplers_matrix_py(amount_rgb, layer_diffusion)
    print_matrix(matrix, "Output matrix")
    
    # Test 2: compute_density_curves_before_dir_couplers
    print("Test 2: compute_density_curves_before_dir_couplers")
    print("=================================================")
    
    # Create fixed test data (same as C++)
    log_exposure = np.array([-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0])
    density_curves = np.array([
        [0.1, 0.2, 0.3],
        [0.2, 0.3, 0.4],
        [0.3, 0.4, 0.5],
        [0.4, 0.5, 0.6],
        [0.5, 0.6, 0.7],
        [0.6, 0.7, 0.8],
        [0.7, 0.8, 0.9]
    ])
    
    print("Input log_exposure: [", end="")
    for i in range(len(log_exposure)):
        print(log_exposure[i], end="")
        if i < len(log_exposure) - 1:
            print(", ", end="")
    print("]")
    print()
    
    print("Input density_curves:")
    print_2d_array(density_curves, "density_curves")
    
    corrected_curves = compute_density_curves_before_dir_couplers_py(
        density_curves, log_exposure, matrix, 0.1)
    print_2d_array(corrected_curves, "Output corrected_curves")
    
    # Test 3: compute_exposure_correction_dir_couplers
    print("Test 3: compute_exposure_correction_dir_couplers")
    print("================================================")
    
    # Create fixed 3D test data (2x2x3) - same as C++
    log_raw = np.array([
        [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
        [[0.7, 0.8, 0.9], [1.0, 1.1, 1.2]]
    ])
    
    density_cmy = np.array([
        [[0.2, 0.3, 0.4], [0.5, 0.6, 0.7]],
        [[0.8, 0.9, 1.0], [1.1, 1.2, 1.3]]
    ])
    
    density_max = np.array([2.0, 2.2, 2.4])
    diffusion_size_pixel = 1
    
    print("Input log_raw:")
    print_3d_array(log_raw, "log_raw")
    
    print("Input density_cmy:")
    print_3d_array(density_cmy, "density_cmy")
    
    print(f"Input density_max: [{density_max[0]}, {density_max[1]}, {density_max[2]}]")
    print(f"Input diffusion_size_pixel: {diffusion_size_pixel}")
    print()
    
    corrected_exposure = compute_exposure_correction_dir_couplers_py(
        log_raw, density_cmy, density_max, matrix, diffusion_size_pixel, 0.1)
    print_3d_array(corrected_exposure, "Output corrected_exposure") 