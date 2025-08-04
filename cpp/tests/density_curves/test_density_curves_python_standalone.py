#!/usr/bin/env python3

import numpy as np
import scipy.stats
from scipy.interpolate import interp1d

def print_vector(vec, name):
    print(f"{name}: [{', '.join([f'{x:.10f}' for x in vec])}]")

def print_matrix(mat, name):
    print(f"{name} ({mat.shape[0]}x{mat.shape[1]}):")
    for i, row in enumerate(mat):
        print(f"  Row {i}: [{', '.join([f'{x:.10f}' for x in row])}]")

def density_curve_model_norm_cdfs_py(loge, x, type='negative', number_of_layers=3):
    """Python implementation matching the C++ version"""
    centers = x[0:3]
    amplitudes = x[3:6]
    sigmas = x[6:9]
    
    dloge_curve = np.zeros(loge.shape)
    for i, (center, amplitude, sigma) in enumerate(zip(centers, amplitudes, sigmas)):
        if i <= number_of_layers - 1:
            if type == 'positive':
                dloge_curve += scipy.stats.norm.cdf(-(loge - center) / sigma) * amplitude
            else:
                dloge_curve += scipy.stats.norm.cdf((loge - center) / sigma) * amplitude
    return dloge_curve

def distribution_model_norm_cdfs_py(loge, x, number_of_layers=3):
    """Python implementation matching the C++ version"""
    centers = x[0:3]
    amplitudes = x[3:6]
    sigmas = x[6:9]
    
    distribution = np.zeros((loge.shape[0], 3))
    for i, (center, amplitude, sigma) in enumerate(zip(centers, amplitudes, sigmas)):
        if i <= number_of_layers - 1:
            distribution[:, i] += scipy.stats.norm.pdf((loge - center) / sigma) * amplitude / sigma
    return distribution

def compute_density_curves_py(log_exposure, parameters, type='negative', model='norm_cdfs'):
    """Python implementation matching the C++ version"""
    density_out = np.zeros((np.size(log_exposure), 3))
    if model == 'norm_cdfs':
        model_function = density_curve_model_norm_cdfs_py
    for i in range(3):
        density_out[:, i] = model_function(log_exposure, parameters[i], type)
    return density_out

def interpolate_exposure_to_density_py(log_exposure_rgb, density_curves, log_exposure, gamma_factor):
    """Python implementation matching the C++ version"""
    if np.size(gamma_factor) == 1:
        gamma_factor = [gamma_factor, gamma_factor, gamma_factor]
    gamma_factor = np.array(gamma_factor)
    
    density_cmy = np.zeros((log_exposure_rgb.shape[0], log_exposure_rgb.shape[1]))
    
    for channel in range(3):
        # Clamp to endpoints
        x = log_exposure_rgb[:, channel] / gamma_factor[channel]
        x = np.clip(x, log_exposure[0], log_exposure[-1])
        
        # Linear interpolation
        density_cmy[:, channel] = np.interp(x, log_exposure, density_curves[:, channel])
    
    return density_cmy

def apply_gamma_shift_correction_py(log_exposure, density_curves, gamma_correction, log_exposure_correction):
    """Python implementation matching the C++ version"""
    dc = density_curves
    le = log_exposure
    gc = gamma_correction
    les = log_exposure_correction
    dc_out = np.zeros_like(dc)
    
    for i in range(3):
        srcx = le / gc[i] + les[i]
        dc_out[:, i] = np.interp(le, srcx, dc[:, i])
    
    return dc_out

def main():
    print("=== Python Density Curves Test Results ===")
    print()

    # Test 1: Fixed input log-exposure grid
    print("Test 1: density_curve_model_norm_cdfs")
    print("=====================================")
    
    loge = np.array([-2.0 + 0.5 * i for i in range(11)])  # [-2.0, -1.5, ..., 3.0]
    print_vector(loge, "Input log_exposure")
    
    # Default parameters (matching C++ defaults)
    x = [0.0, 1.0, 2.0,  # centers
         0.5, 0.5, 0.5,  # amplitudes
         0.3, 0.5, 0.7]  # sigmas
    
    print(f"Parameters: center=[{x[0]}, {x[1]}, {x[2]}]")
    print(f"           amplitude=[{x[3]}, {x[4]}, {x[5]}]")
    print(f"           sigma=[{x[6]}, {x[7]}, {x[8]}]")
    
    # Test negative curve
    negative_curve = density_curve_model_norm_cdfs_py(loge, x, type='negative', number_of_layers=3)
    print_vector(negative_curve, "Negative curve output")
    
    # Test positive curve
    positive_curve = density_curve_model_norm_cdfs_py(loge, x, type='positive', number_of_layers=3)
    print_vector(positive_curve, "Positive curve output")
    
    print()

    # Test 2: distribution_model_norm_cdfs
    print("Test 2: distribution_model_norm_cdfs")
    print("====================================")
    
    distribution = distribution_model_norm_cdfs_py(loge, x, number_of_layers=3)
    print_matrix(distribution, "Distribution matrix")
    
    print()

    # Test 3: compute_density_curves (3-channel)
    print("Test 3: compute_density_curves (3-channel)")
    print("==========================================")
    
    parameters = [x, x, x]  # Same parameters for all 3 channels
    density_curves = compute_density_curves_py(loge, parameters, type='negative', model='norm_cdfs')
    print_matrix(density_curves, "3-channel density curves")
    
    print()

    # Test 4: interpolate_exposure_to_density
    print("Test 4: interpolate_exposure_to_density")
    print("=======================================")
    
    # Create a simple test matrix (2x3)
    log_exposure_rgb = np.array([[-1.0, -0.5, 0.0],
                                 [1.0, 1.5, 2.0]])
    
    gamma_factor = np.array([1.0, 1.0, 1.0])
    
    interpolated = interpolate_exposure_to_density_py(log_exposure_rgb, density_curves, loge, gamma_factor)
    print_matrix(log_exposure_rgb, "Input log_exposure_rgb")
    print_matrix(interpolated, "Interpolated density")
    
    print()

    # Test 5: apply_gamma_shift_correction
    print("Test 5: apply_gamma_shift_correction")
    print("====================================")
    
    gamma_correction = np.array([1.1, 0.9, 1.0])
    log_exposure_correction = np.array([0.1, -0.1, 0.0])
    
    corrected = apply_gamma_shift_correction_py(loge, density_curves, gamma_correction, log_exposure_correction)
    print_matrix(corrected, "Gamma-shift corrected density curves")
    
    print()

    # Test 6: GPU vs CPU comparison (simulated)
    print("Test 6: GPU vs CPU comparison")
    print("=============================")
    
    # In Python, we just use the same function (no GPU)
    gpu_curve = density_curve_model_norm_cdfs_py(loge, x, type='negative', number_of_layers=3)
    
    print("Ran on: CPU fallback")
    print_vector(gpu_curve, "GPU/CPU curve output")
    
    # Compare with CPU (should be identical)
    max_diff = np.max(np.abs(negative_curve - gpu_curve))
    print(f"Max absolute difference (CPU vs GPU): {max_diff:.15f}")
    
    print()
    print("=== Test completed ===")

if __name__ == "__main__":
    main() 