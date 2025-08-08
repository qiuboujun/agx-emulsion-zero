#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the agx_emulsion module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../agx_emulsion'))

def print_array_stats(arr, name):
    """Print array statistics"""
    if arr.size == 0:
        print(f"{name}: empty")
        return
    
    min_val = np.min(arr)
    max_val = np.max(arr)
    mean_val = np.mean(arr)
    
    print(f"{name}: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}")

def print_array_data(arr, name, max_elements=20):
    """Print array data with limited elements"""
    print(f"{name}: [", end="")
    for i in range(min(arr.size, max_elements)):
        print(f"{arr.flat[i]:.6f}", end="")
        if i < min(arr.size, max_elements) - 1:
            print(", ", end="")
    if arr.size > max_elements:
        print(f", ... (showing first {max_elements} of {arr.size} elements)", end="")
    print("]")

def get_fixed_random_values(count):
    """Get fixed random values for deterministic testing"""
    values = [
        0.123456, 0.234567, 0.345678, 0.456789, 0.567890,
        0.678901, 0.789012, 0.890123, 0.901234, 0.012345,
        0.111111, 0.222222, 0.333333, 0.444444, 0.555555,
        0.666666, 0.777777, 0.888888, 0.999999, 0.000001,
        0.101010, 0.202020, 0.303030, 0.404040, 0.505050,
        0.606060, 0.707070, 0.808080, 0.909090, 0.010101,
        0.121212, 0.232323, 0.343434, 0.454545, 0.565656,
        0.676767, 0.787878, 0.898989, 0.909090, 0.020202
    ]
    
    # Cycle through the fixed values
    result = []
    for i in range(count):
        result.append(values[i % len(values)])
    return result

def exposure_to_density_fixed_python(exposure):
    """Python implementation of exposure to density conversion"""
    # Simple characteristic curve: density = 1 - exp(-exposure)
    density = 1.0 - np.exp(-exposure)
    return np.clip(density, 0.0, 2.2)

def apply_grain_fixed_python(density, grain_params):
    """Python implementation of grain simulation using fixed arrays"""
    result = density.copy()
    
    # Get fixed random values
    fixed_rands = get_fixed_random_values(density.size * 2)
    rand_idx = 0
    
    od_particle = 0.22  # density_max / n_particles_per_pixel
    n_particles_per_pixel = 10.0
    
    # Process each pixel and channel
    for y in range(density.shape[0]):
        for x in range(density.shape[1]):
            for c in range(density.shape[2]):
                d = density[y, x, c]
                p = np.clip(d / 2.2, 1e-6, 1.0 - 1e-6)
                saturation = 1.0 - p * grain_params['grain_uniformity'][c] * (1.0 - 1e-6)
                lambda_val = n_particles_per_pixel / max(1e-6, saturation)
                
                # Use fixed values instead of RNG
                rand1 = fixed_rands[rand_idx]
                rand_idx += 1
                rand2 = fixed_rands[rand_idx]
                rand_idx += 1
                
                # Simple Poisson approximation
                n = int(lambda_val * rand1)
                n = max(0, n)
                
                # Simple binomial approximation
                developed = int(n * p * rand2)
                developed = np.clip(developed, 0, n)
                
                grain_val = float(developed) * od_particle * saturation
                result[y, x, c] = d + grain_val
    
    return result

def apply_dir_couplers_fixed_python(density, dir_params):
    """Python implementation of DIR coupler simulation using fixed arrays"""
    if not dir_params['enable_dir_couplers']:
        return density
    
    result = density.copy()
    
    # Get fixed random values
    fixed_rands = get_fixed_random_values(density.size)
    rand_idx = 0
    
    # DIR coupler matrix (simplified 3x3 identity with some cross-coupling)
    dir_matrix = np.array([
        [1.0, -0.1, -0.1],
        [-0.1, 1.0, -0.1],
        [-0.1, -0.1, 1.0]
    ])
    
    # Process each pixel
    for y in range(density.shape[0]):
        for x in range(density.shape[1]):
            input_density = np.zeros(3)
            output_density = np.zeros(3)
            
            # Get input densities for RGB channels
            for c in range(3):
                input_density[c] = density[y, x, c]
            
            # Apply DIR coupler matrix
            for i in range(3):
                for j in range(3):
                    output_density[i] += dir_matrix[i, j] * input_density[j] * dir_params['dir_coupler_scale']
            
            # Add fixed noise and write output
            for c in range(3):
                noise = fixed_rands[rand_idx] * 0.01  # Small fixed noise
                rand_idx += 1
                output_val = np.clip(output_density[c] + noise, 0.0, 2.2)
                result[y, x, c] = output_val
    
    return result

def develop_film_fixed_python(exposure, grain_params, dir_params):
    """Python implementation of full film development using fixed arrays"""
    # Step 1: Convert exposure to density
    density = exposure_to_density_fixed_python(exposure)
    
    # Step 2: Apply DIR couplers (if enabled)
    if dir_params['enable_dir_couplers']:
        density = apply_dir_couplers_fixed_python(density, dir_params)
    
    # Step 3: Apply grain
    density = apply_grain_fixed_python(density, grain_params)
    
    return density

def test_fixed_input_comparison():
    """Test with fixed input data for comparison with C++"""
    print("=== Emulsion Model: Python Fixed Input Comparison (No RNG) ===")
    print("=" * 55)
    
    # Test 1: Simple 2x2x1x3 fixed input
    print("\n1. Test Case: Simple 2x2x1x3 Fixed Input")
    print("-" * 45)
    
    fixed_input = np.array([
        [[0.5, 0.6, 0.7], [1.0, 1.1, 1.2]],
        [[1.5, 1.6, 1.7], [2.0, 2.1, 2.2]]
    ], dtype=np.float32)
    
    print("Input data (RGB):")
    for y in range(2):
        for x in range(2):
            print(f"  ({x},{y}): [{fixed_input[y, x, 0]:.1f}, {fixed_input[y, x, 1]:.1f}, {fixed_input[y, x, 2]:.1f}]")
    
    print_array_stats(fixed_input, "Input image")
    
    # Test parameters
    grain_params = {
        'pixel_size_um': 10.0,
        'agx_particle_area_um2': 0.2,
        'agx_particle_scale': [1.0, 0.8, 3.0],
        'density_min': [0.03, 0.06, 0.04],
        'density_max_curves': [2.2, 2.2, 2.2],
        'grain_uniformity': [0.98, 0.98, 0.98],
        'grain_blur': 0.0,  # No blur for exact comparison
        'n_sub_layers': 1,
        'fixed_seed': True,
        'seed': 42
    }
    
    dir_params = {
        'dir_coupler_scale': 1.0,
        'dir_coupler_blur': 0.0,  # No blur for exact comparison
        'enable_dir_couplers': True,
        'seed': 123
    }
    
    # Python computation with fixed arrays
    python_result = develop_film_fixed_python(fixed_input, grain_params, dir_params)
    
    print_array_stats(python_result, "Python result (fixed arrays)")
    print_array_data(python_result, "Python result (all values)")
    
    # Test 2: Larger fixed pattern (4x4x1x3)
    print("\n\n2. Test Case: Larger Fixed Pattern (4x4x1x3)")
    print("-" * 50)
    
    larger_input = np.zeros((4, 4, 3), dtype=np.float32)
    # Create the same fixed pattern as C++
    for y in range(4):
        for x in range(4):
            for c in range(3):
                val = 0.5 + 0.3 * np.sin(2.0 * np.pi * x / 4.0) + 0.2 * np.cos(2.0 * np.pi * y / 4.0) + 0.1 * (c + 1)
                larger_input[y, x, c] = val
    
    print("Input pattern (4x4x3) - showing first channel:")
    for y in range(4):
        print("  ", end="")
        for x in range(4):
            print(f"{larger_input[y, x, 0]:.3f} ", end="")
        print()
    
    print_array_stats(larger_input, "Larger input")
    
    # Python computation with fixed arrays
    python_larger = develop_film_fixed_python(larger_input, grain_params, dir_params)
    
    print_array_stats(python_larger, "Python larger result (fixed arrays)")
    print_array_data(python_larger, "Python larger result (all values)")
    
    # Test 3: Individual component testing
    print("\n\n3. Test Case: Individual Component Testing")
    print("-" * 40)
    
    # Test exposure to density conversion
    density_only = exposure_to_density_fixed_python(fixed_input)
    print_array_stats(density_only, "Exposure to density only")
    print_array_data(density_only, "Density values (all)")
    
    # Test grain only
    grain_only = apply_grain_fixed_python(density_only, grain_params)
    print_array_stats(grain_only, "Grain only")
    print_array_data(grain_only, "Grain values (all)")
    
    # Test DIR couplers only
    dir_only = apply_dir_couplers_fixed_python(density_only, dir_params)
    print_array_stats(dir_only, "DIR couplers only")
    print_array_data(dir_only, "DIR coupler values (all)")
    
    # Test 4: Statistics computation
    print("\n\n4. Test Case: Statistics Computation")
    print("-" * 30)
    
    # Create the same larger test image for statistics
    stats_input = np.zeros((8, 8, 3), dtype=np.float32)
    for y in range(8):
        for x in range(8):
            for c in range(3):
                val = 0.5 + 0.3 * np.sin(2.0 * np.pi * x / 8.0) + 0.2 * np.cos(2.0 * np.pi * y / 8.0) + 0.1 * (c + 1)
                stats_input[y, x, c] = val
    
    # Compute full emulsion development
    emulsion_result = develop_film_fixed_python(stats_input, grain_params, dir_params)
    
    # Compute statistics using NumPy for each channel
    for c in range(3):
        channel_data = emulsion_result[:, :, c].flatten()
        mean_val = np.mean(channel_data)
        std_val = np.std(channel_data, ddof=0)  # Population std
        
        print(f"Channel {c} statistics (using NumPy):")
        print(f"  Mean: {mean_val:.15f}")
        print(f"  Std:  {std_val:.15f}")
    
    print_array_stats(emulsion_result, "Full emulsion result")
    
    # Test 5: Film characteristic curves
    print("\n\n5. Test Case: Film Characteristic Curves")
    print("-" * 35)
    
    # Create simple characteristic curves
    log_exposure = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    density_curves = np.array([
        [0.1, 0.3, 0.8, 1.5, 2.0],  # Red channel
        [0.1, 0.4, 1.0, 1.8, 2.1],  # Green channel
        [0.1, 0.2, 0.6, 1.2, 1.9]   # Blue channel
    ])
    
    # Test exposure to density conversion using interpolation
    for c in range(3):
        exp_val = 0.5
        # Simple linear interpolation
        for i in range(len(log_exposure) - 1):
            if exp_val >= log_exposure[i] and exp_val <= log_exposure[i + 1]:
                t = (exp_val - log_exposure[i]) / (log_exposure[i + 1] - log_exposure[i])
                den_val = density_curves[c, i] + t * (density_curves[c, i + 1] - density_curves[c, i])
                print(f"Channel {c}: exposure={exp_val} -> density={den_val:.6f}")
                break
    
    print("\n" + "=" * 55)
    print("Python fixed input comparison complete (No RNG)!")
    print("Compare with C++ results above.")

if __name__ == "__main__":
    test_fixed_input_comparison() 