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

def layer_particle_model_fixed_python(density, density_max=2.2, n_particles_per_pixel=10.0, 
                                     grain_uniformity=0.98, seed=0, blur_particle=0.0):
    """Python implementation of the particle model using fixed arrays instead of RNG"""
    # Ensure density is 2D
    if density.ndim == 1:
        density = density.reshape(-1, 1)
    
    H, W = density.shape[:2]
    out = np.zeros_like(density, dtype=np.float32)
    
    od_particle = density_max / max(1.0, n_particles_per_pixel)
    
    # Get fixed random values
    fixed_rands = get_fixed_random_values(H * W * 2)  # Need 2 values per pixel
    rand_idx = 0
    
    for y in range(H):
        for x in range(W):
            d = density[y, x]
            p = np.clip(d / density_max, 1e-6, 1.0 - 1e-6)
            saturation = 1.0 - p * grain_uniformity * (1.0 - 1e-6)
            lambda_val = n_particles_per_pixel / max(1e-6, saturation)
            
            # Use fixed values instead of RNG
            rand1 = fixed_rands[rand_idx]
            rand_idx += 1
            rand2 = fixed_rands[rand_idx]
            rand_idx += 1
            
            # Simple Poisson approximation using fixed random value
            n = int(lambda_val * rand1)
            n = max(0, n)
            
            # Simple binomial approximation using fixed random value
            developed = int(n * p * rand2)
            developed = np.clip(developed, 0, n)
            
            val = float(developed) * od_particle * saturation
            out[y, x] = val
    
    return out

def apply_grain_to_density_fixed_python(density_cmy, pixel_size_um=10.0, agx_particle_area_um2=0.2,
                                       agx_particle_scale=(1.0, 0.8, 3.0), density_min=(0.03, 0.06, 0.04),
                                       density_max_curves=(2.2, 2.2, 2.2), grain_uniformity=(0.98, 0.98, 0.98),
                                       grain_blur=1.0, n_sub_layers=1, fixed_seed=False):
    """Python implementation of apply_grain_to_density using fixed arrays"""
    H, W, C = density_cmy.shape
    assert C == 3
    
    work = density_cmy.copy()
    # Add density_min per channel
    for c in range(3):
        work[:, :, c] += density_min[c]
    
    # Derived params
    density_max = [density_max_curves[c] + density_min[c] for c in range(3)]
    
    pixel_area_um2 = pixel_size_um * pixel_size_um
    agx_particle_area_um2_rgb = [agx_particle_area_um2 * agx_particle_scale[c] for c in range(3)]
    npp = [pixel_area_um2 / agx_particle_area_um2_rgb[c] for c in range(3)]
    
    if n_sub_layers > 1:
        npp = [npp[c] / float(n_sub_layers) for c in range(3)]
    
    out = np.zeros((H, W, 3), dtype=np.float32)
    for ch in range(3):
        acc = np.zeros((H, W), dtype=np.float32)
        for sl in range(n_sub_layers):
            # Extract channel to 1-channel density
            d1 = work[:, :, ch]
            
            seed = 0 if fixed_seed else (ch + sl * 10)
            g = layer_particle_model_fixed_python(d1, density_max[ch], npp[ch], 
                                                grain_uniformity[ch], seed, 0.0)
            
            # Accumulate
            acc += g
        
        # Average sublayers
        inv = 1.0 / max(1, n_sub_layers)
        out[:, :, ch] = acc * inv
    
    # Subtract density_min
    for c in range(3):
        out[:, :, c] -= density_min[c]
    
    return out

def test_fixed_input_comparison():
    """Test with fixed input data for comparison with C++"""
    print("=== Grain Model: Python Fixed Input Comparison (No RNG) ===")
    print("=" * 55)
    
    # Test 1: Simple 2x2x1 fixed input
    print("\n1. Test Case: Simple 2x2x1 Fixed Input")
    print("-" * 40)
    
    fixed_input = np.array([[0.5, 1.0], [1.5, 2.0]], dtype=np.float32)
    print("Input data: [0.5, 1.0, 1.5, 2.0]")
    print_array_stats(fixed_input, "Input image")
    
    # Python computation with fixed arrays
    python_result = layer_particle_model_fixed_python(fixed_input,
                                                     density_max=2.2,
                                                     n_particles_per_pixel=10.0,
                                                     grain_uniformity=0.98,
                                                     seed=42,  # Fixed seed
                                                     blur_particle=0.0)
    
    print_array_stats(python_result, "Python result (fixed arrays)")
    print_array_data(python_result, "Python result (all values)")
    
    # Test 2: Larger fixed pattern (4x4x1)
    print("\n\n2. Test Case: Larger Fixed Pattern (4x4x1)")
    print("-" * 45)
    
    larger_input = np.zeros((4, 4), dtype=np.float32)
    # Create the same fixed pattern as C++
    for y in range(4):
        for x in range(4):
            val = 0.5 + 0.3 * np.sin(2.0 * np.pi * x / 4.0) + 0.2 * np.cos(2.0 * np.pi * y / 4.0)
            larger_input[y, x] = val
    
    print("Input pattern (4x4):")
    for y in range(4):
        print("  ", end="")
        for x in range(4):
            print(f"{larger_input[y, x]:.3f} ", end="")
        print()
    
    print_array_stats(larger_input, "Larger input")
    
    # Python computation with fixed arrays
    python_larger = layer_particle_model_fixed_python(larger_input,
                                                     density_max=2.2,
                                                     n_particles_per_pixel=10.0,
                                                     grain_uniformity=0.98,
                                                     seed=123,  # Fixed seed
                                                     blur_particle=0.0)
    
    print_array_stats(python_larger, "Python larger result (fixed arrays)")
    print_array_data(python_larger, "Python larger result (all values)")
    
    # Test 3: 3-channel fixed input
    print("\n\n3. Test Case: 3-Channel Fixed Input")
    print("-" * 35)
    
    rgb_input = np.array([
        [[0.5, 0.6, 0.7], [1.0, 1.1, 1.2]],
        [[1.5, 1.6, 1.7], [2.0, 2.1, 2.2]]
    ], dtype=np.float32)
    
    print("RGB input data:")
    for y in range(2):
        for x in range(2):
            print(f"  ({x},{y}): [{rgb_input[y, x, 0]:.1f}, {rgb_input[y, x, 1]:.1f}, {rgb_input[y, x, 2]:.1f}]")
    
    print_array_stats(rgb_input, "RGB input")
    
    # Python computation for RGB with fixed arrays
    python_rgb = apply_grain_to_density_fixed_python(rgb_input,
                                                    pixel_size_um=10.0,
                                                    agx_particle_area_um2=0.2,
                                                    agx_particle_scale=(1.0, 0.8, 3.0),
                                                    density_min=(0.03, 0.06, 0.04),
                                                    density_max_curves=(2.2, 2.2, 2.2),
                                                    grain_uniformity=(0.98, 0.98, 0.98),
                                                    grain_blur=0.0,
                                                    n_sub_layers=1,
                                                    fixed_seed=True)
    
    print_array_stats(python_rgb, "Python RGB result (fixed arrays)")
    print_array_data(python_rgb, "Python RGB result (all values)")
    
    # Test 4: Statistics computation
    print("\n\n4. Test Case: Statistics Computation")
    print("-" * 30)
    
    # Create the same larger test image for statistics
    stats_input = np.zeros((8, 8), dtype=np.float32)
    for y in range(8):
        for x in range(8):
            val = 0.5 + 0.3 * np.sin(2.0 * np.pi * x / 8.0) + 0.2 * np.cos(2.0 * np.pi * y / 8.0)
            stats_input[y, x] = val
    
    # Compute grain with fixed arrays
    grain_result = layer_particle_model_fixed_python(stats_input,
                                                    density_max=2.2,
                                                    n_particles_per_pixel=10.0,
                                                    grain_uniformity=0.98,
                                                    seed=42,
                                                    blur_particle=0.0)
    
    # Compute statistics using NumPy
    mean_val = np.mean(grain_result)
    std_val = np.std(grain_result, ddof=0)  # Population std
    
    print("Grain result statistics (using NumPy):")
    print(f"  Mean: {mean_val:.15f}")
    print(f"  Std:  {std_val:.15f}")
    
    print_array_stats(grain_result, "Grain result")
    
    print("\n" + "=" * 55)
    print("Python fixed input comparison complete (No RNG)!")
    print("Compare with C++ results above.")

if __name__ == "__main__":
    test_fixed_input_comparison() 