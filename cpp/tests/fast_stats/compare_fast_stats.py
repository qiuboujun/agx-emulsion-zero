#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the agx_emulsion module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../../agx_emulsion'))

def compute_mean_stddev_python(data):
    """Compute mean and standard deviation using Python/NumPy"""
    if len(data) == 0:
        return 0.0, 0.0
    
    # Convert to numpy array for consistency
    arr = np.array(data, dtype=np.float32)
    
    # Compute mean
    mean_val = np.mean(arr)
    
    # Compute population standard deviation
    stddev_val = np.std(arr, ddof=0)  # ddof=0 for population std
    
    return mean_val, stddev_val

def print_comparison_results():
    """Print comparison between Python and C++ results"""
    
    print("=== FastStats Python vs C++/CUDA Comparison ===")
    print("=" * 50)
    
    # Test case 1: Known values (same as C++ test)
    print("\n1. Test Case: Known Values")
    print("-" * 30)
    data1 = [1.5, 2.3, 3.7, 4.2, 5.8]
    expected_mean = 3.5
    expected_std = 1.5006665185843255
    
    print(f"Input data: {data1}")
    print(f"Expected mean: {expected_mean}")
    print(f"Expected std: {expected_std}")
    
    # Python computation
    py_mean, py_std = compute_mean_stddev_python(data1)
    print(f"\nPython results:")
    print(f"  Mean: {py_mean:.15f}")
    print(f"  Std:  {py_std:.15f}")
    
    # Differences from expected
    mean_diff = abs(py_mean - expected_mean)
    std_diff = abs(py_std - expected_std)
    print(f"\nDifferences from expected:")
    print(f"  Mean diff: {mean_diff:.15f}")
    print(f"  Std diff:  {std_diff:.15f}")
    
    # Test case 2: Large dataset (same as C++ test)
    print("\n\n2. Test Case: Large Dataset (Normal Distribution)")
    print("-" * 50)
    
    # Generate the same large dataset as C++ (using same seed)
    np.random.seed(42)
    large_data = np.random.normal(10.0, 2.0, 10000).astype(np.float32)
    
    print(f"Dataset size: {len(large_data)}")
    print(f"Expected mean: ~10.0")
    print(f"Expected std:  ~2.0")
    
    # Python computation
    py_large_mean, py_large_std = compute_mean_stddev_python(large_data)
    print(f"\nPython results:")
    print(f"  Mean: {py_large_mean:.15f}")
    print(f"  Std:  {py_large_std:.15f}")
    
    # Test case 3: Edge cases
    print("\n\n3. Test Case: Edge Cases")
    print("-" * 25)
    
    # Empty array
    empty_data = []
    py_empty_mean, py_empty_std = compute_mean_stddev_python(empty_data)
    print(f"Empty array:")
    print(f"  Mean: {py_empty_mean:.15f}")
    print(f"  Std:  {py_empty_std:.15f}")
    
    # Single element
    single_data = [42.0]
    py_single_mean, py_single_std = compute_mean_stddev_python(single_data)
    print(f"Single element [42.0]:")
    print(f"  Mean: {py_single_mean:.15f}")
    print(f"  Std:  {py_single_std:.15f}")
    
    # Test case 4: Fixed pattern (same as diffusion test)
    print("\n\n4. Test Case: Fixed Pattern (Sine/Cosine)")
    print("-" * 40)
    
    # Create the same fixed pattern as used in diffusion tests
    size = 100
    fixed_data = []
    for i in range(size):
        x = i / float(size - 1)
        val = 0.5 + 0.3 * np.sin(2 * np.pi * x) + 0.2 * np.cos(4 * np.pi * x)
        fixed_data.append(val)
    
    print(f"Fixed pattern size: {len(fixed_data)}")
    print(f"First 10 values: {fixed_data[:10]}")
    
    # Python computation
    py_fixed_mean, py_fixed_std = compute_mean_stddev_python(fixed_data)
    print(f"\nPython results:")
    print(f"  Mean: {py_fixed_mean:.15f}")
    print(f"  Std:  {py_fixed_std:.15f}")
    
    print("\n" + "=" * 50)
    print("Python computation complete!")
    print("Run the C++ test to compare results.")

if __name__ == "__main__":
    print_comparison_results() 