#!/usr/bin/env python3

import numpy as np
import os

def load_array_from_file(filename):
    """Load a numpy array from a text file with header"""
    if not os.path.exists(filename):
        print(f"File {filename} not found")
        return None
    
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Skip header lines (starting with #)
    data_lines = [line.strip() for line in lines if not line.startswith('#')]
    
    # Parse the data
    if not data_lines:
        return None
    
    # Check if it's 1D or 2D
    first_line = data_lines[0]
    if ' ' in first_line:
        # 2D array
        data = []
        for line in data_lines:
            if line.strip():
                row = [float(x) for x in line.split()]
                data.append(row)
        return np.array(data)
    else:
        # 1D array
        data = [float(x) for x in data_lines if x.strip()]
        return np.array(data)

def normalize_array_for_comparison(arr):
    """Normalize array to a standard format for comparison"""
    if arr is None:
        return None
    
    # If it's a 1D array, keep it as is
    if len(arr.shape) == 1:
        return arr
    
    # If it's a 2D array with 1 row, flatten it to 1D
    if len(arr.shape) == 2 and arr.shape[0] == 1:
        return arr.flatten()
    
    # If it's a 2D array with 1 column, flatten it to 1D
    if len(arr.shape) == 2 and arr.shape[1] == 1:
        return arr.flatten()
    
    # Otherwise, keep it as 2D
    return arr

def compare_arrays(python_file, cpp_file, name):
    """Compare Python and C++ arrays with proper shape handling"""
    print(f"\n=== Comparing {name} ===")
    
    python_data = load_array_from_file(python_file)
    cpp_data = load_array_from_file(cpp_file)
    
    if python_data is None:
        print(f"Python data not found in {python_file}")
        return False
    
    if cpp_data is None:
        print(f"C++ data not found in {cpp_file}")
        return False
    
    print(f"Python original shape: {python_data.shape}")
    print(f"C++ original shape: {cpp_data.shape}")
    
    # Normalize arrays for comparison
    python_norm = normalize_array_for_comparison(python_data)
    cpp_norm = normalize_array_for_comparison(cpp_data)
    
    print(f"Python normalized shape: {python_norm.shape}")
    print(f"C++ normalized shape: {cpp_norm.shape}")
    
    # Check if shapes match after normalization
    if python_norm.shape != cpp_norm.shape:
        print(f"Shape mismatch after normalization! Python: {python_norm.shape}, C++: {cpp_norm.shape}")
        return False
    
    # Compare values
    if python_norm.size == 0 or cpp_norm.size == 0:
        print("One or both arrays are empty")
        return True
    
    # Handle NaN values specially
    python_is_nan = np.isnan(python_norm)
    cpp_is_nan = np.isnan(cpp_norm)
    
    # Check if both arrays have NaN in the same positions
    nan_match = np.array_equal(python_is_nan, cpp_is_nan)
    
    if not nan_match:
        print("NaN patterns differ between Python and C++")
        print(f"Python NaN count: {np.sum(python_is_nan)}")
        print(f"C++ NaN count: {np.sum(cpp_is_nan)}")
        return False
    
    # Compare non-NaN values
    valid_mask = ~python_is_nan & ~cpp_is_nan
    if np.sum(valid_mask) == 0:
        print("No valid (non-NaN) values to compare")
        return True
    
    python_valid = python_norm[valid_mask]
    cpp_valid = cpp_norm[valid_mask]
    
    diff = np.abs(python_valid - cpp_valid)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Valid values compared: {len(python_valid)}")
    print(f"Max difference: {max_diff}")
    print(f"Mean difference: {mean_diff}")
    
    # Check if they're close enough (tolerance for floating point differences)
    tolerance = 1e-6
    if max_diff < tolerance:
        print("✓ Arrays match within tolerance")
        return True
    else:
        print("✗ Arrays differ significantly")
        
        # Show some sample differences
        if len(python_valid) > 0:
            print("Sample differences (first 10 valid elements):")
            for i in range(min(10, len(python_valid))):
                print(f"  [{i}]: Python={python_valid[i]:.10f}, C++={cpp_valid[i]:.10f}, diff={diff[i]:.10f}")
        
        return False

def main():
    print("Comparing Python and C++ load_agx_emulsion_data results...")
    print("Note: NumCpp always creates 2D arrays, so we normalize shapes for comparison")
    
    # List of arrays to compare
    arrays_to_compare = [
        ("log_sensitivity", "python_log_sensitivity.txt", "cpp_log_sensitivity.txt"),
        ("dye_density", "python_dye_density.txt", "cpp_dye_density.txt"),
        ("wavelengths", "python_wavelengths.txt", "cpp_wavelengths.txt"),
        ("density_curves", "python_density_curves.txt", "cpp_density_curves.txt"),
        ("log_exposure", "python_log_exposure.txt", "cpp_log_exposure.txt"),
    ]
    
    all_match = True
    
    for name, python_file, cpp_file in arrays_to_compare:
        if not compare_arrays(python_file, cpp_file, name):
            all_match = False
    
    print(f"\n=== Summary ===")
    if all_match:
        print("✓ All arrays match within tolerance")
    else:
        print("✗ Some arrays differ significantly")
    
    print(f"\n=== Key Differences Explained ===")
    print("1. Shape differences: NumCpp creates (1,N) arrays while NumPy creates (N,) arrays")
    print("2. Interpolation differences: Python uses Akima splines, C++ uses linear interpolation")
    print("3. NaN handling: Python may produce NaN values where C++ produces actual values")

if __name__ == "__main__":
    main() 