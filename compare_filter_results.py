#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the agx_emulsion directory to the path
sys.path.insert(0, 'agx_emulsion')

from agx_emulsion.utils import io as py_io

def compare_arrays(py_array, cpp_array, tolerance=1e-4):
    """Compare Python 1D array with C++ 2D array (flattened to 1D)"""
    
    # Flatten C++ array to 1D for comparison
    cpp_flat = cpp_array.flatten()
    
    # Ensure both arrays have the same length
    if len(py_array) != len(cpp_flat):
        print(f"  ❌ Length mismatch: Python {len(py_array)} vs C++ {len(cpp_flat)}")
        return False
    
    # Compare values
    diff = np.abs(py_array - cpp_flat)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    if max_diff <= tolerance:
        print(f"  ✅ Match! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        return True
    else:
        print(f"  ❌ Mismatch! Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        print(f"     Python min/max: {np.min(py_array):.6f}/{np.max(py_array):.6f}")
        print(f"     C++ min/max: {np.min(cpp_flat):.6f}/{np.max(cpp_flat):.6f}")
        return False

def main():
    print("Comparing Python and C++ load_filter results...")
    
    # Test different filter configurations
    test_configs = [
        {'name': 'KG3', 'brand': 'schott', 'filter_type': 'heat_absorbing', 'percent_transmittance': False},
        {'name': 'KG5', 'brand': 'schott', 'filter_type': 'heat_absorbing', 'percent_transmittance': False},
        {'name': 'canon_24_f28_is', 'brand': 'canon', 'filter_type': 'lens_transmission', 'percent_transmittance': False},
    ]
    
    all_passed = True
    
    for config in test_configs:
        print(f"\n=== Testing {config['name']} ({config['brand']}) ===")
        
        try:
            # Get wavelengths from spectral shape
            wavelengths = py_io.SPECTRAL_SHAPE.wavelengths
            
            # Call Python function
            py_result = py_io.load_filter(
                wavelengths, 
                name=config['name'],
                brand=config['brand'], 
                filter_type=config['filter_type'],
                percent_transmittance=config['percent_transmittance']
            )
            
            print(f"Python result shape: {py_result.shape}")
            print(f"Python result dtype: {py_result.dtype}")
            print(f"Python min/max: {np.min(py_result):.6f}/{np.max(py_result):.6f}")
            
            # Load C++ result from file
            cpp_filename = f"cpp_filter_{config['name']}_{config['brand']}.txt"
            if os.path.exists(cpp_filename):
                # Parse C++ output file
                cpp_data = []
                with open(cpp_filename, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('#') or line.strip() == '':
                            continue
                        # Parse the data line
                        values = [float(x) for x in line.strip().split()]
                        cpp_data.extend(values)
                
                cpp_result = np.array(cpp_data)
                print(f"C++ result shape: {cpp_result.shape}")
                print(f"C++ min/max: {np.min(cpp_result):.6f}/{np.max(cpp_result):.6f}")
                
                # Compare results
                passed = compare_arrays(py_result, cpp_result)
                if not passed:
                    all_passed = False
                    
            else:
                print(f"  ❌ C++ result file not found: {cpp_filename}")
                all_passed = False
                
        except Exception as e:
            print(f"  ❌ Error testing {config['name']}: {e}")
            all_passed = False
    
    print(f"\n{'='*50}")
    if all_passed:
        print("🎉 ALL TESTS PASSED! Python and C++ results match.")
    else:
        print("❌ SOME TESTS FAILED! Check the differences above.")
    print(f"{'='*50}")

if __name__ == "__main__":
    main() 