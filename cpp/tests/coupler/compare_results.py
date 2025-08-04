#!/usr/bin/env python3
"""
Compare C++ and Python couplers results to verify they are identical.
"""

import subprocess
import sys
import re

def extract_matrix_from_output(output, test_name):
    """Extract matrix values from test output."""
    lines = output.split('\n')
    matrix_lines = []
    in_matrix = False
    
    for line in lines:
        if test_name in line:
            in_matrix = True
            continue
        if in_matrix and line.strip() == '':
            break
        if in_matrix and 'Output matrix:' in line:
            continue
        if in_matrix and line.strip() and not line.startswith('=') and not line.startswith('Input'):
            matrix_lines.append(line.strip())
    
    # Parse the matrix values
    matrix = []
    for line in matrix_lines:
        if line and ',' in line and not line.startswith('Input'):
            try:
                values = [float(x.strip()) for x in line.split(',')]
                matrix.append(values)
            except ValueError:
                continue
    
    return matrix

def extract_2d_array_from_output(output, test_name, array_name):
    """Extract 2D array values from test output."""
    lines = output.split('\n')
    array_lines = []
    in_array = False
    
    for line in lines:
        if test_name in line:
            in_array = False
        if array_name in line:
            in_array = True
            continue
        if in_array and line.strip() == '':
            break
        if in_array and line.strip() and not line.startswith('=') and not line.startswith('Input'):
            array_lines.append(line.strip())
    
    # Parse the array values
    array = []
    for line in array_lines:
        if line and ',' in line and not line.startswith('Input'):
            try:
                values = [float(x.strip()) for x in line.split(',')]
                array.append(values)
            except ValueError:
                continue
    
    return array

def extract_3d_array_from_output(output, test_name, array_name):
    """Extract 3D array values from test output."""
    lines = output.split('\n')
    array_lines = []
    in_array = False
    
    for line in lines:
        if test_name in line:
            in_array = False
        if array_name in line:
            in_array = True
            continue
        if in_array and line.strip() == '':
            break
        if in_array and line.strip() and not line.startswith('=') and not line.startswith('Input'):
            array_lines.append(line.strip())
    
    # Parse the array values
    array = []
    current_row = []
    for line in array_lines:
        if line.startswith('[') and ']: ' in line and not line.startswith('Input'):
            if current_row:
                array.append(current_row)
                current_row = []
            # Extract values after the [i,j]: part
            try:
                values_str = line.split(']: ')[1]
                values = [float(x.strip()) for x in values_str.split(',')]
                current_row.append(values)
            except ValueError:
                continue
    
    if current_row:
        array.append(current_row)
    
    return array

def compare_arrays(arr1, arr2, tolerance=1e-10):
    """Compare two arrays with given tolerance."""
    if len(arr1) != len(arr2):
        return False, f"Different lengths: {len(arr1)} vs {len(arr2)}"
    
    for i in range(len(arr1)):
        if len(arr1[i]) != len(arr2[i]):
            return False, f"Different row lengths at row {i}: {len(arr1[i])} vs {len(arr2[i])}"
        
        for j in range(len(arr1[i])):
            if isinstance(arr1[i][j], list):
                for k in range(len(arr1[i][j])):
                    if abs(arr1[i][j][k] - arr2[i][j][k]) > tolerance:
                        return False, f"Mismatch at [{i}][{j}][{k}]: {arr1[i][j][k]} vs {arr2[i][j][k]}"
            else:
                if abs(arr1[i][j] - arr2[i][j]) > tolerance:
                    return False, f"Mismatch at [{i}][{j}]: {arr1[i][j]} vs {arr2[i][j]}"
    
    return True, "Arrays are identical"

def main():
    print("Running C++ test...")
    try:
        cpp_result = subprocess.run(['./test_couplers_standalone'], 
                                   capture_output=True, text=True, check=True)
        cpp_output = cpp_result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ test: {e}")
        return 1
    
    print("Running Python test...")
    try:
        py_result = subprocess.run([sys.executable, 'test_couplers_python_standalone.py'], 
                                  capture_output=True, text=True, check=True)
        py_output = py_result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running Python test: {e}")
        return 1
    
    print("\n=== Comparing Results ===")
    
    # Test 1: Matrix comparison
    print("\nTest 1: compute_dir_couplers_matrix")
    cpp_matrix = extract_matrix_from_output(cpp_output, "Test 1: compute_dir_couplers_matrix")
    py_matrix = extract_matrix_from_output(py_output, "Test 1: compute_dir_couplers_matrix")
    
    match, message = compare_arrays(cpp_matrix, py_matrix)
    if match:
        print("✓ Matrix results are identical")
    else:
        print(f"✗ Matrix results differ: {message}")
        return 1
    
    # Test 2: 2D array comparison
    print("\nTest 2: compute_density_curves_before_dir_couplers")
    cpp_curves = extract_2d_array_from_output(cpp_output, "Test 2: compute_density_curves_before_dir_couplers", "Output corrected_curves")
    py_curves = extract_2d_array_from_output(py_output, "Test 2: compute_density_curves_before_dir_couplers", "Output corrected_curves")
    
    match, message = compare_arrays(cpp_curves, py_curves)
    if match:
        print("✓ Density curves results are identical")
    else:
        print(f"✗ Density curves results differ: {message}")
        return 1
    
    # Test 3: 3D array comparison
    print("\nTest 3: compute_exposure_correction_dir_couplers")
    cpp_exposure = extract_3d_array_from_output(cpp_output, "Test 3: compute_exposure_correction_dir_couplers", "Output corrected_exposure")
    py_exposure = extract_3d_array_from_output(py_output, "Test 3: compute_exposure_correction_dir_couplers", "Output corrected_exposure")
    
    match, message = compare_arrays(cpp_exposure, py_exposure)
    if match:
        print("✓ Exposure correction results are identical")
    else:
        print(f"✗ Exposure correction results differ: {message}")
        return 1
    
    print("\n🎉 All tests passed! C++ and Python implementations produce identical results.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 