#!/usr/bin/env python3

import subprocess
import re
import numpy as np

def extract_vector_from_output(output, name):
    """Extract a vector from the output text"""
    pattern = rf"{re.escape(name)}: \[(.*?)\]"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        values_str = match.group(1)
        values = []
        for x in values_str.split(','):
            x = x.strip()
            if x and not x.startswith('===') and not x.startswith('Test'):
                try:
                    values.append(float(x))
                except ValueError:
                    continue
        return values
    return None

def extract_matrix_from_output(output, name):
    """Extract a matrix from the output text"""
    pattern = rf"{re.escape(name)} \(\d+x\d+\):(.*?)(?=\n\n|\nTest|\n==|$)"
    match = re.search(pattern, output, re.DOTALL)
    if match:
        matrix_text = match.group(1)
        rows = []
        for line in matrix_text.strip().split('\n'):
            if line.strip().startswith('Row'):
                # Extract values from "Row X: [val1, val2, val3]"
                values_match = re.search(r'\[(.*?)\]', line)
                if values_match:
                    values_str = values_match.group(1)
                    row_values = []
                    for x in values_str.split(','):
                        x = x.strip()
                        if x and not x.startswith('===') and not x.startswith('Test'):
                            try:
                                row_values.append(float(x))
                            except ValueError:
                                continue
                    if row_values:
                        rows.append(row_values)
        if rows:
            return np.array(rows)
    return None

def extract_parameters_from_output(output):
    """Extract parameters from the output text"""
    center_match = re.search(r'center=\[([\d.-]+), ([\d.-]+), ([\d.-]+)\]', output)
    amplitude_match = re.search(r'amplitude=\[([\d.-]+), ([\d.-]+), ([\d.-]+)\]', output)
    sigma_match = re.search(r'sigma=\[([\d.-]+), ([\d.-]+), ([\d.-]+)\]', output)
    
    if center_match and amplitude_match and sigma_match:
        center = [float(x) for x in center_match.groups()]
        amplitude = [float(x) for x in amplitude_match.groups()]
        sigma = [float(x) for x in sigma_match.groups()]
        return center, amplitude, sigma
    return None, None, None

def compare_vectors(cpp_vec, py_vec, test_name, tolerance=1e-10):
    """Compare two vectors and return True if they match within tolerance"""
    if cpp_vec is None or py_vec is None:
        print(f"❌ {test_name}: Could not extract vectors")
        return False
    
    if len(cpp_vec) != len(py_vec):
        print(f"❌ {test_name}: Vector lengths differ ({len(cpp_vec)} vs {len(py_vec)})")
        return False
    
    cpp_array = np.array(cpp_vec)
    py_array = np.array(py_vec)
    max_diff = np.max(np.abs(cpp_array - py_array))
    
    if max_diff <= tolerance:
        print(f"✓ {test_name}: Vectors match (max diff: {max_diff:.2e})")
        return True
    else:
        print(f"❌ {test_name}: Vectors differ (max diff: {max_diff:.2e})")
        return False

def compare_matrices(cpp_mat, py_mat, test_name, tolerance=1e-10):
    """Compare two matrices and return True if they match within tolerance"""
    if cpp_mat is None or py_mat is None:
        print(f"❌ {test_name}: Could not extract matrices")
        return False
    
    if cpp_mat.shape != py_mat.shape:
        print(f"❌ {test_name}: Matrix shapes differ ({cpp_mat.shape} vs {py_mat.shape})")
        return False
    
    max_diff = np.max(np.abs(cpp_mat - py_mat))
    
    if max_diff <= tolerance:
        print(f"✓ {test_name}: Matrices match (max diff: {max_diff:.2e})")
        return True
    else:
        print(f"❌ {test_name}: Matrices differ (max diff: {max_diff:.2e})")
        return False

def main():
    print("=== Density Curves C++ vs Python Comparison ===")
    print()
    
    # Run C++ test
    print("Running C++ test...")
    try:
        cpp_result = subprocess.run(['./test_density_curves_standalone'], 
                                   capture_output=True, text=True, cwd='.')
        cpp_output = cpp_result.stdout
        if cpp_result.returncode != 0:
            print(f"❌ C++ test failed with return code {cpp_result.returncode}")
            print("C++ stderr:", cpp_result.stderr)
            return
    except FileNotFoundError:
        print("❌ C++ executable not found. Please compile first:")
        print("   g++ -std=c++17 -I../include -o test_density_curves_standalone test_density_curves_standalone.cpp ../src/model/density_curves.cpp")
        return
    
    # Run Python test
    print("Running Python test...")
    try:
        py_result = subprocess.run(['python3', 'test_density_curves_python_standalone.py'], 
                                  capture_output=True, text=True, cwd='.')
        py_output = py_result.stdout
        if py_result.returncode != 0:
            print(f"❌ Python test failed with return code {py_result.returncode}")
            print("Python stderr:", py_result.stderr)
            return
    except FileNotFoundError:
        print("❌ Python executable not found")
        return
    
    print("Comparing results...")
    print()
    
    # Extract and compare parameters
    cpp_center, cpp_amplitude, cpp_sigma = extract_parameters_from_output(cpp_output)
    py_center, py_amplitude, py_sigma = extract_parameters_from_output(py_output)
    
    if cpp_center and py_center:
        compare_vectors(cpp_center, py_center, "Parameters - center")
        compare_vectors(cpp_amplitude, py_amplitude, "Parameters - amplitude")
        compare_vectors(cpp_sigma, py_sigma, "Parameters - sigma")
    print()
    
    # Test 1: density_curve_model_norm_cdfs
    cpp_negative = extract_vector_from_output(cpp_output, "Negative curve output")
    py_negative = extract_vector_from_output(py_output, "Negative curve output")
    compare_vectors(cpp_negative, py_negative, "Test 1: Negative curve")
    
    cpp_positive = extract_vector_from_output(cpp_output, "Positive curve output")
    py_positive = extract_vector_from_output(py_output, "Positive curve output")
    compare_vectors(cpp_positive, py_positive, "Test 1: Positive curve")
    print()
    
    # Test 2: distribution_model_norm_cdfs
    cpp_distribution = extract_matrix_from_output(cpp_output, "Distribution matrix")
    py_distribution = extract_matrix_from_output(py_output, "Distribution matrix")
    compare_matrices(cpp_distribution, py_distribution, "Test 2: Distribution matrix")
    print()
    
    # Test 3: compute_density_curves
    cpp_density_curves = extract_matrix_from_output(cpp_output, "3-channel density curves")
    py_density_curves = extract_matrix_from_output(py_output, "3-channel density curves")
    compare_matrices(cpp_density_curves, py_density_curves, "Test 3: 3-channel density curves")
    print()
    
    # Test 4: interpolate_exposure_to_density
    cpp_interpolated = extract_matrix_from_output(cpp_output, "Interpolated density")
    py_interpolated = extract_matrix_from_output(py_output, "Interpolated density")
    compare_matrices(cpp_interpolated, py_interpolated, "Test 4: Interpolated density")
    print()
    
    # Test 5: apply_gamma_shift_correction
    cpp_corrected = extract_matrix_from_output(cpp_output, "Gamma-shift corrected density curves")
    py_corrected = extract_matrix_from_output(py_output, "Gamma-shift corrected density curves")
    compare_matrices(cpp_corrected, py_corrected, "Test 5: Gamma-shift corrected density curves")
    print()
    
    # Test 6: GPU vs CPU comparison
    cpp_gpu = extract_vector_from_output(cpp_output, "GPU/CPU curve output")
    py_gpu = extract_vector_from_output(py_output, "GPU/CPU curve output")
    compare_vectors(cpp_gpu, py_gpu, "Test 6: GPU/CPU curve")
    print()
    
    # Extract max difference from C++ output
    max_diff_match = re.search(r'Max absolute difference \(CPU vs GPU\): ([\d.]+)', cpp_output)
    if max_diff_match:
        max_diff = float(max_diff_match.group(1))
        print(f"C++ internal CPU vs GPU max difference: {max_diff:.2e}")
    
    print("🎉 Comparison completed!")

if __name__ == "__main__":
    main() 