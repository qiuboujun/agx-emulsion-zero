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
            if x and not x.startswith('===') and not x.startswith('Test') and not x.startswith('...'):
                try:
                    values.append(float(x))
                except ValueError:
                    continue
        return values
    return None

def extract_stats_from_output(output, name):
    """Extract statistics from the output text"""
    pattern = rf"{re.escape(name)} stats: min=([\d.-]+), max=([\d.-]+), mean=([\d.-]+), size=(\d+)"
    match = re.search(pattern, output)
    if match:
        return {
            'min': float(match.group(1)),
            'max': float(match.group(2)),
            'mean': float(match.group(3)),
            'size': int(match.group(4))
        }
    return None

def compare_vectors(cpp_vec, py_vec, test_name, tolerance=1e-5):
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

def compare_stats(cpp_stats, py_stats, test_name, tolerance=1e-5):
    """Compare two statistics dictionaries"""
    if cpp_stats is None or py_stats is None:
        print(f"❌ {test_name}: Could not extract statistics")
        return False
    
    for key in ['min', 'max', 'mean']:
        diff = abs(cpp_stats[key] - py_stats[key])
        if diff > tolerance:
            print(f"❌ {test_name}: {key} differs ({cpp_stats[key]} vs {py_stats[key]}, diff: {diff:.2e})")
            return False
    
    if cpp_stats['size'] != py_stats['size']:
        print(f"❌ {test_name}: sizes differ ({cpp_stats['size']} vs {py_stats['size']})")
        return False
    
    print(f"✓ {test_name}: Statistics match")
    return True

def main():
    print("=== Diffusion C++ vs Python Comparison ===")
    print()
    
    # Run C++ test
    print("Running C++ test...")
    try:
        cpp_result = subprocess.run(['./test_diffusion_standalone'], 
                                   capture_output=True, text=True, cwd='.')
        cpp_output = cpp_result.stdout
        if cpp_result.returncode != 0:
            print(f"❌ C++ test failed with return code {cpp_result.returncode}")
            print("C++ stderr:", cpp_result.stderr)
            return
    except FileNotFoundError:
        print("❌ C++ executable not found. Please compile first:")
        print("   g++ -std=c++17 -I../../include -o test_diffusion_standalone test_diffusion_standalone.cpp ../../src/model/diffusion.cpp")
        return
    
    # Run Python test
    print("Running Python test...")
    try:
        py_result = subprocess.run(['python3', 'test_diffusion_python_standalone.py'], 
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
    
    # Test 1: Gaussian blur
    cpp_blurred = extract_vector_from_output(cpp_output, "Gaussian blurred image (first 20 elements)")
    py_blurred = extract_vector_from_output(py_output, "Gaussian blurred image (first 20 elements)")
    compare_vectors(cpp_blurred, py_blurred, "Test 1: Gaussian blur")
    
    cpp_blurred_stats = extract_stats_from_output(cpp_output, "Gaussian blurred image")
    py_blurred_stats = extract_stats_from_output(py_output, "Gaussian blurred image")
    compare_stats(cpp_blurred_stats, py_blurred_stats, "Test 1: Gaussian blur stats")
    print()
    
    # Test 2: Gaussian blur with micrometres
    cpp_blurred_um = extract_vector_from_output(cpp_output, "Gaussian blurred (um) image (first 20 elements)")
    py_blurred_um = extract_vector_from_output(py_output, "Gaussian blurred (um) image (first 20 elements)")
    compare_vectors(cpp_blurred_um, py_blurred_um, "Test 2: Gaussian blur (um)")
    
    cpp_blurred_um_stats = extract_stats_from_output(cpp_output, "Gaussian blurred (um) image")
    py_blurred_um_stats = extract_stats_from_output(py_output, "Gaussian blurred (um) image")
    compare_stats(cpp_blurred_um_stats, py_blurred_um_stats, "Test 2: Gaussian blur (um) stats")
    print()
    
    # Test 3: Unsharp mask
    cpp_unsharped = extract_vector_from_output(cpp_output, "Unsharp masked image (first 20 elements)")
    py_unsharped = extract_vector_from_output(py_output, "Unsharp masked image (first 20 elements)")
    compare_vectors(cpp_unsharped, py_unsharped, "Test 3: Unsharp mask")
    
    cpp_unsharped_stats = extract_stats_from_output(cpp_output, "Unsharp masked image")
    py_unsharped_stats = extract_stats_from_output(py_output, "Unsharp masked image")
    compare_stats(cpp_unsharped_stats, py_unsharped_stats, "Test 3: Unsharp mask stats")
    print()
    
    # Test 4: Halation
    cpp_halated = extract_vector_from_output(cpp_output, "Halated image (first 20 elements)")
    py_halated = extract_vector_from_output(py_output, "Halated image (first 20 elements)")
    compare_vectors(cpp_halated, py_halated, "Test 4: Halation")
    
    cpp_halated_stats = extract_stats_from_output(cpp_output, "Halated image")
    py_halated_stats = extract_stats_from_output(py_output, "Halated image")
    compare_stats(cpp_halated_stats, py_halated_stats, "Test 4: Halation stats")
    print()
    
    # Test 5: GPU vs CPU comparison
    cpp_gpu = extract_vector_from_output(cpp_output, "GPU blurred image (first 20 elements)")
    py_gpu = extract_vector_from_output(py_output, "GPU blurred image (first 20 elements)")
    compare_vectors(cpp_gpu, py_gpu, "Test 5: GPU/CPU comparison")
    
    cpp_gpu_stats = extract_stats_from_output(cpp_output, "GPU blurred image")
    py_gpu_stats = extract_stats_from_output(py_output, "GPU blurred image")
    compare_stats(cpp_gpu_stats, py_gpu_stats, "Test 5: GPU/CPU comparison stats")
    print()
    
    # Extract max difference from C++ output
    max_diff_match = re.search(r'Max absolute difference \(CPU vs GPU\): ([\d.]+)', cpp_output)
    if max_diff_match:
        max_diff = float(max_diff_match.group(1))
        print(f"C++ internal CPU vs GPU max difference: {max_diff:.2e}")
    
    print("🎉 Comparison completed!")

if __name__ == "__main__":
    main() 