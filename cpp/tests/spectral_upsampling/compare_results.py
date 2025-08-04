#!/usr/bin/env python3
"""
Compare C++ and Python spectral upsampling results to verify they are identical.
"""

import subprocess
import sys
import re
import math

def extract_coords_from_output(output, test_name, coord_type):
    """Extract coordinate values from test output."""
    lines = output.split('\n')
    coords = []
    in_test = False
    
    for line in lines:
        if test_name in line:
            in_test = True
            continue
        if in_test and line.strip() == '':
            break
        if in_test and 'Output ' + coord_type + ' coordinates' in line:
            # Extract coordinates from line like "Output square coordinates: [0.1234567890, 0.9876543210]"
            match = re.search(r'\[([0-9.-]+), ([0-9.-]+)\]', line)
            if match:
                coords.append([float(match.group(1)), float(match.group(2))])
    
    return coords

def extract_spectra_from_output(output, test_name):
    """Extract spectrum values from test output."""
    lines = output.split('\n')
    spectra = []
    in_test = False
    current_spectrum = []
    
    for line in lines:
        if test_name in line:
            in_test = True
            continue
        if in_test and line.strip() == '':
            break
        if in_test and 'Output spectrum' in line:
            # Extract values from line like "Output spectrum: [0.1234567890, 0.9876543210, ...]"
            match = re.search(r'\[([0-9.-]+(?:, [0-9.-]+)*)', line)
            if match:
                values_str = match.group(1)
                # Filter out any non-numeric parts like "..."
                values = []
                for x in values_str.split(','):
                    x = x.strip()
                    if x and x != '...' and not x.startswith('...'):
                        try:
                            values.append(float(x))
                        except ValueError:
                            continue
                current_spectrum.extend(values)
            # Check if this is the end of the spectrum (contains length info)
            if 'Spectrum length:' in line:
                spectra.append(current_spectrum)
                current_spectrum = []
    
    return spectra

def extract_round_trip_errors(output, test_name):
    """Extract round-trip error values from test output."""
    lines = output.split('\n')
    errors = []
    in_test = False
    
    for line in lines:
        if test_name in line:
            in_test = True
            continue
        if in_test and line.strip() == '':
            break
        if in_test and 'Round-trip error:' in line:
            # Extract error from line like "Round-trip error: 0.0000000000"
            match = re.search(r'Round-trip error: ([0-9.-]+)', line)
            if match:
                errors.append(float(match.group(1)))
    
    return errors

def compare_coords(coords1, coords2, tolerance=1e-10):
    """Compare two lists of coordinates with given tolerance."""
    if len(coords1) != len(coords2):
        return False, f"Different number of coordinates: {len(coords1)} vs {len(coords2)}"
    
    for i in range(len(coords1)):
        for j in range(2):
            if abs(coords1[i][j] - coords2[i][j]) > tolerance:
                return False, f"Mismatch at coordinate {i}, component {j}: {coords1[i][j]} vs {coords2[i][j]}"
    
    return True, "Coordinates are identical"

def compare_spectra(spectra1, spectra2, tolerance=1e-10):
    """Compare two lists of spectra with given tolerance."""
    if len(spectra1) != len(spectra2):
        return False, f"Different number of spectra: {len(spectra1)} vs {len(spectra2)}"
    
    for i in range(len(spectra1)):
        if len(spectra1[i]) != len(spectra2[i]):
            return False, f"Different spectrum lengths at index {i}: {len(spectra1[i])} vs {len(spectra2[i])}"
        
        for j in range(len(spectra1[i])):
            if abs(spectra1[i][j] - spectra2[i][j]) > tolerance:
                return False, f"Mismatch at spectrum {i}, element {j}: {spectra1[i][j]} vs {spectra2[i][j]}"
    
    return True, "Spectra are identical"

def compare_errors(errors1, errors2, tolerance=1e-10):
    """Compare two lists of error values with given tolerance."""
    if len(errors1) != len(errors2):
        return False, f"Different number of errors: {len(errors1)} vs {len(errors2)}"
    
    for i in range(len(errors1)):
        if abs(errors1[i] - errors2[i]) > tolerance:
            return False, f"Mismatch at error {i}: {errors1[i]} vs {errors2[i]}"
    
    return True, "Errors are identical"

def main():
    print("Running C++ test...")
    try:
        cpp_result = subprocess.run(['./test_spectral_upsampling_standalone'], 
                                   capture_output=True, text=True, check=True)
        cpp_output = cpp_result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running C++ test: {e}")
        return 1
    
    print("Running Python test...")
    try:
        py_result = subprocess.run([sys.executable, 'test_spectral_upsampling_python_standalone.py'], 
                                  capture_output=True, text=True, check=True)
        py_output = py_result.stdout
    except subprocess.CalledProcessError as e:
        print(f"Error running Python test: {e}")
        return 1
    
    print("\n=== Comparing Results ===")
    
    # Test 1: tri2quad coordinate transformation
    print("\nTest 1: tri2quad coordinate transformation")
    cpp_tri2quad = extract_coords_from_output(cpp_output, "Test 1: tri2quad coordinate transformation", "square")
    py_tri2quad = extract_coords_from_output(py_output, "Test 1: tri2quad coordinate transformation", "square")
    
    match, message = compare_coords(cpp_tri2quad, py_tri2quad)
    if match:
        print("✓ tri2quad results are identical")
    else:
        print(f"✗ tri2quad results differ: {message}")
        return 1
    
    # Test 2: quad2tri coordinate transformation
    print("\nTest 2: quad2tri coordinate transformation")
    cpp_quad2tri = extract_coords_from_output(cpp_output, "Test 2: quad2tri coordinate transformation", "triangular")
    py_quad2tri = extract_coords_from_output(py_output, "Test 2: quad2tri coordinate transformation", "triangular")
    
    match, message = compare_coords(cpp_quad2tri, py_quad2tri)
    if match:
        print("✓ quad2tri results are identical")
    else:
        print(f"✗ quad2tri results differ: {message}")
        return 1
    
    # Test 3: Round-trip transformation
    print("\nTest 3: Round-trip transformation")
    cpp_errors = extract_round_trip_errors(cpp_output, "Test 3: Round-trip transformation")
    py_errors = extract_round_trip_errors(py_output, "Test 3: Round-trip transformation")
    
    match, message = compare_errors(cpp_errors, py_errors)
    if match:
        print("✓ Round-trip errors are identical")
    else:
        print(f"✗ Round-trip errors differ: {message}")
        return 1
    
    # Test 4: computeSpectraFromCoeffs
    print("\nTest 4: computeSpectraFromCoeffs")
    cpp_spectra = extract_spectra_from_output(cpp_output, "Test 4: computeSpectraFromCoeffs")
    py_spectra = extract_spectra_from_output(py_output, "Test 4: computeSpectraFromCoeffs")
    
    match, message = compare_spectra(cpp_spectra, py_spectra)
    if match:
        print("✓ Spectra results are identical")
    else:
        print(f"✗ Spectra results differ: {message}")
        return 1
    
    # Test 5: Edge cases
    print("\nTest 5: Edge cases")
    cpp_edge_tri2quad = extract_coords_from_output(cpp_output, "Test 5: Edge cases", "square")
    py_edge_tri2quad = extract_coords_from_output(py_output, "Test 5: Edge cases", "square")
    
    match, message = compare_coords(cpp_edge_tri2quad, py_edge_tri2quad)
    if match:
        print("✓ Edge case results are identical")
    else:
        print(f"✗ Edge case results differ: {message}")
        return 1
    
    print("\n🎉 All tests passed! C++ and Python implementations produce identical results.")
    return 0

if __name__ == "__main__":
    sys.exit(main()) 