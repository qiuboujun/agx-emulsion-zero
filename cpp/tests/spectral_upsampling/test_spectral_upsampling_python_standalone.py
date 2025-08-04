"""
Python standalone test for spectral upsampling with fixed input data.
This script uses the same input data as the C++ test for comparison.
"""

import numpy as np
import math

def tri2quad_py(tc):
    """Python implementation of tri2quad from spectral_upsampling.py."""
    tc = np.array(tc)
    tx = tc[0]
    ty = tc[1]
    y = ty / np.fmax(1.0 - tx, 1e-10)
    x = (1.0 - tx) * (1.0 - tx)
    x = np.clip(x, 0, 1)
    y = np.clip(y, 0, 1)
    return np.array([x, y])

def quad2tri_py(xy):
    """Python implementation of quad2tri from spectral_upsampling.py."""
    x = xy[0]
    y = xy[1]
    tx = 1 - np.sqrt(x)
    ty = y * np.sqrt(x)
    return np.array([tx, ty])

def compute_spectra_from_coeffs_py(coeffs, smooth_steps=0):
    """Python implementation of compute_spectra_from_coeffs from spectral_upsampling.py."""
    # Simplified version without smoothing and downsampling to match C++
    wl_up = np.linspace(360, 800, 441)  # upsampled wl for finer initial calculation
    x = (coeffs[0] * wl_up + coeffs[1]) * wl_up + coeffs[2]
    y = 1.0 / np.sqrt(x * x + 1.0)
    spectra = 0.5 * x * y + 0.5
    # Divide by c3, guarding against zero to avoid division by zero
    if coeffs[3] != 0.0:
        spectra /= coeffs[3]
    return spectra

def print_coords(coords, name):
    """Print coordinates in the same format as C++."""
    print(f"{name}: [{coords[0]:.10f}, {coords[1]:.10f}]")

def print_vector(vec, name, max_elements=10):
    """Print a vector in the same format as C++."""
    print(f"{name}: [", end="")
    for i in range(min(len(vec), max_elements)):
        print(f"{vec[i]:.10f}", end="")
        if i < len(vec) - 1 and i < max_elements - 1:
            print(", ", end="")
    if len(vec) > max_elements:
        print(f", ... (showing first {max_elements} of {len(vec)} elements)", end="")
    print("]")

if __name__ == "__main__":
    print("=== Python Spectral Upsampling Test Results ===")
    print()
    
    # Test 1: tri2quad coordinate transformation
    print("Test 1: tri2quad coordinate transformation")
    print("===========================================")
    
    # Test cases for triangular to square coordinates (same as C++)
    tri_coords = [
        [0.0, 0.0],      # Corner of triangle
        [0.5, 0.5],      # Middle of triangle
        [1.0, 0.0],      # Another corner
        [0.25, 0.25],    # Quarter point
        [0.75, 0.25]     # Three-quarter point
    ]
    
    for i, tri in enumerate(tri_coords):
        print(f"Input triangular coordinates [{i}]: [{tri[0]}, {tri[1]}]")
        quad = tri2quad_py(tri)
        print_coords(quad, "Output square coordinates")
        print()
    
    # Test 2: quad2tri coordinate transformation
    print("Test 2: quad2tri coordinate transformation")
    print("===========================================")
    
    # Test cases for square to triangular coordinates (same as C++)
    quad_coords = [
        [0.0, 0.0],      # Corner of square
        [0.5, 0.5],      # Center of square
        [1.0, 1.0],      # Opposite corner
        [0.25, 0.25],    # Quarter point
        [0.75, 0.75]     # Three-quarter point
    ]
    
    for i, quad in enumerate(quad_coords):
        print(f"Input square coordinates [{i}]: [{quad[0]}, {quad[1]}]")
        tri = quad2tri_py(quad)
        print_coords(tri, "Output triangular coordinates")
        print()
    
    # Test 3: Round-trip transformation (tri2quad then quad2tri)
    print("Test 3: Round-trip transformation (tri2quad -> quad2tri)")
    print("========================================================")
    
    for i, original_tri in enumerate(tri_coords):
        print(f"Original triangular coordinates [{i}]: [{original_tri[0]}, {original_tri[1]}]")
        
        quad = tri2quad_py(original_tri)
        print_coords(quad, "After tri2quad")
        
        recovered_tri = quad2tri_py(quad)
        print_coords(recovered_tri, "After quad2tri (recovered)")
        
        error = math.sqrt((original_tri[0] - recovered_tri[0])**2 + 
                         (original_tri[1] - recovered_tri[1])**2)
        print(f"Round-trip error: {error:.10f}")
        print()
    
    # Test 4: computeSpectraFromCoeffs
    print("Test 4: computeSpectraFromCoeffs")
    print("================================")
    
    # Test coefficient sets (same as C++)
    coeff_sets = [
        [1.0, 0.0, 0.0, 1.0],      # Simple case
        [0.5, 0.1, 0.2, 1.0],      # More complex case
        [0.0, 1.0, 0.5, 2.0],      # Another case
        [0.1, 0.2, 0.3, 0.5]      # Small coefficients
    ]
    
    for i, coeffs in enumerate(coeff_sets):
        print(f"Input coefficients [{i}]: [{coeffs[0]}, {coeffs[1]}, {coeffs[2]}, {coeffs[3]}]")
        spectra = compute_spectra_from_coeffs_py(coeffs)
        print_vector(spectra, "Output spectrum", 10)
        print(f"Spectrum length: {len(spectra)} samples")
        print()
    
    # Test 5: Edge cases
    print("Test 5: Edge cases")
    print("==================")
    
    # Test edge cases for tri2quad (same as C++)
    edge_tri_coords = [
        [0.999, 0.001],  # Near boundary
        [0.001, 0.999],  # Near boundary
        [0.5, 0.0],      # On edge
        [0.0, 0.5]       # On edge
    ]
    
    for i, tri in enumerate(edge_tri_coords):
        print(f"Edge case triangular coordinates [{i}]: [{tri[0]}, {tri[1]}]")
        quad = tri2quad_py(tri)
        print_coords(quad, "Output square coordinates")
        print()
    
    # Test edge case for computeSpectraFromCoeffs (zero coefficient)
    zero_coeffs = [0.0, 0.0, 0.0, 0.0]
    print("Edge case coefficients (all zero): [0.0, 0.0, 0.0, 0.0]")
    zero_spectra = compute_spectra_from_coeffs_py(zero_coeffs)
    print_vector(zero_spectra, "Output spectrum (zero coeffs)", 5)
    print() 