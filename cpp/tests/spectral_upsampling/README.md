# Spectral Upsampling C++/CUDA Implementation Testing

This directory contains tests for the C++ and CUDA implementation of the spectral upsampling algorithms from the AGX Emulsion Zero project.

## Files

- `test_spectral_upsampling_standalone.cpp` - Standalone C++ test program with fixed input data
- `test_spectral_upsampling_python_standalone.py` - Python equivalent with same fixed input data
- `compare_results.py` - Script to compare C++ and Python outputs
- `test_spectral_upsampling_cuda.cpp` - CUDA test program (for reference)

## Implementation Files

- `../../include/spectral_upsampling.hpp` - C++ header with class definition
- `../../src/utils/spectral_upsampling.cpp` - CPU implementation
- `../../src/utils/spectral_upsampling.cu` - CUDA implementation

## Testing Approach

We use a standalone approach with fixed input data:

1. **Fixed Input Data**: Both C++ and Python tests use identical, fixed input data
2. **Independent Execution**: Each implementation runs independently
3. **Output Comparison**: Results are compared numerically to verify correctness

## Test Results

✅ **All tests pass!**

### Test 1: tri2quad coordinate transformation
- Input: 5 triangular coordinate pairs including corners, center, and edge points
- Result: C++ and Python produce identical square coordinates

### Test 2: quad2tri coordinate transformation
- Input: 5 square coordinate pairs including corners, center, and edge points
- Result: C++ and Python produce identical triangular coordinates

### Test 3: Round-trip transformation (tri2quad -> quad2tri)
- Input: Same triangular coordinates as Test 1
- Result: Perfect round-trip with zero error (0.0000000000)

### Test 4: computeSpectraFromCoeffs
- Input: 4 different coefficient sets including edge cases
- Result: C++ and Python produce identical 441-sample spectra

### Test 5: Edge cases
- Input: Boundary coordinates and zero coefficients
- Result: C++ and Python handle edge cases identically

## Running the Tests

```bash
# Compile the C++ test
g++ -std=c++17 -I../../include -O2 test_spectral_upsampling_standalone.cpp -o test_spectral_upsampling_standalone

# Run individual tests
./test_spectral_upsampling_standalone
python3 test_spectral_upsampling_python_standalone.py

# Run comparison
python3 compare_results.py
```

## Implementation Details

### C++ Implementation
- **Header**: `spectral_upsampling.hpp` defines the `SpectralUpsampling` class with static methods
- **CPU**: `spectral_upsampling.cpp` provides reference implementation with:
  - Coordinate transformations (tri2quad, quad2tri)
  - Spectral synthesis from coefficients
  - No third-party dependencies
- **CUDA**: `spectral_upsampling.cu` provides optional GPU acceleration for:
  - Batch coordinate transformations
  - Parallel processing of large arrays

### Algorithm Fidelity
The C++ implementation faithfully reproduces the Python reference:
- Same coordinate transformation formulas
- Same spectral synthesis algorithm
- Same boundary handling and edge cases
- Same numerical precision

### Key Functions Tested

#### tri2quad(tx, ty)
Converts triangular barycentric coordinates to square coordinates:
- `y = ty / max(1.0 - tx, 1e-10)`
- `x = (1.0 - tx)²`
- Clamps results to [0, 1] range

#### quad2tri(x, y)
Converts square coordinates back to triangular coordinates:
- `tx = 1 - √x`
- `ty = y * √x`

#### computeSpectraFromCoeffs(coeffs)
Generates spectral distribution from 4 coefficients:
- Creates 441 samples from 360-800 nm
- Evaluates polynomial: `x = (c0 * wl + c1) * wl + c2`
- Computes: `spectra = 0.5 * x / √(x² + 1) + 0.5`
- Divides by c3 (with zero protection)

### Performance
- CPU implementation is optimized for clarity and correctness
- CUDA implementation provides optional GPU acceleration for large batches
- Both implementations avoid heavy dependencies

## Verification

The implementation has been verified through:
1. **Numerical Comparison**: C++ and Python produce identical results
2. **Round-trip Testing**: tri2quad -> quad2tri preserves original coordinates
3. **Edge Cases**: Handles boundary conditions and zero coefficients correctly
4. **Compilation**: Both CPU and CUDA versions compile successfully

## CUDA Implementation

The CUDA implementation provides:
- `tri2quad_cuda()` - Batch triangular to square coordinate conversion
- `quad2tri_cuda()` - Batch square to triangular coordinate conversion
- Parallel processing of coordinate arrays
- Automatic memory management and error handling

The CUDA implementation is optional and gracefully falls back to CPU when CUDA is not available.

## Sample Results

### Coordinate Transformations
```
Input triangular coordinates [0]: [0.0, 0.0]
Output square coordinates: [1.0000000000, 0.0000000000]

Input triangular coordinates [1]: [0.5, 0.5]
Output square coordinates: [0.2500000000, 1.0000000000]
```

### Round-trip Accuracy
```
Original triangular coordinates [1]: [0.5, 0.5]
After tri2quad: [0.2500000000, 1.0000000000]
After quad2tri (recovered): [0.5000000000, 0.5000000000]
Round-trip error: 0.0000000000
```

### Spectral Synthesis
```
Input coefficients [0]: [1.0, 0.0, 0.0, 1.0]
Output spectrum: [1.0000000000, 1.0000000000, ...]
Spectrum length: 441 samples
```

The C++/CUDA implementation is ready for production use and provides a drop-in replacement for the Python spectral upsampling module with the added benefit of optional GPU acceleration for large batches. 