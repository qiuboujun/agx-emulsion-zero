# Density Curves C++/CUDA Implementation Tests

This directory contains tests for the C++/CUDA implementation of the density curves module, which mirrors the functionality of the original Python `agx_emulsion/model/density_curves.py`.

## Files

### Implementation Files
- `../../include/density_curves.hpp` - Header file with class definitions and function declarations
- `../../src/model/density_curves.cpp` - CPU implementation of density curve functions
- `../../src/model/density_curves.cu` - CUDA implementation with GPU kernels and CPU fallback

### Test Files
- `test_density_curves_standalone.cpp` - Standalone C++ test executable
- `test_density_curves_python_standalone.py` - Standalone Python test script
- `compare_results.py` - Automated comparison script for C++ vs Python outputs

## Functions Implemented

### Core Density Curve Models
1. **`density_curve_model_norm_cdfs`** - Normal CDF-based density curve model
   - Mirrors Python's `scipy.stats.norm.cdf((loge - center)/sigma) * amplitude`
   - Supports negative, positive, and paper curve types
   - Handles multiple layers (up to 3)

2. **`distribution_model_norm_cdfs`** - Distribution model for normal CDFs
   - Returns Nx3 matrix with per-layer PDFs
   - Mirrors Python's `scipy.stats.norm.pdf((loge-center)/sigma) * amplitude / sigma`

3. **`compute_density_curves`** - Multi-channel density curve computation
   - Computes Nx3 density matrix for RGB/CMY channels
   - Uses the same parameters for all channels in this test

### Interpolation and Correction Functions
4. **`interpolate_exposure_to_density`** - Linear interpolation from exposure to density
   - Channel-wise interpolation with gamma factor correction
   - Clamps to endpoints for out-of-range values

5. **`apply_gamma_shift_correction`** - Gamma and exposure shift correction
   - Applies gamma correction and log exposure shift per channel
   - Uses 1D linear interpolation for curve transformation

### GPU Acceleration
6. **`gpu_density_curve_model_norm_cdfs`** - GPU-accelerated density curve computation
   - CUDA kernel for parallel normal CDF computation
   - Automatic CPU fallback when CUDA is not available
   - Returns true if GPU path was used, false for CPU fallback

## Data Structures

### `DensityParams`
```cpp
struct DensityParams {
    std::array<double,3> center{0.0, 1.0, 2.0};      // Layer centers
    std::array<double,3> amplitude{0.5, 0.5, 0.5};   // Layer amplitudes
    std::array<double,3> sigma{0.3, 0.5, 0.7};       // Layer standard deviations
};
```

### `Matrix`
```cpp
struct Matrix {
    std::size_t rows{0}, cols{0};
    std::vector<double> data;  // Row-major storage
    double& operator()(std::size_t r, std::size_t c);  // Access operator
};
```

### `CurveType`
```cpp
enum class CurveType { Negative, Positive, Paper };
```

## Test Coverage

The tests verify:

1. **Basic Functionality** - All core functions produce expected outputs
2. **Numerical Accuracy** - C++ and Python implementations match to high precision
3. **GPU vs CPU Consistency** - GPU implementation (when available) matches CPU
4. **Edge Cases** - Proper handling of boundary conditions and parameter limits
5. **Multi-channel Processing** - Correct behavior for RGB/CMY channel processing
6. **Interpolation Accuracy** - Linear interpolation matches reference implementation

## Fixed Test Data

The tests use consistent, fixed input data:
- **Log Exposure Range**: [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
- **Default Parameters**: center=[0.0, 1.0, 2.0], amplitude=[0.5, 0.5, 0.5], sigma=[0.3, 0.5, 0.7]
- **Test Matrices**: 2x3 exposure matrices for interpolation tests
- **Correction Factors**: gamma_correction=[1.1, 0.9, 1.0], log_exp_correction=[0.1, -0.1, 0.0]

## Building and Running

### Compile the C++ Test
```bash
cd cpp/tests/density_curves
g++ -std=c++17 -I../../include -o test_density_curves_standalone \
    test_density_curves_standalone.cpp ../../src/model/density_curves.cpp
```

### Run Individual Tests
```bash
# C++ test
./test_density_curves_standalone

# Python test
python3 test_density_curves_python_standalone.py
```

### Run Automated Comparison
```bash
python3 compare_results.py
```

## Expected Results

When the tests pass successfully, you should see:
- All vector and matrix comparisons show "✓ match" with differences < 1e-10
- GPU vs CPU differences (when CUDA is available) < 1e-10
- Consistent numerical output between C++ and Python implementations
- Proper handling of all curve types (negative, positive, paper)

## Integration with Main Project

The density curves implementation is integrated into the main `agx_core` library:
- Header included in `cpp/include/density_curves.hpp`
- Source files added to `cpp/src/model/` directory
- CMakeLists.txt updated to include the new source files
- Namespace `agx_emulsion` used for consistency with other modules

## Performance Notes

- **CPU Implementation**: Optimized with minimal memory allocations and efficient loops
- **GPU Implementation**: Parallel processing of exposure values with CUDA kernels
- **Memory Management**: Automatic cleanup of CUDA memory allocations
- **Fallback Strategy**: Seamless CPU fallback when CUDA is not available

## Future Extensions

The implementation can be extended to support:
- Additional density curve models (e.g., log_line model)
- More sophisticated interpolation methods
- Multi-layer grain simulation
- Real-time curve fitting and optimization 