# Diffusion C++/CUDA Implementation Tests

This directory contains tests for the C++/CUDA implementation of the diffusion module, which mirrors the functionality of the original Python `agx_emulsion/model/diffusion.py`.

## Files

### Implementation Files
- `../../include/diffusion.hpp` - Header file with class definitions and function declarations
- `../../src/model/diffusion.cpp` - CPU implementation of diffusion functions
- `../../src/model/diffusion.cu` - CUDA implementation with GPU kernels and CPU fallback

### Test Files
- `test_diffusion_standalone.cpp` - Standalone C++ test executable
- `test_diffusion_python_standalone.py` - Standalone Python test script
- `compare_results.py` - Automated comparison script for C++ vs Python outputs

## Functions Implemented

### Core Diffusion Functions
1. **`apply_gaussian_blur`** - Gaussian blur with sigma in pixels
   - Supports configurable truncation factor (default 4.0)
   - Uses separable convolution with reflect boundary conditions
   - Optional CUDA acceleration with CPU fallback

2. **`apply_gaussian_blur_um`** - Gaussian blur with sigma in micrometres
   - Converts micrometres to pixels using pixel_size_um
   - Calls the pixel-based blur function

3. **`apply_unsharp_mask`** - Unsharp masking effect
   - Formula: `image + amount * (image - gaussian_blur(image, sigma))`
   - Uses standard Gaussian blur with truncate=4.0

4. **`apply_halation_um`** - Halation and scattering effects
   - In-place modification of image data
   - Per-channel halation with configurable size and strength
   - Per-channel scattering with configurable size and strength
   - Uses truncate=7.0 for both halation and scattering (matching Python)

### GPU Acceleration
5. **`diffusion_cuda::gaussian_blur_rgb`** - GPU-accelerated Gaussian blur
   - CUDA kernels for separable convolution
   - Automatic CPU fallback when CUDA is not available
   - Returns true if GPU path executed, false for CPU fallback

## Data Structures

### `HalationParams`
```cpp
struct HalationParams {
    bool active = false;
    std::array<float,3> size_um{{0.f,0.f,0.f}};           // Halation size in micrometres
    std::array<float,3> strength{{0.f,0.f,0.f}};          // Halation strength
    std::array<float,3> scattering_size_um{{0.f,0.f,0.f}}; // Scattering size in micrometres
    std::array<float,3> scattering_strength{{0.f,0.f,0.f}}; // Scattering strength
};
```

### Image Format
- **Interleaved RGB**: `std::vector<float>` with layout `[r,g,b, r,g,b, ...]`
- **Row-major**: `(height * width * 3)` total elements
- **Channel order**: Red, Green, Blue

## Test Coverage

The tests verify:

1. **Basic Functionality** - All core functions produce expected outputs
2. **Numerical Accuracy** - C++ and Python implementations match to high precision
3. **GPU vs CPU Consistency** - GPU implementation (when available) matches CPU
4. **Edge Cases** - Proper handling of boundary conditions and parameter limits
5. **Image Processing** - Correct behavior for different image sizes and parameters
6. **Halation Effects** - Proper application of halation and scattering effects

## Fixed Test Data

The tests use consistent, fixed input data:
- **Image Size**: 64x80x3 = 15,360 elements
- **Random Seed**: 42 (for reproducible results)
- **Test Parameters**:
  - Gaussian blur: sigma = 2.0 pixels
  - Gaussian blur (um): sigma_um = 3.25, pixel_um = 2.5 (→ sigma_px = 1.3)
  - Unsharp mask: sigma = 1.5, amount = 0.6
  - Halation: size_um = [5.0, 3.0, 2.0], strength = [0.10, 0.05, 0.00]
  - Scattering: size_um = [2.0, 0.0, 1.5], strength = [0.02, 0.00, 0.01]

## Building and Running

### Compile the C++ Test
```bash
cd cpp/tests/diffusion
g++ -std=c++17 -I../../include -o test_diffusion_standalone \
    test_diffusion_standalone.cpp ../../src/model/diffusion.cpp
```

### Run Individual Tests
```bash
# C++ test
./test_diffusion_standalone

# Python test
python3 test_diffusion_python_standalone.py
```

### Run Automated Comparison
```bash
python3 compare_results.py
```

## Expected Results

When the tests pass successfully, you should see:
- All vector and statistics comparisons show "✓ match" with differences < 1e-5
- GPU vs CPU differences (when CUDA is available) < 1e-10
- Consistent numerical output between C++ and Python implementations
- Proper handling of all diffusion effects (blur, unsharp mask, halation)

## Integration with Main Project

The diffusion implementation is integrated into the main `agx_core` library:
- Header included in `cpp/include/diffusion.hpp`
- Source files added to `cpp/src/model/` directory
- CMakeLists.txt updated to include the new source files
- Namespace `agx_emulsion` used for consistency with other modules

## Performance Notes

- **CPU Implementation**: Optimized separable convolution with minimal memory allocations
- **GPU Implementation**: Parallel processing with CUDA kernels for horizontal and vertical passes
- **Memory Management**: Automatic cleanup of CUDA memory allocations
- **Fallback Strategy**: Seamless CPU fallback when CUDA is not available
- **Boundary Handling**: Reflect boundary conditions matching SciPy's default behavior

## Algorithm Details

### Gaussian Kernel Generation
- Kernel size: `2 * ceil(truncate * sigma) + 1`
- Normalized weights: `exp(-(i²)/(2σ²))` with sum normalization
- Truncation: Default 4.0 for standard blur, 7.0 for halation/scattering

### Separable Convolution
- Two-pass approach: horizontal then vertical
- Reflect boundary conditions: `... 3 2 1 0 0 1 2 3 ...`
- Per-channel processing for RGB images

### Halation/Scattering
- In-place modification: `raw[:,:,c] = (raw[:,:,c] + strength * blur) / (1 + strength)`
- Per-channel processing with individual parameters
- Two-stage application: halation first, then scattering

## Future Extensions

The implementation can be extended to support:
- Additional blur kernels (e.g., box, triangle)
- More sophisticated boundary conditions
- Multi-scale processing
- Real-time video processing
- Additional image effects and filters 