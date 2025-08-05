# FastStats C++/CUDA Module

This module provides high-performance CPU and GPU implementations for computing basic statistics (mean and standard deviation) on large datasets. It's designed to complement the existing `agx_emulsion/utils/fast_stats.py` module by providing simple statistical calculations that aren't available in the Python version.

## Features

- **CPU Implementation**: Templated functions for computing mean and standard deviation
- **GPU Acceleration**: CUDA kernel for parallel reduction on large datasets
- **Automatic Fallback**: CPU fallback when CUDA is not available
- **Memory Efficient**: Single-pass algorithms for combined mean/stddev computation
- **Error Handling**: Comprehensive CUDA error checking and exception handling

## API

### CPU Functions

```cpp
#include "fast_stats.hpp"

// Individual functions
double mean = FastStats::mean(data_vector);
double stddev = FastStats::stddev(data_vector);

// Combined function (more efficient)
auto [mean, stddev] = FastStats::mean_stddev(data_vector);
```

### GPU Functions

```cpp
// GPU computation with automatic fallback
auto [mean, stddev] = FastStats::compute_gpu(data_ptr, data_size);
```

## Files

- `../../include/fast_stats.hpp` - Header with class definition and CPU templates
- `../../src/utils/fast_stats.cpp` - CPU implementation and CUDA wrapper
- `../../src/utils/fast_stats.cu` - CUDA kernel implementation
- `test_fast_stats.cpp` - Comprehensive test suite

## Building and Testing

### With CUDA Support

```bash
# Compile with CUDA
nvcc -std=c++17 -I../../include -o test_fast_stats \
    test_fast_stats.cpp ../../src/utils/fast_stats.cpp ../../src/utils/fast_stats.cu -lcudart

# Run tests
./test_fast_stats
```

### Without CUDA Support

```bash
# Compile without CUDA (uses CPU fallback)
g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -o test_fast_stats \
    test_fast_stats.cpp ../../src/utils/fast_stats.cpp

# Run tests
./test_fast_stats
```

## Test Coverage

The test suite verifies:

1. **CPU Functions**: Individual and combined mean/stddev calculations
2. **GPU Functions**: CUDA kernel execution and results
3. **Edge Cases**: Empty vectors, single elements
4. **Large Datasets**: Performance on 10,000+ element arrays
5. **Fallback Behavior**: CPU fallback when CUDA unavailable
6. **Numerical Accuracy**: Comparison with expected results

## Performance

- **CPU**: Single-pass O(n) algorithms with minimal memory overhead
- **GPU**: Parallel reduction with shared memory optimization
- **Memory**: Efficient use of device memory with proper cleanup
- **Accuracy**: Double precision calculations for high numerical stability

## Integration

The FastStats module is integrated into the AGX Emulsion Zero project:

- **Namespace**: `agx_emulsion::FastStats` for consistency
- **Error Handling**: Uses standard C++ exceptions for CUDA errors
- **Memory Management**: Automatic CUDA memory allocation/deallocation
- **Build System**: Compatible with existing CMake configuration

## Usage Example

```cpp
#include "fast_stats.hpp"

// CPU usage
std::vector<float> data = {1.5f, 2.3f, 3.7f, 4.2f, 5.8f};
auto [mean, stddev] = agx_emulsion::FastStats::mean_stddev(data);

// GPU usage (with fallback)
auto [gpu_mean, gpu_stddev] = agx_emulsion::FastStats::compute_gpu(data.data(), data.size());
```

## Dependencies

- **C++17**: For structured bindings and other modern features
- **CUDA**: Optional, for GPU acceleration
- **Standard Library**: `<vector>`, `<cmath>`, `<utility>`

## Notes

- The GPU implementation uses atomic operations for thread-safe reduction
- Shared memory is used to optimize memory access patterns
- The CPU fallback ensures the code works even without CUDA
- All functions handle empty input gracefully 