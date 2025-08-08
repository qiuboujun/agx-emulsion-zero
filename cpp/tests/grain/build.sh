#!/bin/bash

# Grain Model Build Script
# This script compiles the grain model with or without CUDA support

set -e

echo "Building Grain Model..."

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, building with GPU support..."
    
    # Compile CUDA kernel
    nvcc -std=c++17 -I../../include -c ../../src/model/grain.cu -o grain_cuda.o
    
    # Compile C++ files
    g++ -std=c++17 -I../../include -c test_grain_fixed_input.cpp -o test_grain_fixed_input.o
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -c ../../src/model/grain.cpp -o grain_cpp.o
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -c ../../src/utils/fast_stats.cpp -o fast_stats_cpp.o
    
    # Link everything together
    nvcc -std=c++17 -o test_grain_fixed_input test_grain_fixed_input.o grain_cpp.o fast_stats_cpp.o grain_cuda.o -lcudart -lcurand
    
    echo "CUDA build successful!"
else
    echo "CUDA not detected, building with CPU fallback..."
    
    # Compile without CUDA
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -o test_grain_fixed_input \
        test_grain_fixed_input.cpp ../../src/model/grain.cpp ../../src/utils/fast_stats.cpp
    
    echo "CPU-only build successful!"
fi

# Clean up object files
rm -f *.o

echo "Build complete! Run './test_grain_fixed_input' to test." 