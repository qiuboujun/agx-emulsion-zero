#!/bin/bash

# FastStats Build Script
# This script compiles the FastStats module with or without CUDA support

set -e

echo "Building FastStats module..."

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, building with GPU support..."
    
    # Compile CUDA kernel
    nvcc -std=c++17 -c ../../src/utils/fast_stats.cu -o fast_stats_cuda.o
    
    # Compile C++ files (without CUDA headers for the main C++ file)
    g++ -std=c++17 -I../../include -c test_fast_stats.cpp -o test_fast_stats.o
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -c ../../src/utils/fast_stats.cpp -o fast_stats_cpp.o
    
    # Link everything together
    nvcc -std=c++17 -o test_fast_stats test_fast_stats.o fast_stats_cpp.o fast_stats_cuda.o -lcudart
    
    echo "CUDA build successful!"
else
    echo "CUDA not detected, building with CPU fallback..."
    
    # Compile without CUDA
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -o test_fast_stats \
        test_fast_stats.cpp ../../src/utils/fast_stats.cpp
    
    echo "CPU-only build successful!"
fi

# Clean up object files
rm -f *.o

echo "Build complete! Run './test_fast_stats' to test." 