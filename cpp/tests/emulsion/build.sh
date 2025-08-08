#!/bin/bash

echo "Building Emulsion Model..."

# Check if CUDA is available
if command -v nvcc &> /dev/null; then
    echo "CUDA detected, building with GPU support..."
    
    # Compile CUDA files
    nvcc -std=c++17 -I../../include -c ../../src/model/emulsion.cu -o emulsion_cuda.o
    
    # Compile C++ files
    g++ -std=c++17 -I../../include -c test_emulsion_fixed_input.cpp -o test_emulsion_fixed_input.o
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -c ../../src/model/emulsion.cpp -o emulsion_cpp.o
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -c ../../src/utils/fast_stats.cpp -o fast_stats_cpp.o
    
    # Link everything together
    nvcc -std=c++17 -o test_emulsion_fixed_input test_emulsion_fixed_input.o emulsion_cpp.o fast_stats_cpp.o emulsion_cuda.o -lcudart -lcurand
    
    if [ $? -eq 0 ]; then
        echo "CUDA build successful!"
        echo "Build complete! Run './test_emulsion_fixed_input' to test."
    else
        echo "CUDA build failed!"
        exit 1
    fi
else
    echo "CUDA not detected, building CPU-only version..."
    
    # Compile C++ files only
    g++ -std=c++17 -I../../include -c test_emulsion_fixed_input.cpp -o test_emulsion_fixed_input.o
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -c ../../src/model/emulsion.cpp -o emulsion_cpp.o
    g++ -std=c++17 -DFAST_STATS_NO_CUDA -I../../include -c ../../src/utils/fast_stats.cpp -o fast_stats_cpp.o
    
    # Link CPU-only version
    g++ -std=c++17 -o test_emulsion_fixed_input test_emulsion_fixed_input.o emulsion_cpp.o fast_stats_cpp.o
    
    if [ $? -eq 0 ]; then
        echo "CPU-only build successful!"
        echo "Build complete! Run './test_emulsion_fixed_input' to test."
    else
        echo "CPU-only build failed!"
        exit 1
    fi
fi

# Make executable
chmod +x test_emulsion_fixed_input 