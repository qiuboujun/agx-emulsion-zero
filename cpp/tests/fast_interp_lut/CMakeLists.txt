# Place this file in your top-level `cpp` directory.

cmake_minimum_required(VERSION 3.18)
project(AGX_CPP_Project LANGUAGES CXX CUDA)

# --- Add Dependencies ---
# All paths are now relative to this file's location (`cpp`), which is much cleaner.
add_subdirectory(external/NumCpp ${CMAKE_BINARY_DIR}/numcpp_build)
add_subdirectory(external/pybind11 ${CMAKE_BINARY_DIR}/pybind11_build)
find_package(CUDAToolkit REQUIRED)

# --- Core Library Target ---
# This library contains your actual C++ source code.
# It can be shared by multiple test modules or a final application.
add_library(agx_core STATIC
    src/fast_interp_lut.cu
    # Add other .cu/.cpp source files here as you create them
    # e.g., src/spectral_upsampling.cu
)
target_include_directories(agx_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
)
# The necessary include paths for NumCpp and pybind11 are handled automatically.
# Link the core library to CUDA.
target_link_libraries(agx_core PUBLIC CUDA::cudart)


# --- Test Module for fast_interp_lut ---
# This builds the pybind11 wrapper into a Python module (.so or .pyd).
pybind11_add_module(fast_interp_cpp_tests
    tests/fast_interp_lut/pybind_fast_interp_lut.cpp
)
# Link our core C++ library to the python module so it can access the functions.
target_link_libraries(fast_interp_cpp_tests PRIVATE agx_core)

# **FIX**: Set the output directory for the compiled Python module.
# This places the .so/.pyd file right next to the test script.
set_target_properties(fast_interp_cpp_tests PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/fast_interp_lut"
)


# --- Add other test modules here as you create them ---
# For example, when you are ready to test spectral_upsampling:
#
# pybind11_add_module(spectral_upsampling_cpp_tests
#     tests/spectral_upsampling/pybind_wrapper.cpp
# )
# target_link_libraries(spectral_upsampling_cpp_tests PRIVATE agx_core)


# --- New, Simplified Build Instructions ---
# 1. cd to the `cpp` directory.
# 2. rm -rf build && mkdir build && cd build
# 3. cmake ..
# 4. make fast_interp_cpp_tests  <- This builds *only* the target you need.
# 5. cd ../tests/fast_interp_lut/ <- Go to the test directory
# 6. python3 fast_interp_lut_test.py <- Run the test directly
