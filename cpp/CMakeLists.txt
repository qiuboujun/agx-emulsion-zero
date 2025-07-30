# Place this file in your top-level `cpp` directory.

cmake_minimum_required(VERSION 3.18)
project(AGX_CPP_Project LANGUAGES CXX CUDA)

# --- Add Dependencies ---
add_subdirectory(external/NumCpp ${CMAKE_BINARY_DIR}/numcpp_build)
add_subdirectory(external/pybind11 ${CMAKE_BINARY_DIR}/pybind11_build)
add_subdirectory(external/json ${CMAKE_BINARY_DIR}/json_build)
find_package(CUDAToolkit REQUIRED)

# --- Core Library Target ---
add_library(agx_core STATIC
    src/utils/fast_interp_lut.cu
    src/utils/io.cpp
    # src/utils/spectral_upsampling.cu  # Temporarily excluded due to compilation issues
    # scipy.cpp is no longer needed as scipy.hpp is header-only
    # colour.cpp doesn't exist - colour.hpp is header-only
)
set_target_properties(agx_core PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(agx_core PUBLIC
    "${CMAKE_CURRENT_SOURCE_DIR}/include"
    "${CMAKE_CURRENT_SOURCE_DIR}/external/json/include"
)
target_link_libraries(agx_core PUBLIC CUDA::cudart)


# --- Test Module for fast_interp_lut ---
pybind11_add_module(fast_interp_cpp_tests
    tests/fast_interp_lut/pybind_fast_interp_lut.cpp
)
target_link_libraries(fast_interp_cpp_tests PRIVATE agx_core)
set_target_properties(fast_interp_cpp_tests PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/fast_interp_lut"
)

# --- Test Module for config ---
pybind11_add_module(config_cpp_tests
    tests/config/pybind_config.cpp
)
target_link_libraries(config_cpp_tests PRIVATE agx_core)
set_target_properties(config_cpp_tests PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/config"
)

# --- Test Module for io ---
pybind11_add_module(io_cpp_tests
    tests/io/pyblind_io.cpp
)
target_link_libraries(io_cpp_tests PRIVATE agx_core)
set_target_properties(io_cpp_tests PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/io"
)


# --- Build Instructions ---
# 1. cd cpp/build
# 2. cmake ..
# 3. make <target_name>  (e.g., make fast_interp_cpp_tests)
# 4. cd ../tests/<test_name>/
# 5. python3 <test_script>.py
