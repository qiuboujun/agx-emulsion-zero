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
    src/utils/measure.cpp
    src/utils/autoexposure.cpp
    src/utils/autoexposure.cu
    src/utils/crop_resize.cpp
    src/utils/crop_resize.cu
    src/utils/density_spectral.cpp
    src/utils/spectral_upsampling.cpp
    src/utils/spectral_upsampling.cu
    src/utils/profile_io.cpp
    src/model/reconstruct.cpp
    src/model/parametric.cpp
    src/model/parametric.cu
    src/model/color_filters.cpp
    src/model/illuminants.cpp
    src/model/balance.cpp
    src/model/couplers.cpp
    src/model/couplers.cu
    src/model/density_curves.cpp
    src/model/density_curves.cu
    src/model/diffusion.cpp
    src/model/diffusion.cu
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
# Ensure headers that use AGX_SOURCE_DIR get the repo root path
target_compile_definitions(agx_core PUBLIC AGX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# --- Standalone test for parametric model ---
add_executable(independent_parametric_test tests/parametric/independent_parametric_test.cpp)
target_link_libraries(independent_parametric_test PRIVATE agx_core)
set_target_properties(independent_parametric_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)

# --- Standalone test for autoexposure ---
add_executable(independent_autoexposure_test tests/autoexposure/independent_autoexposure_test.cpp)
target_link_libraries(independent_autoexposure_test PRIVATE agx_core)
set_target_properties(independent_autoexposure_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)

# --- Standalone test for crop/resize ---
add_executable(independent_crop_resize_test tests/crop_resize/independent_crop_resize_test.cpp)
target_link_libraries(independent_crop_resize_test PRIVATE agx_core)
set_target_properties(independent_crop_resize_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)
# --- Standalone test for profile I/O ---
add_executable(independent_profile_io_test tests/io/test_profile_io.cpp)
target_link_libraries(independent_profile_io_test PRIVATE agx_core)
set_target_properties(independent_profile_io_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)


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

# --- Test Module for balance ---
pybind11_add_module(balance_cpp_tests
    tests/balance/pybind_balance.cpp
)
target_link_libraries(balance_cpp_tests PRIVATE agx_core)
set_target_properties(balance_cpp_tests PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/balance"
)

# --- Debug executable for balance ---
add_executable(debug_balance
    tests/balance/debug_balance.cpp
)
target_link_libraries(debug_balance PRIVATE agx_core)
set_target_properties(debug_balance PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)

# --- Test Module for io ---
pybind11_add_module(io_cpp_tests
    tests/io/pyblind_io.cpp
)
target_link_libraries(io_cpp_tests PRIVATE agx_core)
set_target_properties(io_cpp_tests PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/io"
)

# --- Standalone test for reconstruct ---
add_executable(independent_reconstruct_test tests/reconstruct/independent_reconstruct_test.cpp)
target_link_libraries(independent_reconstruct_test PRIVATE agx_core)
set_target_properties(independent_reconstruct_test PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)

# --- Test for rgb_to_raw_hanatos2025 ---
add_executable(test_rgb_to_raw_hanatos tests/spectral_upsampling/test_rgb_to_raw_hanatos.cpp)
target_link_libraries(test_rgb_to_raw_hanatos PRIVATE agx_core)
set_target_properties(test_rgb_to_raw_hanatos PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/build"
)
target_compile_definitions(test_rgb_to_raw_hanatos PRIVATE AGX_SOURCE_DIR="${CMAKE_SOURCE_DIR}")

# --- Convenience target to run C++ test and Python parity comparison ---
add_custom_target(compare_rgb_to_raw
    COMMAND ${CMAKE_COMMAND} -E echo "Running C++ rgb_to_raw_hanatos test..."
    COMMAND ${CMAKE_SOURCE_DIR}/build/test_rgb_to_raw_hanatos
    COMMAND ${CMAKE_COMMAND} -E echo "Running Python comparison..."
    COMMAND ${CMAKE_SOURCE_DIR}/tests/spectral_upsampling/run_comparison.sh
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    DEPENDS test_rgb_to_raw_hanatos
)

# --- Test Module for couplers ---
pybind11_add_module(couplers_cpp_tests
    tests/coupler/pybind_couplers.cpp
)
target_link_libraries(couplers_cpp_tests PRIVATE agx_core)
set_target_properties(couplers_cpp_tests PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/tests/coupler"
)

# Add scipy test target
add_custom_target(scipy_test
    COMMAND ${CMAKE_COMMAND} -E echo "Running scipy interpolation test..."
    COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tests/io/pybind_scipy_test.py
    DEPENDS io_cpp_tests
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tests/io
    COMMENT "Running scipy interpolation test"
)

# --- Test Executable for illuminants comparison ---
# Commented out as the source file doesn't exist
# add_executable(test_illuminants_cpp
#     ${CMAKE_CURRENT_SOURCE_DIR}/../test_illuminants_cpp.cpp
# )
# target_link_libraries(test_illuminants_cpp PRIVATE agx_core)
# set_target_properties(test_illuminants_cpp PROPERTIES
#     RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/../"
# )




# --- Build Instructions ---
# 1. cd cpp/build
# 2. cmake ..
# 3. make <target_name>  (e.g., make fast_interp_cpp_tests)
# 4. cd ../tests/<test_name>/
# 5. python3 <test_script>.py
