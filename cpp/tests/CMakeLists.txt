cmake_minimum_required(VERSION 3.16)
project(agx_emulsion_io_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find required packages
find_package(OpenMP)
find_package(CUDA)

# Set CUDA flags
if(CUDA_FOUND)
    set(CMAKE_CUDA_STANDARD 17)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
endif()

# Include directories
include_directories(
    ${CMAKE_SOURCE_DIR}/../include
    ${CMAKE_SOURCE_DIR}/../external/NumCpp/include
    ${CMAKE_SOURCE_DIR}/../external/json/single_include
    ${CMAKE_SOURCE_DIR}/../external/pybind11/include
)

# Source files
set(SOURCES
    independent_io_test.cpp
    ${CMAKE_SOURCE_DIR}/../src/utils/io.cpp
)

# Create executable
add_executable(independent_io_test ${SOURCES})

# Link libraries
target_link_libraries(independent_io_test
    ${OpenMP_CXX_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${CUDA_CUDART_LIBRARY}
)

# Set compiler flags
target_compile_options(independent_io_test PRIVATE
    -O3
    -Wall
    -Wextra
)

# Add OpenMP flags if found
if(OpenMP_CXX_FOUND)
    target_compile_options(independent_io_test PRIVATE ${OpenMP_CXX_FLAGS})
endif() 