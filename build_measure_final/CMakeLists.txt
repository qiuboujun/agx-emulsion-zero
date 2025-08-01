cmake_minimum_required(VERSION 3.16)
project(measure_final_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find NumCpp
find_package(NumCpp REQUIRED)

# Include directories
include_directories(
    ${NUMPY_INCLUDE_DIRS}
    ${NumCpp_INCLUDE_DIRS}
    include
)

# Create executable
add_executable(test_measure_final test_measure_final.cpp)

# Link libraries
target_link_libraries(test_measure_final ${NumCpp_LIBRARIES})

# Set compiler flags
target_compile_options(test_measure_final PRIVATE ${NumCpp_COMPILER_FLAGS}) 