cmake_minimum_required(VERSION 3.18)
project(ColorFiltersTest LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add NumCpp
add_subdirectory(cpp/external/NumCpp ${CMAKE_BINARY_DIR}/numcpp_build)

# Create executable
add_executable(test_color_filters_cpp test_color_filters_cpp.cpp)

# Include directories
target_include_directories(test_color_filters_cpp PRIVATE
    cpp/include
    cpp/external/NumCpp/include
)

# Link libraries
target_link_libraries(test_color_filters_cpp PRIVATE
    NumCpp::NumCpp
)

# Add the color_filters source file
target_sources(test_color_filters_cpp PRIVATE
    cpp/src/model/color_filters.cpp
    cpp/src/utils/io.cpp
)

# Copy necessary data files if needed
# (This might be needed depending on how the data loading works) 