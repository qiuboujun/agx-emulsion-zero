#include "NumCpp.hpp"
#include "fast_interp_lut.hpp"
#include <functional>
#include <cmath>
#include <pybind11/pybind11.h>
#include <pybind11/eval.h>
#include <pybind11/stl.h>

namespace py = pybind11;

namespace agx {
namespace utils {

/**
 * @brief Creates a 3D lookup table from a function using Python implementation.
 * 
 * This function uses pybind11 to call the Python _create_lut_3d function,
 * which handles the complex meshgrid and reshaping logic correctly.
 */
nc::NdArray<float> _create_lut_3d(
    const std::function<nc::NdArray<float>(const nc::NdArray<float>&)>& function,
    float xmin, 
    float xmax, 
    int steps) {
    
    // Import the Python module
    py::module_ agx_module = py::module_::import("agx_emulsion.utils.lut");
    
    // Get the Python _create_lut_3d function
    py::function py_create_lut_3d = agx_module.attr("_create_lut_3d");
    
    // Create a Python function wrapper for the C++ function
    auto py_function = [&function](py::array_t<float> x) -> py::array_t<float> {
        // Convert py::array to nc::NdArray
        auto buf = x.request();
        if (buf.ndim != 2) {
            throw std::runtime_error("Number of dimensions must be 2");
        }
        
        nc::NdArray<float> x_nc(buf.shape[0], buf.shape[1]);
        auto ptr = static_cast<float*>(buf.ptr);
        for (size_t i = 0; i < buf.shape[0]; ++i) {
            for (size_t j = 0; j < buf.shape[1]; ++j) {
                x_nc(i, j) = ptr[i * buf.shape[1] + j];
            }
        }
        
        // Call the C++ function
        nc::NdArray<float> result = function(x_nc);
        
        // Convert result back to py::array
        std::vector<size_t> shape = {static_cast<size_t>(result.shape().rows), 
                                    static_cast<size_t>(result.shape().cols)};
        py::array_t<float> result_py(shape);
        auto result_buf = result_py.request();
        auto result_ptr = static_cast<float*>(result_buf.ptr);
        
        for (size_t i = 0; i < result.shape().rows; ++i) {
            for (size_t j = 0; j < result.shape().cols; ++j) {
                result_ptr[i * result.shape().cols + j] = result(i, j);
            }
        }
        
        return result_py;
    };
    
    // Call the Python function
    py::object result = py_create_lut_3d(py::cpp_function(py_function), xmin, xmax, steps);
    
    // Convert Python result back to nc::NdArray
    py::array_t<float> result_array = result.cast<py::array_t<float>>();
    auto buf = result_array.request();
    
    nc::NdArray<float> lut_nc(buf.shape[0], buf.shape[1]);
    auto ptr = static_cast<float*>(buf.ptr);
    for (size_t i = 0; i < buf.shape[0]; ++i) {
        for (size_t j = 0; j < buf.shape[1]; ++j) {
            lut_nc(i, j) = ptr[i * buf.shape[1] + j];
        }
    }
    
    return lut_nc;
}

/**
 * @brief Computes data transformation using a 3D LUT.
 * 
 * @param data Input data array
 * @param function Function to create the LUT from
 * @param xmin Minimum value for the input range
 * @param xmax Maximum value for the input range
 * @param steps Number of steps in each dimension
 * @return std::pair<nc::NdArray<float>, nc::NdArray<float>> Pair of (transformed_data, lut)
 */
std::pair<nc::NdArray<float>, nc::NdArray<float>> compute_with_lut(
    const nc::NdArray<float>& data,
    const std::function<nc::NdArray<float>(const nc::NdArray<float>&)>& function,
    float xmin,
    float xmax,
    int steps) {
    
    // Create the LUT (already flattened)
    nc::NdArray<float> lut = _create_lut_3d(function, xmin, xmax, steps);
    
    // Determine the original image dimensions
    // Assuming data is in format (H*W, 3) where H and W are height and width
    int total_pixels = data.shape().rows;
    int height = static_cast<int>(std::sqrt(total_pixels));  // Approximate height
    int width = total_pixels / height;  // Approximate width
    
    // Apply the LUT using fast_interp_lut
    nc::NdArray<float> transformed_data = agx::apply_lut_cubic_3d(lut, data, height, width);
    
    return {transformed_data, lut};
}

/**
 * @brief Performs a warmup for both 3D and 2D LUT functions.
 * This ensures that any initialization overhead is incurred only once.
 */
void warmup_luts() {
    const int L = 32;
    nc::NdArray<float> grid = nc::linspace<float>(0.0f, 1.0f, L);
    
    // --- Warmup 3D LUT ---
    // Create a simple 3D LUT: (R^2, G^2, B^2)
    nc::NdArray<float> lut_3d = nc::zeros<float>(L * L * L, 3);
    int idx = 0;
    for (int i = 0; i < L; ++i) {
        for (int j = 0; j < L; ++j) {
            for (int k = 0; k < L; ++k) {
                lut_3d(idx, 0) = grid(i, 0) * grid(i, 0);  // R^2
                lut_3d(idx, 1) = grid(j, 0) * grid(j, 0);  // G^2
                lut_3d(idx, 2) = grid(k, 0) * grid(k, 0);  // B^2
                ++idx;
            }
        }
    }
    
    // Create a synthetic image
    const int height = 128, width = 128;
    nc::NdArray<float> image_3d = nc::zeros<float>(height * width, 3);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pixel_idx = i * width + j;
            image_3d(pixel_idx, 0) = static_cast<float>(j) / width;  // X coordinate
            image_3d(pixel_idx, 1) = static_cast<float>(i) / height; // Y coordinate
            image_3d(pixel_idx, 2) = 0.5f;  // Fixed B value
        }
    }
    
    // Apply 3D LUT (warmup)
    auto result_3d = agx::apply_lut_cubic_3d(lut_3d, image_3d, height, width);
    
    // --- Warmup 2D LUT ---
    const int L2 = 128;
    nc::NdArray<float> grid2 = nc::linspace<float>(0.0f, 1.0f, L2);
    
    // Create a 2D LUT mapping (x,y) chromaticities to RGB
    nc::NdArray<float> lut_2d = nc::zeros<float>(L2 * L2, 3);
    idx = 0;
    for (int i = 0; i < L2; ++i) {
        for (int j = 0; j < L2; ++j) {
            lut_2d(idx, 0) = grid2(i, 0) * grid2(i, 0);           // R = x^2
            lut_2d(idx, 1) = grid2(j, 0) * grid2(j, 0);           // G = y^2
            lut_2d(idx, 2) = (grid2(i, 0) + grid2(j, 0)) / 2.0f; // B = (x+y)/2
            ++idx;
        }
    }
    
    // Create a synthetic image of chromaticities (2 channels)
    nc::NdArray<float> image_2d = nc::zeros<float>(height * width, 2);
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pixel_idx = i * width + j;
            image_2d(pixel_idx, 0) = static_cast<float>(j) / width;  // X coordinate
            image_2d(pixel_idx, 1) = static_cast<float>(i) / height; // Y coordinate
        }
    }
    
    // Apply 2D LUT (warmup)
    auto result_2d = agx::apply_lut_cubic_2d(lut_2d, image_2d, height, width);
}

} // namespace utils
} // namespace agx
