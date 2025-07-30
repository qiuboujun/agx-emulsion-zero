#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "NumCpp.hpp"
#include "pybind11/numpy.h"

#include "fast_interp_lut.hpp" // The header for the functions we are testing

namespace py = pybind11;

// Helper to convert py::array to nc::NdArray (creates a copy and flattens)
template<typename T>
nc::NdArray<T> py_to_nc_flat(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info info = arr.request();
    if (info.ndim == 0) return nc::NdArray<T>();

    // For the C++ functions, we work with flattened arrays [N, C]
    size_t num_pixels = 1;
    for (size_t i = 0; i < info.ndim - 1; ++i) {
        num_pixels *= info.shape[i];
    }
    const size_t num_channels = info.shape.back();
    
    nc::Shape shape(num_pixels, num_channels);
    
    nc::NdArray<T> nc_arr(shape);
    std::memcpy(nc_arr.data(), info.ptr, sizeof(T) * arr.size());
    
    return nc_arr;
}

// Helper to convert py::array to nc::NdArray for LUTs (flattens multi-dimensional LUTs)
template<typename T>
nc::NdArray<T> py_lut_to_nc_flat(py::array_t<T, py::array::c_style | py::array::forcecast> lut) {
    py::buffer_info info = lut.request();
    if (info.ndim == 0) return nc::NdArray<T>();

    // For 2D LUTs: [L, L, C] -> [L*L, C]
    // For 3D LUTs: [L, L, L, C] -> [L*L*L, C]
    size_t total_pixels = 1;
    for (size_t i = 0; i < info.ndim - 1; ++i) {
        total_pixels *= info.shape[i];
    }
    const size_t num_channels = info.shape.back();
    
    nc::Shape shape(total_pixels, num_channels);
    nc::NdArray<T> nc_arr(shape);
    std::memcpy(nc_arr.data(), info.ptr, sizeof(T) * lut.size());
    
    return nc_arr;
}

// Helper to convert nc::NdArray back to py::array (creates a copy)
template<typename T>
py::array_t<T> nc_to_py(const nc::NdArray<T>& arr) {
    // FIX: Manually create the shape vector instead of using .toStlVec()
    std::vector<size_t> shape = {arr.shape().rows, arr.shape().cols};
    return py::array_t<T>(shape, arr.data());
}


PYBIND11_MODULE(fast_interp_cpp_tests, m) {
    m.doc() = "Pybind11 wrapper for fast_interp_lut C++/CUDA implementations";

    m.def("apply_lut_cubic_2d_cpp", [](py::array_t<float, py::array::c_style | py::array::forcecast> lut_py, 
                                      py::array_t<float, py::array::c_style | py::array::forcecast> image_py) {
        
        const int height = image_py.shape(0);
        const int width = image_py.shape(1);

        auto lut_nc = py_lut_to_nc_flat(lut_py);
        auto image_nc = py_to_nc_flat(image_py);

        auto result_flat = agx::apply_lut_cubic_2d(lut_nc, image_nc, height, width);
        
        // Reshape back to 3D for Python
        auto result_py = nc_to_py(result_flat);
        result_py.resize({height, width, (int)result_flat.shape().cols});
        return result_py;

    }, py::arg("lut"), py::arg("image"));

    m.def("apply_lut_cubic_3d_cpp", [](py::array_t<float, py::array::c_style | py::array::forcecast> lut_py, 
                                      py::array_t<float, py::array::c_style | py::array::forcecast> image_py) {
        
        const int height = image_py.shape(0);
        const int width = image_py.shape(1);

        auto lut_nc = py_lut_to_nc_flat(lut_py);
        auto image_nc = py_to_nc_flat(image_py);

        auto result_flat = agx::apply_lut_cubic_3d(lut_nc, image_nc, height, width);
        
        // Reshape back to 3D for Python
        auto result_py = nc_to_py(result_flat);
        result_py.resize({height, width, (int)result_flat.shape().cols});
        return result_py;

    }, py::arg("lut"), py::arg("image"));

    m.def("cubic_interp_lut_at_2d_cpp", [](py::array_t<float, py::array::c_style | py::array::forcecast> lut_py, float x, float y) {
        auto lut_nc = py_lut_to_nc_flat(lut_py);
        return agx::cubic_interp_lut_at_2d(lut_nc, x, y);
    }, py::arg("lut"), py::arg("x"), py::arg("y"));

    m.def("cubic_interp_lut_at_3d_cpp", [](py::array_t<float, py::array::c_style | py::array::forcecast> lut_py, float r, float g, float b) {
        auto lut_nc = py_lut_to_nc_flat(lut_py);
        return agx::cubic_interp_lut_at_3d(lut_nc, r, g, b);
    }, py::arg("lut"), py::arg("r"), py::arg("g"), py::arg("b"));
}
