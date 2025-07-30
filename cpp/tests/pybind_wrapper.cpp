// cpp/tests/pybind_wrapper.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // Needed for automatic type conversion of std::vector, etc.
#include "NumCpp/NdArray.hpp"
#include "pybind11/numpy.h" // Support for converting between nc::NdArray and numpy.ndarray

#include "../include/spectral_upsampling.hpp" // Include your main C++ header

namespace py = pybind11;
using namespace py::literals; // for _a shorthand

// Helper function to convert py::array to nc::NdArray
// This creates a copy of the data.
template<typename T>
nc::NdArray<T> py_to_nc(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info info = arr.request();
    if (info.ndim == 0) return nc::NdArray<T>();

    // NumCpp Shape constructor takes a std::vector<size_t>
    const nc::Shape shape(std::vector<size_t>(info.shape.begin(), info.shape.end()));
    
    // Create a new nc::NdArray and copy the data
    nc::NdArray<T> nc_arr(shape);
    std::memcpy(nc_arr.data(), info.ptr, sizeof(T) * nc_arr.size());
    
    return nc_arr;
}

// Helper function to convert nc::NdArray back to py::array
// This also creates a copy to avoid lifetime issues.
template<typename T>
py::array_t<T> nc_to_py(const nc::NdArray<T>& arr) {
    // Create a new numpy array and copy the data from the nc::NdArray
    return py::array_t<T>(arr.shape().toStlVec(), arr.data());
}


PYBIND11_MODULE(agx_cpp_tests, m) {
    m.doc() = "Pybind11 wrapper for AGX C++/CUDA implementations";

    //================================================================================
    // LUT Generation Bindings
    //================================================================================

    m.def("load_coeffs_lut_cpp", [](const std::string& filename){
        return nc_to_py(agx::utils::load_coeffs_lut(filename));
    }, "filename"_a = "hanatos_irradiance_xy_coeffs_250304.lut", "C++ version of load_coeffs_lut");

    m.def("tri2quad_cpp", [](py::array_t<float> tc_py) {
        return nc_to_py(agx::utils::tri2quad(py_to_nc(tc_py)));
    }, "tc"_a, "C++ version of tri2quad");

    m.def("quad2tri_cpp", [](py::array_t<float> xy_py) {
        return nc_to_py(agx::utils::quad2tri(py_to_nc(xy_py)));
    }, "xy"_a, "C++ version of quad2tri");

    m.def("fetch_coeffs_cpp", [](py::array_t<float> tc_py, py::array_t<float> lut_coeffs_py) {
        return nc_to_py(agx::utils::fetch_coeffs(py_to_nc(tc_py), py_to_nc(lut_coeffs_py)));
    }, "tc"_a, "lut_coeffs"_a, "C++ version of fetch_coeffs");

    m.def("compute_spectra_from_coeffs_cpp", [](py::array_t<float> coeffs_py, int smooth_steps) {
        return nc_to_py(agx::utils::compute_spectra_from_coeffs(py_to_nc(coeffs_py), smooth_steps));
    }, "coeffs"_a, "smooth_steps"_a = 1, "C++ version of compute_spectra_from_coeffs");
    
    m.def("compute_lut_spectra_cpp", [](int lut_size, int smooth_steps, const std::string& filename) {
        return nc_to_py(agx::utils::compute_lut_spectra(lut_size, smooth_steps, filename));
    }, "lut_size"_a=128, "smooth_steps"_a=1, "lut_coeffs_filename"_a="hanatos_irradiance_xy_coeffs_250304.lut", "C++ version of compute_lut_spectra");

    m.def("load_spectra_lut_cpp", [](const std::string& filename) {
        return nc_to_py(agx::utils::load_spectra_lut(filename));
    }, "filename"_a = "irradiance_xy_tc.npy", "C++ version of load_spectra_lut");

    m.def("illuminant_to_xy_cpp", [](const std::string& label) {
        return nc_to_py(agx::utils::illuminant_to_xy(label));
    }, "illuminant_label"_a, "C++ version of illuminant_to_xy");

    m.def("rgb_to_tc_b_cpp", [](py::array_t<float> rgb_py, const std::string& cs, bool decode, const std::string& illuminant) {
        auto [tc_nc, b_nc] = agx::utils::rgb_to_tc_b(py_to_nc(rgb_py), cs, decode, illuminant);
        return std::make_pair(nc_to_py(tc_nc), nc_to_py(b_nc));
    }, "rgb"_a, "color_space"_a = "ITU-R BT.2020", "apply_cctf_decoding"_a = false, "reference_illuminant"_a = "D55", "C++ version of rgb_to_tc_b");

    //================================================================================
    // Band Pass Filter Bindings
    //================================================================================

    m.def("compute_band_pass_filter_cpp", [](py::array_t<float> uv_py, py::array_t<float> ir_py) {
        return nc_to_py(agx::utils::compute_band_pass_filter(py_to_nc(uv_py), py_to_nc(ir_py)));
    }, "filter_uv"_a = py::cast(std::vector<float>{1.0f, 410.0f, 8.0f}), "filter_ir"_a = py::cast(std::vector<float>{1.0f, 675.0f, 15.0f}), "C++ version of compute_band_pass_filter");

    //================================================================================
    // Spectral Recovery Bindings
    //================================================================================

    m.def("rgb_to_raw_mallett2019_cpp", [](py::array_t<float> rgb_py, py::array_t<float> sens_py, const std::string& cs, bool decode, const std::string& illuminant) {
        auto result = agx::utils::rgb_to_raw_mallett2019(py_to_nc(rgb_py), py_to_nc(sens_py), cs, decode, illuminant);
        return nc_to_py(result);
    }, "RGB"_a, "sensitivity"_a, "color_space"_a = "sRGB", "apply_cctf_decoding"_a = true, "reference_illuminant"_a = "D65", "C++ version of rgb_to_raw_mallett2019");

    m.def("rgb_to_raw_hanatos2025_cpp", [](py::array_t<float> rgb_py, py::array_t<float> sens_py, const std::string& cs, bool decode, const std::string& illuminant) {
        auto result = agx::utils::rgb_to_raw_hanatos2025(py_to_nc(rgb_py), py_to_nc(sens_py), cs, decode, illuminant);
        return nc_to_py(result);
    }, "rgb"_a, "sensitivity"_a, "color_space"_a, "apply_cctf_decoding"_a, "reference_illuminant"_a, "C++ version of rgb_to_raw_hanatos2025");

    m.def("rgb_to_spectrum_cpp", [](py::array_t<float> rgb_py, const std::string& cs, bool decode, const std::string& illuminant) {
        auto result = agx::utils::rgb_to_spectrum(py_to_nc(rgb_py), cs, decode, illuminant);
        return nc_to_py(result);
    }, "rgb"_a, "color_space"_a, "apply_cctf_decoding"_a, "reference_illuminant"_a, "C++ version of rgb_to_spectrum");
}
