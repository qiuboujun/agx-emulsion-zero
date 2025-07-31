#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include "NumCpp.hpp"
#include "pybind11/numpy.h"
#include <nlohmann/json.hpp>
#include "io.hpp"       // The header we are testing
#include "config.hpp"   // For agx::config::SPECTRAL_SHAPE

namespace py = pybind11;

// Helper to convert py::array to nc::NdArray (creates a copy)
template<typename T>
nc::NdArray<T> py_to_nc(py::array_t<T, py::array::c_style | py::array::forcecast> arr) {
    py::buffer_info info = arr.request();
    if (info.ndim == 0) return nc::NdArray<T>();
    
    // Handle different dimensions
    if (info.ndim == 1) {
        // For 1D arrays, create a 1D NumCpp array using Shape constructor
        nc::NdArray<T> nc_arr(nc::Shape(info.shape[0]));
        std::memcpy(nc_arr.data(), info.ptr, sizeof(T) * nc_arr.size());
        return nc_arr;
    } else if (info.ndim == 2) {
        nc::NdArray<T> nc_arr(info.shape[0], info.shape[1]);
        std::memcpy(nc_arr.data(), info.ptr, sizeof(T) * nc_arr.size());
        return nc_arr;
    } else {
        // For higher dimensions, flatten to 2D
        size_t total_size = 1;
        for (size_t i = 0; i < info.ndim - 1; ++i) {
            total_size *= info.shape[i];
        }
        nc::NdArray<T> nc_arr(total_size, info.shape[info.ndim - 1]);
        std::memcpy(nc_arr.data(), info.ptr, sizeof(T) * nc_arr.size());
        return nc_arr;
    }
}

// Helper to convert nc::NdArray back to py::array (creates a copy)
template<typename T>
py::array_t<T> nc_to_py(const nc::NdArray<T>& arr) {
    std::vector<size_t> shape = {arr.shape().rows, arr.shape().cols};
    if (shape[0] == 1 || shape[1] == 1) {
        shape = {arr.size()};
    }
    return py::array_t<T>(shape, arr.data());
}

PYBIND11_MODULE(io_cpp_tests, m) {
    m.doc() = "Pybind11 wrapper for io.cpp";

    m.def("interpolate_to_common_axis_cpp", [](py::array_t<float, py::array::c_style | py::array::forcecast> data, py::array_t<float, py::array::c_style | py::array::forcecast> new_x, bool extrapolate, const std::string& method) {
        return nc_to_py(agx::utils::interpolate_to_common_axis(py_to_nc(data), py_to_nc(new_x), extrapolate, method));
    }, py::arg("data"), py::arg("new_x"), py::arg("extrapolate") = false, py::arg("method") = "akima");

    m.def("load_agx_emulsion_data_cpp", [](const std::string& stock, const std::string& ls_donor, const std::string& dc_donor, const std::string& ddc_donor, const std::string& ddmm_donor, const std::string& type, bool color) {
        // This test requires the config to be initialized first
        agx::config::initialize_config();
        auto result_struct = agx::utils::load_agx_emulsion_data(stock, ls_donor, dc_donor, ddc_donor, ddmm_donor, type, color);
        
        py::dict d;
        d["log_sensitivity"] = nc_to_py(result_struct.log_sensitivity);
        d["dye_density"] = nc_to_py(result_struct.dye_density);
        d["wavelengths"] = nc_to_py(result_struct.wavelengths);
        d["density_curves"] = nc_to_py(result_struct.density_curves);
        d["log_exposure"] = nc_to_py(result_struct.log_exposure);
        return d;

    }, py::arg("stock") = "kodak_portra_400", py::arg("log_sensitivity_donor") = "", py::arg("denisty_curves_donor") = "", py::arg("dye_density_cmy_donor") = "", py::arg("dye_density_min_mid_donor") = "", py::arg("type") = "negative", py::arg("color") = true);

    m.def("load_densitometer_data_cpp", [](const std::string& type){
        agx::config::initialize_config();
        return nc_to_py(agx::utils::load_densitometer_data(type));
    }, py::arg("type") = "status_A");
    
    m.def("load_csv_cpp", [](const std::string& datapkg, const std::string& filename) {
        return nc_to_py(agx::utils::load_csv(datapkg, filename));
    }, py::arg("datapkg"), py::arg("filename"));
    
    m.def("interpolate_to_common_axis_cpp", [](const py::array_t<float, py::array::c_style | py::array::forcecast>& data, 
                                                const py::array_t<float, py::array::c_style | py::array::forcecast>& new_x) {
        return nc_to_py(agx::utils::interpolate_to_common_axis(py_to_nc(data), py_to_nc(new_x)));
    }, py::arg("data"), py::arg("new_x"));
    
    m.def("get_spectral_shape_wavelengths_cpp", []() {
        return nc_to_py(agx::config::SPECTRAL_SHAPE.wavelengths);
    });

    m.def("read_neutral_ymc_filter_values_cpp", []() {
        auto json_data = agx::utils::read_neutral_ymc_filter_values();
        // Return JSON as string, let Python parse it
        return json_data.dump();
    });

    m.def("load_dichroic_filters_cpp", [](const std::string& brand) {
        agx::config::initialize_config();
        return nc_to_py(agx::utils::load_dichroic_filters(agx::config::SPECTRAL_SHAPE.wavelengths, brand));
    }, py::arg("brand") = "thorlabs");

    m.def("load_filter_cpp", [](const std::string& name, const std::string& brand, const std::string& filter_type, bool percent) {
        agx::config::initialize_config();
        return nc_to_py(agx::utils::load_filter(agx::config::SPECTRAL_SHAPE.wavelengths, name, brand, filter_type, percent));
    }, py::arg("name") = "KG3", py::arg("brand") = "schott", py::arg("filter_type") = "heat_absorbing", py::arg("percent_transmittance") = false);
}
