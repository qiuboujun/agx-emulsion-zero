#include <pybind11/pybind11.h>
#include "NumCpp.hpp"
#include "pybind11/numpy.h"

#include "config.hpp" // The header we are testing

namespace py = pybind11;

// Helper to convert nc::NdArray back to py::array (creates a copy)
template<typename T>
py::array_t<T> nc_to_py(const nc::NdArray<T>& arr) {
    std::vector<size_t> shape = {arr.shape().rows, arr.shape().cols};
    // Handle 1D arrays correctly
    if (shape[0] == 1 || shape[1] == 1) {
        shape = {arr.size()};
    }
    return py::array_t<T>(shape, arr.data());
}

PYBIND11_MODULE(config_cpp_tests, m) {
    m.doc() = "Pybind11 wrapper for config.hpp";

    // 1. Bind the initialization function
    m.def("initialize_config_cpp", &agx::config::initialize_config, "Initializes all global config variables in C++");

    // 2. Bind functions to get the initialized values
    m.def("get_log_exposure_cpp", []() {
        return nc_to_py(agx::config::LOG_EXPOSURE);
    });

    m.def("get_spectral_shape_wavelengths_cpp", []() {
        return nc_to_py(agx::config::SPECTRAL_SHAPE.wavelengths);
    });

    m.def("get_standard_observer_cmfs_cpp", []() {
        return nc_to_py(agx::config::STANDARD_OBSERVER_CMFS);
    });
}
