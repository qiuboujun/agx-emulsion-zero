#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "couplers.hpp"

namespace py = pybind11;

// Helper function to convert numpy array to std::array<double, 3>
std::array<double, 3> numpy_to_array3(py::array_t<double> arr) {
    if (arr.size() != 3) {
        throw std::invalid_argument("Array must have exactly 3 elements");
    }
    std::array<double, 3> result;
    for (int i = 0; i < 3; ++i) {
        result[i] = arr.at(i);
    }
    return result;
}

// Helper function to convert numpy array to std::array<std::array<double, 3>, 3>
std::array<std::array<double, 3>, 3> numpy_to_array3x3(py::array_t<double> arr) {
    if (arr.ndim() != 2 || arr.shape(0) != 3 || arr.shape(1) != 3) {
        throw std::invalid_argument("Array must be 3x3");
    }
    std::array<std::array<double, 3>, 3> result;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            result[i][j] = arr.at(i, j);
        }
    }
    return result;
}

// Helper function to convert std::array<std::array<double, 3>, 3> to numpy array
py::array_t<double> array3x3_to_numpy(const std::array<std::array<double, 3>, 3>& arr) {
    auto result = py::array_t<double>({3, 3});
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            ptr[i * 3 + j] = arr[i][j];
        }
    }
    return result;
}

// Helper function to convert 2D numpy array to vector<vector<double>>
std::vector<std::vector<double>> numpy_to_vector2d(py::array_t<double> arr) {
    if (arr.ndim() != 2) {
        throw std::invalid_argument("Array must be 2D");
    }
    std::vector<std::vector<double>> result(arr.shape(0), std::vector<double>(arr.shape(1)));
    for (int i = 0; i < arr.shape(0); ++i) {
        for (int j = 0; j < arr.shape(1); ++j) {
            result[i][j] = arr.at(i, j);
        }
    }
    return result;
}

// Helper function to convert vector<vector<double>> to numpy array
py::array_t<double> vector2d_to_numpy(const std::vector<std::vector<double>>& vec) {
    if (vec.empty()) {
        std::vector<size_t> shape = {0, 0};
        return py::array_t<double>(shape);
    }
    std::vector<size_t> shape = {vec.size(), vec[0].size()};
    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = 0; j < vec[i].size(); ++j) {
            ptr[i * vec[i].size() + j] = vec[i][j];
        }
    }
    return result;
}

// Helper function to convert 3D numpy array to vector<vector<array<double, 3>>>
std::vector<std::vector<std::array<double, 3>>> numpy_to_vector3d(py::array_t<double> arr) {
    if (arr.ndim() != 3 || arr.shape(2) != 3) {
        throw std::invalid_argument("Array must be 3D with last dimension of size 3");
    }
    std::vector<std::vector<std::array<double, 3>>> result(arr.shape(0), 
        std::vector<std::array<double, 3>>(arr.shape(1)));
    for (int i = 0; i < arr.shape(0); ++i) {
        for (int j = 0; j < arr.shape(1); ++j) {
            for (int k = 0; k < 3; ++k) {
                result[i][j][k] = arr.at(i, j, k);
            }
        }
    }
    return result;
}

// Helper function to convert vector<vector<array<double, 3>>> to numpy array
py::array_t<double> vector3d_to_numpy(const std::vector<std::vector<std::array<double, 3>>>& vec) {
    if (vec.empty() || vec[0].empty()) {
        std::vector<size_t> shape = {0, 0, 3};
        return py::array_t<double>(shape);
    }
    std::vector<size_t> shape = {vec.size(), vec[0].size(), 3};
    auto result = py::array_t<double>(shape);
    auto buf = result.request();
    double* ptr = static_cast<double*>(buf.ptr);
    for (size_t i = 0; i < vec.size(); ++i) {
        for (size_t j = 0; j < vec[i].size(); ++j) {
            for (int k = 0; k < 3; ++k) {
                ptr[(i * vec[i].size() + j) * 3 + k] = vec[i][j][k];
            }
        }
    }
    return result;
}

PYBIND11_MODULE(couplers_cpp_tests, m) {
    m.doc() = "C++ implementation of DIR couplers for film emulsion simulation";

    m.def("compute_dir_couplers_matrix", [](py::array_t<double> amount_rgb, double layer_diffusion) {
        auto arr = numpy_to_array3(amount_rgb);
        auto result = agx_emulsion::Couplers::compute_dir_couplers_matrix(arr, layer_diffusion);
        return array3x3_to_numpy(result);
    }, "Compute DIR couplers matrix", 
         py::arg("amount_rgb"), py::arg("layer_diffusion"));

    m.def("compute_density_curves_before_dir_couplers", 
          [](py::array_t<double> density_curves, 
             py::array_t<double> log_exposure,
             py::array_t<double> dir_couplers_matrix,
             double high_exposure_couplers_shift = 0.0) {
        auto dc_vec = numpy_to_vector2d(density_curves);
        auto log_exp_vec = std::vector<double>(log_exposure.data(), 
                                              log_exposure.data() + log_exposure.size());
        auto matrix = numpy_to_array3x3(dir_couplers_matrix);
        auto result = agx_emulsion::Couplers::compute_density_curves_before_dir_couplers(
            dc_vec, log_exp_vec, matrix, high_exposure_couplers_shift);
        return vector2d_to_numpy(result);
    }, "Compute density curves before DIR couplers",
         py::arg("density_curves"), py::arg("log_exposure"), 
         py::arg("dir_couplers_matrix"), 
         py::arg("high_exposure_couplers_shift") = 0.0);

    m.def("compute_exposure_correction_dir_couplers",
          [](py::array_t<double> log_raw,
             py::array_t<double> density_cmy,
             py::array_t<double> density_max,
             py::array_t<double> dir_couplers_matrix,
             int diffusion_size_pixel,
             double high_exposure_couplers_shift = 0.0) {
        auto log_raw_vec = numpy_to_vector3d(log_raw);
        auto density_cmy_vec = numpy_to_vector3d(density_cmy);
        auto density_max_arr = numpy_to_array3(density_max);
        auto matrix = numpy_to_array3x3(dir_couplers_matrix);
        auto result = agx_emulsion::Couplers::compute_exposure_correction_dir_couplers(
            log_raw_vec, density_cmy_vec, density_max_arr, matrix,
            diffusion_size_pixel, high_exposure_couplers_shift);
        return vector3d_to_numpy(result);
    }, "Apply exposure correction with DIR couplers",
         py::arg("log_raw"), py::arg("density_cmy"), py::arg("density_max"),
         py::arg("dir_couplers_matrix"), py::arg("diffusion_size_pixel"),
         py::arg("high_exposure_couplers_shift") = 0.0);
} 