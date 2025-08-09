// SPDX-License-Identifier: MIT

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "balance.hpp"
#include "config.hpp"

namespace py = pybind11;

static AgxEmulsionData numpy_to_agx(py::array_t<float> log_sensitivity,
                                    py::array_t<float> dye_density,
                                    py::array_t<float> wavelengths,
                                    py::array_t<float> density_curves,
                                    py::array_t<float> log_exposure) {
    AgxEmulsionData d;
    d.log_sensitivity = nc::NdArray<float>(log_sensitivity.data(), log_sensitivity.size());
    d.log_sensitivity.reshape({log_sensitivity.shape(0), log_sensitivity.shape(1)});
    d.dye_density = nc::NdArray<float>(dye_density.data(), dye_density.size());
    d.dye_density.reshape({dye_density.shape(0), dye_density.shape(1)});
    d.wavelengths = nc::NdArray<float>(wavelengths.data(), wavelengths.size());
    d.wavelengths.reshape({wavelengths.shape(0), 1});
    d.density_curves = nc::NdArray<float>(density_curves.data(), density_curves.size());
    d.density_curves.reshape({density_curves.shape(0), density_curves.shape(1)});
    d.log_exposure = nc::NdArray<float>(log_exposure.data(), log_exposure.size());
    d.log_exposure.reshape({log_exposure.shape(0), 1});
    return d;
}

static py::array_t<float> nc_to_numpy_2d(const nc::NdArray<float>& a) {
    auto result = py::array_t<float>({a.shape().rows, a.shape().cols});
    auto buf = result.request();
    float* ptr = static_cast<float*>(buf.ptr);
    for (nc::uint32 i = 0; i < a.shape().rows; ++i)
        for (nc::uint32 j = 0; j < a.shape().cols; ++j)
            ptr[i * a.shape().cols + j] = a(i, j);
    return result;
}

PYBIND11_MODULE(balance_cpp_tests, m) {
    m.doc() = "PyBind11 bridge for C++ balance implementation";

    // Ensure global spectral config is initialised once on import
    try {
        agx::config::initialize_config();
    } catch (...) {
        // swallow; test modules may re-init
    }

    m.def("balance_sensitivity", [](py::array_t<float> log_sensitivity,
                                     py::array_t<float> dye_density,
                                     py::array_t<float> wavelengths,
                                     py::array_t<float> density_curves,
                                     py::array_t<float> log_exposure,
                                     const std::string &reference_illuminant,
                                     bool correct_log_exposure) {
        auto data = numpy_to_agx(log_sensitivity, dye_density, wavelengths, density_curves, log_exposure);
        agx::profiles::balance_sensitivity(data, reference_illuminant, correct_log_exposure);
        return py::make_tuple(nc_to_numpy_2d(data.log_sensitivity), nc_to_numpy_2d(data.density_curves));
    });

    m.def("balance_density", [](py::array_t<float> log_sensitivity,
                                 py::array_t<float> dye_density,
                                 py::array_t<float> wavelengths,
                                 py::array_t<float> density_curves,
                                 py::array_t<float> log_exposure) {
        auto data = numpy_to_agx(log_sensitivity, dye_density, wavelengths, density_curves, log_exposure);
        agx::profiles::balance_density(data);
        return py::make_tuple(nc_to_numpy_2d(data.log_sensitivity), nc_to_numpy_2d(data.density_curves));
    });

    m.def("balance_metameric_neutral", [](py::array_t<float> dye_density,
                                           const std::string &viewing_illuminant,
                                           float midgray_value) {
        // convert dye_density to nc::NdArray
        nc::NdArray<float> dd(dye_density.data(), dye_density.size());
        dd.reshape({dye_density.shape(0), dye_density.shape(1)});
        auto res = agx::profiles::balance_metameric_neutral(dd, viewing_illuminant, midgray_value);

        py::array_t<float> dd_out = nc_to_numpy_2d(res.dye_density_out);
        py::array_t<float> d_metameric({3});
        py::array_t<float> d_scale({3});
        auto b1 = d_metameric.request();
        auto b2 = d_scale.request();
        auto *p1 = static_cast<float*>(b1.ptr);
        auto *p2 = static_cast<float*>(b2.ptr);
        for (int i = 0; i < 3; ++i) { p1[i] = res.d_cmy_metameric[i]; p2[i] = res.d_cmy_scale[i]; }

        return py::make_tuple(dd_out, d_metameric, d_scale);
    });

    m.def("balance_metameric_neutral_with_illuminant", [](py::array_t<float> dye_density,
                                                            py::array_t<float> illuminant,
                                                            float midgray_value) {
        nc::NdArray<float> dd(dye_density.data(), dye_density.size());
        dd.reshape({dye_density.shape(0), dye_density.shape(1)});
        nc::NdArray<float> ill(illuminant.data(), illuminant.size());
        ill.reshape({illuminant.shape(0), 1});
        auto res = agx::profiles::balance_metameric_neutral_with_illuminant(dd, ill, midgray_value);
        py::array_t<float> dd_out = nc_to_numpy_2d(res.dye_density_out);
        py::array_t<float> d_metameric({3});
        py::array_t<float> d_scale({3});
        auto b1 = d_metameric.request();
        auto b2 = d_scale.request();
        auto *p1 = static_cast<float*>(b1.ptr);
        auto *p2 = static_cast<float*>(b2.ptr);
        for (int i = 0; i < 3; ++i) { p1[i] = res.d_cmy_metameric[i]; p2[i] = res.d_cmy_scale[i]; }
        return py::make_tuple(dd_out, d_metameric, d_scale);
    });

    m.def("debug_rgb_from_density_params", [](py::array_t<float> dye_density,
                                               py::array_t<float> d_cmy,
                                               const std::string &viewing_illuminant) {
        nc::NdArray<float> dd(dye_density.data(), dye_density.size());
        dd.reshape({dye_density.shape(0), dye_density.shape(1)});
        std::array<float,3> params{d_cmy.at(0), d_cmy.at(1), d_cmy.at(2)};
        auto rgb = agx::profiles::debug_rgb_from_density_params(dd, params, viewing_illuminant);
        return nc_to_numpy_2d(rgb);
    });
}


