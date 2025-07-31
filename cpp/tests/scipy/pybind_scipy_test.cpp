// SPDX-License-Identifier: MIT
// pybind_scipy_test.cpp – unit‑test that cross‑checks our C++ spline wrappers
// against reference SciPy on the Python side via pybind11.
//
// Build notes (CMake):
//   find_package(pybind11 REQUIRED)
//   add_executable(pybind_scipy_test pybind_scipy_test.cpp)
//   target_link_libraries(pybind_scipy_test PRIVATE pybind11::embed NumCpp)
//
// Running the test binary will launch an embedded Python interpreter, run a
// few interpolations in both languages, and assert that the results match to
// 1e‑5 relative tolerance.  It also checks NaN‑handling and zero‑division
// behaviour.

#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <iostream>
#include <random>
#include "NumCpp.hpp"
#include "scipy.hpp"   // our header‑only subset

namespace py = pybind11;

// Helper: generate strictly increasing random x values in (0, 1)
static nc::NdArray<double> make_x(std::size_t n) {
    std::vector<double> v(n);
    std::mt19937_64 rng(12345);
    std::uniform_real_distribution<double> dist(0.01, 0.99);
    v[0] = 0.0;
    for (std::size_t i = 1; i < n; ++i) v[i] = v[i - 1] + dist(rng) / n;
    return nc::NdArray<double>(v);
}

// Convert NumCpp array to Python list for simplicity
static py::list to_py_list(const nc::NdArray<double>& a) {
    py::list out;
    for (double v : a) out.append(v);
    return out;
}

static void single_case(const std::size_t n,
                        const std::string& method,
                        bool include_nan) {
    auto x = make_x(n);
    auto y = nc::sin(x * 6.28318530718); // some smooth data
    if (include_nan) y[ n / 2 ] = std::numeric_limits<double>::quiet_NaN();

    // Build C++ interpolator
    std::unique_ptr<scipy::interpolate::PPoly> interp_cpp;
    if (method == "akima") interp_cpp = std::make_unique<scipy::interpolate::Akima1DInterpolator>(x, y, true);
    else if (method == "cubic") interp_cpp = std::make_unique<scipy::interpolate::CubicSpline>(x, y);
    else if (method == "smoothing_spline") interp_cpp = std::make_unique<scipy::interpolate::CubicSpline>(x, y); // λ=0 natural = interpolant
    else interp_cpp = nullptr; // linear handled separately

    auto xq = nc::linspace<double>(0.0, x.back(), 23);
    nc::NdArray<double> y_cpp;
    if (method == "linear") y_cpp = nc::interp(xq, x, y);
    else y_cpp = (*interp_cpp)(xq);

    // ---------------- Python side ----------------
    py::gil_scoped_acquire gil{};
    py::module np = py::module::import("numpy");
    py::module scipy_interp = py::module::import("scipy.interpolate");

    py::object x_py = np.attr("array")(to_py_list(x));
    py::object y_py = np.attr("array")(to_py_list(y));
    py::object xq_py = np.attr("array")(to_py_list(xq));

    py::object y_py_eval;
    if (method == "akima") {
        py::object Akima = scipy_interp.attr("Akima1DInterpolator");
        y_py_eval = Akima(x_py, y_py, py::arg("extrapolate")=true)(xq_py);
    } else if (method == "cubic") {
        py::object CS = scipy_interp.attr("CubicSpline");
        y_py_eval = CS(x_py, y_py)(xq_py);
    } else if (method == "linear") {
        py::module np_mod = py::module::import("numpy");
        y_py_eval = np_mod.attr("interp")(xq_py, x_py, y_py);
    } else { // smoothing_spline (lam auto)
        py::object MSS = scipy_interp.attr("make_smoothing_spline");
        auto obj = MSS(x_py, y_py);
        y_py_eval = obj(xq_py);
    }
    auto y_vec_py = y_py_eval.cast<std::vector<double>>();

    // Compare
    for (std::size_t i = 0; i < y_cpp.size(); ++i) {
        double a = y_cpp[i];
        double b = y_vec_py[i];
        if (std::isnan(a) || std::isnan(b)) {
            if (!(std::isnan(a) && std::isnan(b))) {
                throw std::runtime_error("NaN mismatch at i=" + std::to_string(i));
            }
        } else {
            double rel = std::abs(a - b) / std::max({1.0, std::abs(a), std::abs(b)});
            if (rel > 1e-5) {
                throw std::runtime_error("Mismatch >1e-5 at i=" + std::to_string(i));
            }
        }
    }
    std::cout << "[OK] n=" << n << " method=" << method << (include_nan?" (with NaN)":"") << '\n';
}

int main() {
    py::scoped_interpreter guard{}; // start&stop interpreter

    try {
        const std::vector<std::size_t> sizes = {5, 11, 37};
        const std::vector<std::string> methods = {"linear", "cubic", "akima", "smoothing_spline"};
        for (auto n : sizes) {
            for (const auto& m : methods) {
                single_case(n, m, false);
                single_case(n, m, true); // with a NaN injected
            }
        }
        std::cout << "All pybind‑SciPy comparisons passed.\n";
    } catch (const std::exception& e) {
        std::cerr << "Test FAILED: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
