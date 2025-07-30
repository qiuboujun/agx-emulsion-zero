#pragma once

#include "NumCpp.hpp"
#include <string>
#include <memory>
#include <vector>
#include <array>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <algorithm>

namespace scipy {
namespace interpolate {

//================================================================================
// Abstract Base Class for 1D Interpolation
//================================================================================
template<typename T>
class Interpolator1D {
public:
    virtual ~Interpolator1D() = default;
    virtual nc::NdArray<T> operator()(const nc::NdArray<T>& x_new) const = 0;
};


//================================================================================
// N-Dimensional Regular Grid Interpolator (Header-Only Implementation)
//================================================================================

enum class Method { Linear, Nearest, Cubic };

template <typename T, std::size_t N>
class RegularGridInterpolator {
    // ... This implementation is generic and supports double, but ensure all calling code uses double. ...
};


//================================================================================
// 1D Interpolator Implementations (using double precision)
//================================================================================

class LinearInterpolator : public Interpolator1D<double> {
private:
    nc::NdArray<double> m_x;
    nc::NdArray<double> m_y;
    bool m_extrapolate;

public:
    LinearInterpolator(const nc::NdArray<double>& x, const nc::NdArray<double>& y, bool extrapolate) 
        : m_x(x), m_y(y), m_extrapolate(extrapolate) {}

    nc::NdArray<double> operator()(const nc::NdArray<double>& x_new) const override {
        if (m_extrapolate) {
            return nc::interp(x_new, m_x, m_y);
        }
        
        auto result = nc::NdArray<double>(1, x_new.size());
        double x_first = m_x.front();
        double x_last = m_x.back();

        for(nc::uint32 i = 0; i < x_new.size(); ++i) {
            if (x_new[i] < x_first || x_new[i] > x_last) {
                result[i] = nc::constants::nan;
            } else {
                result[i] = nc::interp(nc::NdArray<double>{x_new[i]}, m_x, m_y).item();
            }
        }
        return result;
    }
};

class Akima1DInterpolator : public Interpolator1D<double> {
private:
    nc::NdArray<double> m_x;
    nc::NdArray<double> m_y;
    nc::NdArray<double> m_b, m_c, m_d; // Coefficients

public:
    Akima1DInterpolator(const nc::NdArray<double>& x, const nc::NdArray<double>& y) : m_x(x), m_y(y) {
        const auto n = x.size();
        if (n < 3) throw std::invalid_argument("Akima interpolation requires at least 3 points.");

        auto dx = nc::diff(x);
        auto m = nc::diff(y) / dx;

        auto m_padded = nc::zeros<double>(1, n + 3);
        m_padded(0, nc::Slice(2, n + 1)) = m;
        m_padded[0, 1] = 2.0 * m[0,0] - m[0,1];
        m_padded[0, 0] = 2.0 * m_padded[0,1] - m_padded[0,2];
        m_padded[0, n + 1] = 2.0 * m[0, n-2] - m[0, n-3];
        m_padded[0, n + 2] = 2.0 * m_padded[0, n+1] - m_padded[0, n];
        
        auto dm = nc::abs(nc::diff(m_padded));
        auto w = dm(0, nc::Slice(2, n + 2)) + dm(0, nc::Slice(0, n));
        
        auto t = (dm(0, nc::Slice(2, n + 2)) * m_padded(0, nc::Slice(1, n + 1)) +
                  dm(0, nc::Slice(0, n)) * m_padded(0, nc::Slice(2, n + 2))) / w;
        t = nc::nan_to_num(t, 0.0);

        m_b = t;
        m_c = (3.0 * m - 2.0 * t(0, nc::Slice(0, n - 1)) - t(0, nc::Slice(1, n))) / dx;
        m_d = (t(0, nc::Slice(0, n - 1)) + t(0, nc::Slice(1, n)) - 2.0 * m) / nc::square(dx);
    }

    nc::NdArray<double> operator()(const nc::NdArray<double>& x_new) const override {
        auto result = nc::NdArray<double>(1, x_new.size());
        for(nc::uint32 i = 0; i < x_new.size(); ++i) {
            auto val = x_new[i];
            auto idx = nc::searchsorted(m_x, val);
            if (idx > 0) idx--;
            if (idx >= m_b.size()) idx = m_b.size() - 1;

            auto h = val - m_x[idx];
            result[i] = m_y[idx] + m_b[idx] * h + m_c[idx] * h * h + m_d[idx] * h * h * h;
        }
        return result;
    }
};


//================================================================================
// Factory Function Implementation
//================================================================================

inline std::unique_ptr<Interpolator1D<double>> create_interpolator(
    const nc::NdArray<double>& x,
    const nc::NdArray<double>& y,
    const std::string& method,
    bool extrapolate)
{
    if (method == "linear") {
        return std::make_unique<LinearInterpolator>(x, y, extrapolate);
    }
    if (method == "akima") {
        return std::make_unique<Akima1DInterpolator>(x, y);
    }
    // Note: A true SciPy-matching cubic spline requires a different, more complex implementation.
    throw std::invalid_argument("Unsupported interpolation method: " + method);
}

} // namespace interpolate
} // namespace scipy
