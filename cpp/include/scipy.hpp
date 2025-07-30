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

/**
 * @brief An abstract base class for all 1D interpolator objects.
 */
class Interpolator1D {
public:
    virtual ~Interpolator1D() = default;
    virtual nc::NdArray<float> operator()(const nc::NdArray<float>& x_new) const = 0;
};


//================================================================================
// N-Dimensional Regular Grid Interpolator (Header-Only Implementation)
//================================================================================

enum class Method { Linear, Nearest, Cubic };

template <typename T, std::size_t N>
class RegularGridInterpolator {
public:
    using Point = std::array<T, N>;

    RegularGridInterpolator(const std::array<std::vector<T>, N>& points,
                            const std::vector<T>& values,
                            Method method = Method::Linear,
                            bool bounds_error = false,
                            T fill_value = T{})
        : points_(points), values_(values), method_(method),
          bounds_error_(bounds_error), fill_value_(fill_value) {
        compute_strides();
        check_shapes();
    }

    T operator()(const Point& xi) const {
        switch (method_) {
            case Method::Linear:  return linear(xi);
            case Method::Nearest: return nearest(xi);
            case Method::Cubic:   return cubic(xi);
            default:              return linear(xi);
        }
    }

    std::vector<T> operator()(const std::vector<Point>& xis) const {
        std::vector<T> out;
        out.reserve(xis.size());
        for (const auto& p : xis) out.push_back((*this)(p));
        return out;
    }

private:
    std::array<std::vector<T>, N> points_;
    std::vector<T> values_;
    Method method_;
    bool bounds_error_;
    T fill_value_;
    std::array<std::size_t, N> strides_{};

    void compute_strides() {
        strides_[N - 1] = 1;
        for (int i = static_cast<int>(N) - 2; i >= 0; --i)
            strides_[i] = strides_[i + 1] * points_[i + 1].size();
    }

    void check_shapes() const {
        std::size_t expected = 1;
        for (const auto& p : points_) expected *= p.size();
        if (expected != values_.size())
            throw std::invalid_argument("values size does not match grid shape");
    }

    std::size_t flat_index(const std::array<std::size_t, N>& idx) const {
        std::size_t off = 0;
        for (std::size_t i = 0; i < N; ++i) off += idx[i] * strides_[i];
        return off;
    }

    T nearest(const Point& xi) const {
        std::array<std::size_t, N> idx;
        for (std::size_t d = 0; d < N; ++d) {
            const auto& v = points_[d];
            auto it = std::lower_bound(v.begin(), v.end(), xi[d]);
            if (it == v.end() || it == v.begin()) {
                 if (bounds_error_) throw std::out_of_range("xi out of bounds");
                 return fill_value_;
            }
            if ((xi[d] - *(it - 1)) < (*it - xi[d])) --it;
            idx[d] = static_cast<std::size_t>(std::distance(v.begin(), it));
        }
        return values_[flat_index(idx)];
    }

    T linear(const Point& xi) const {
        std::array<std::size_t, N> lower{};
        std::array<T, N> t{};
        for (std::size_t d = 0; d < N; ++d) {
            const auto& v = points_[d];
            auto it = std::upper_bound(v.begin(), v.end(), xi[d]);
            if (it == v.begin() || it == v.end()) {
                if (bounds_error_) throw std::out_of_range("xi out of bounds");
                return fill_value_;
            }
            lower[d] = static_cast<std::size_t>(std::distance(v.begin(), it) - 1);
            T x0 = v[lower[d]];
            T x1 = v[lower[d] + 1];
            t[d] = (xi[d] - x0) / (x1 - x0);
        }
        T acc = T{0};
        for (std::size_t mask = 0; mask < (1u << N); ++mask) {
            T w = T{1};
            std::array<std::size_t, N> idx = lower;
            for (std::size_t d = 0; d < N; ++d) {
                if (mask & (1u << d)) {
                    w *= t[d];
                    ++idx[d];
                } else {
                    w *= (T{1} - t[d]);
                }
            }
            acc += w * values_[flat_index(idx)];
        }
        return acc;
    }

    static inline T c0(T t) { return ((-t + 2) * t - 1) * t / 2; }
    static inline T c1(T t) { return (((3 * t - 5) * t) * t + 2) / 2; }
    static inline T c2(T t) { return ((-3 * t + 4) * t + 1) * t / 2; }
    static inline T c3(T t) { return ((t - 1) * t * t) / 2; }

    T cubic(const Point& xi) const {
        std::array<std::size_t, N> base{};
        std::array<std::array<T,4>, N> w{};

        for (std::size_t d = 0; d < N; ++d) {
            const auto& v = points_[d];
            auto it = std::upper_bound(v.begin(), v.end(), xi[d]);
            if (it < v.begin() + 2 || it > v.end() - 1) {
                return linear(xi); // fallback near edges
            }
            base[d] = static_cast<std::size_t>(std::distance(v.begin(), it) - 1);
            T x0 = v[base[d]];
            T x1 = v[base[d] + 1];
            T t = (xi[d] - x0) / (x1 - x0);
            w[d] = {c0(t), c1(t), c2(t), c3(t)};
            base[d]--; // Adjust base for 4-point stencil
        }

        const std::size_t total = static_cast<std::size_t>(std::pow(4u, N));
        T acc = T{0};
        std::array<std::size_t, N> idx;
        for (std::size_t i = 0; i < total; ++i) {
            std::size_t tmp = i;
            T wprod = T{1};
            for (std::size_t d = 0; d < N; ++d) {
                std::size_t offset = tmp & 0x3;
                tmp >>= 2;
                wprod *= w[d][offset];
                idx[d] = base[d] + offset;
            }
            acc += wprod * values_[flat_index(idx)];
        }
        return acc;
    }
};


//================================================================================
// 1D Interpolator Implementations
//================================================================================

// Wrapper for RegularGridInterpolator to fit the Interpolator1D interface
class RegularGridInterpolator1DWrapper : public Interpolator1D {
private:
    RegularGridInterpolator<float, 1> m_interp;
public:
    RegularGridInterpolator1DWrapper(const nc::NdArray<float>& x, const nc::NdArray<float>& y, Method method, bool extrapolate)
        : m_interp({x.toStlVector()}, y.toStlVector(), method, !extrapolate, nc::constants::nan) {}

    nc::NdArray<float> operator()(const nc::NdArray<float>& x_new) const override {
        auto result = nc::NdArray<float>(x_new.shape());
        for (nc::uint32 i = 0; i < x_new.size(); ++i) {
            result[i] = m_interp({x_new[i]});
        }
        return result;
    }
};

// C++ implementation of Akima's 1D interpolator
class Akima1DInterpolator : public Interpolator1D {
private:
    nc::NdArray<float> m_x;
    nc::NdArray<float> m_y;
    nc::NdArray<float> m_b, m_c, m_d; // Coefficients

public:
    Akima1DInterpolator(const nc::NdArray<float>& x, const nc::NdArray<float>& y) : m_x(x), m_y(y) {
        const auto n = x.size();
        if (n < 3) throw std::invalid_argument("Akima interpolation requires at least 3 points.");

        auto dx = nc::diff(x);
        auto m = nc::diff(y) / dx;

        auto m_padded = nc::zeros<float>(1, n + 3);
        m_padded(0, nc::Slice(2, n + 1)) = m;
        m_padded[0, 1] = 2 * m[0,0] - m[0,1];
        m_padded[0, 0] = 2 * m_padded[0,1] - m_padded[0,2];
        m_padded[0, n + 1] = 2 * m[0, n-2] - m[0, n-3];
        m_padded[0, n + 2] = 2 * m_padded[0, n+1] - m_padded[0, n];
        
        auto dm = nc::abs(nc::diff(m_padded));
        auto w = dm(0, nc::Slice(2, n + 2)) + dm(0, nc::Slice(0, n));
        
        auto t = (dm(0, nc::Slice(2, n + 2)) * m_padded(0, nc::Slice(1, n + 1)) +
                  dm(0, nc::Slice(0, n)) * m_padded(0, nc::Slice(2, n + 2))) / w;
        t = nc::nan_to_num(t, 0.0f);

        m_b = t;
        m_c = (3.0 * m - 2.0 * t(0, nc::Slice(0, n - 1)) - t(0, nc::Slice(1, n))) / dx;
        m_d = (t(0, nc::Slice(0, n - 1)) + t(0, nc::Slice(1, n)) - 2.0 * m) / nc::square(dx);
    }

    nc::NdArray<float> operator()(const nc::NdArray<float>& x_new) const override {
        auto result = nc::NdArray<float>(x_new.shape());
        for(nc::uint32 i = 0; i < x_new.size(); ++i) {
            auto val = x_new[i];
            auto idx = nc::searchsorted(m_x, val).item();
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

inline std::unique_ptr<Interpolator1D> create_interpolator(
    const nc::NdArray<float>& x,
    const nc::NdArray<float>& y,
    const std::string& method,
    bool extrapolate)
{
    if (method == "linear") {
        return std::make_unique<RegularGridInterpolator1DWrapper>(x, y, Method::Linear, extrapolate);
    }
    if (method == "cubic") {
        return std::make_unique<RegularGridInterpolator1DWrapper>(x, y, Method::Cubic, extrapolate);
    }
    if (method == "akima") {
        return std::make_unique<Akima1DInterpolator>(x, y);
    }
    throw std::invalid_argument("Unsupported interpolation method: " + method);
}

} // namespace interpolate
} // namespace scipy
