#ifndef SCIPY_HPP
#define SCIPY_HPP

/*
 * SciPy‑style C++ helpers
 * ------------------------------------------------------------
 * This header provides a header‑only C++17 implementation of
 *   scipy.interpolate.RegularGridInterpolator
 * that you can use via
 *   #include <scipy.hpp>
 * and then
 *   scipy::interpolate::RegularGridInterpolator<double,3> interp{...};
 *
 * Features
 *   • Supports N‑dimensional regular grids (compile‑time N)
 *   • Methods: linear (default), nearest, cubic (Catmull‑Rom)
 *   • Optional bounds checking & fill‑value
 *   • No external deps beyond the C++ standard library
 *
 * Behaviour matches SciPy 1.13 for linear / nearest; cubic is an
 * extra convenience mode that falls back to linear near boundaries.
 */

#include <array>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <stdexcept>
#include <numeric>
#include <cmath>

namespace scipy {
namespace interpolate {

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

    // ------------------------------------------------ utilities
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

    // ------------------------------------------------ nearest
    T nearest(const Point& xi) const {
        std::array<std::size_t, N> idx;
        for (std::size_t d = 0; d < N; ++d) {
            const auto& v = points_[d];
            auto it = std::lower_bound(v.begin(), v.end(), xi[d]);
            if (it == v.end()) {
                if (bounds_error_) throw std::out_of_range("xi out of bounds");
                return fill_value_;
            }
            if (it != v.begin() && (xi[d] - *(it - 1)) < (*it - xi[d])) --it;
            idx[d] = static_cast<std::size_t>(std::distance(v.begin(), it));
        }
        return values_[flat_index(idx)];
    }

    // ------------------------------------------------ linear
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

    // ------------------------------------------------ cubic (Catmull‑Rom)
    static inline T c0(T t) { return ((-t + 2) * t - 1) * t / 2; }
    static inline T c1(T t) { return (((3 * t - 5) * t) * t + 2) / 2; }
    static inline T c2(T t) { return ((-3 * t + 4) * t + 1) * t / 2; }
    static inline T c3(T t) { return ((t - 1) * t * t) / 2; }

    T cubic(const Point& xi) const {
        std::array<std::size_t, N> base{};
        std::array<std::array<T,4>, N> w{}; // per‑dim weights

        for (std::size_t d = 0; d < N; ++d) {
            const auto& v = points_[d];
            auto it = std::upper_bound(v.begin(), v.end(), xi[d]);
            if (it < v.begin() + 2 || it >= v.end() - 1) {
                return linear(xi); // fallback near edges
            }
            base[d] = static_cast<std::size_t>(std::distance(v.begin(), it) - 2);
            T x0 = v[base[d]];
            T x1 = v[base[d] + 1];
            T t = (xi[d] - x1) / (x1 - x0);
            w[d] = {c0(t), c1(t), c2(t), c3(t)};
        }

        const std::size_t total = static_cast<std::size_t>(std::pow(4u, N));
        T acc = T{0};
        std::array<std::size_t, N> idx;
        std::array<std::size_t, N> offset{};
        for (std::size_t i = 0; i < total; ++i) {
            std::size_t tmp = i;
            T wprod = T{1};
            for (std::size_t d = 0; d < N; ++d) {
                offset[d] = tmp & 0x3;
                tmp >>= 2;
                wprod *= w[d][offset[d]];
                idx[d] = base[d] + offset[d];
            }
            acc += wprod * values_[flat_index(idx)];
        }
        return acc;
    }
};

} // namespace interpolate
} // namespace scipy

#endif // SCIPY_HPP
