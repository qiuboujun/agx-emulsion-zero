#ifndef FAST_STATS_HPP
#define FAST_STATS_HPP

#include <vector>
#include <utility>
#include <cmath>
#include <type_traits>

namespace agx_emulsion {

class FastStats {
public:
    // CPU mean computation
    template <typename T>
    static double mean(const std::vector<T>& data) {
        if (data.empty()) return 0.0;
        double sum = 0.0;
        for (T v : data) sum += v;
        return sum / data.size();
    }

    // CPU standard deviation (population)
    template <typename T>
    static double stddev(const std::vector<T>& data) {
        if (data.empty()) return 0.0;
        double mu = mean(data);
        double sq_sum = 0.0;
        for (T v : data) {
            double diff = v - mu;
            sq_sum += diff * diff;
        }
        return std::sqrt(sq_sum / data.size());
    }

    // CPU mean and stddev combined (more efficient)
    template <typename T>
    static std::pair<double, double> mean_stddev(const std::vector<T>& data) {
        if (data.empty()) return {0.0, 0.0};
        
        double sum = 0.0;
        double sq_sum = 0.0;
        size_t n = data.size();
        
        for (T v : data) {
            sum += v;
            sq_sum += static_cast<double>(v) * static_cast<double>(v);
        }
        
        double mean_val = sum / n;
        double variance = (sq_sum / n) - (mean_val * mean_val);
        double stddev_val = std::sqrt(variance);
        
        return {mean_val, stddev_val};
    }

    // GPU computation entrypoint (declaration only)
    static std::pair<double, double> compute_gpu(const float* data, size_t size);
};

} // namespace agx_emulsion

#endif // FAST_STATS_HPP 