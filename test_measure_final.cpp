#include "NumCpp.hpp"
#include "scipy.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>

namespace agx {
namespace utils {

nc::NdArray<float> measure_gamma(const nc::NdArray<float>& log_exposure,
                                const nc::NdArray<float>& density_curves,
                                float density_0,
                                float density_1) {
    nc::NdArray<float> gamma(1, 3);

    for (int i = 0; i < 3; ++i) {
        nc::NdArray<float> density_channel = density_curves(nc::Slice(0, density_curves.shape().rows), i);

        // Convert to NumCpp arrays (no need to sort manually - interp1d handles it)
        nc::NdArray<double> density_channel_double = density_channel.astype<double>();
        nc::NdArray<double> log_exposure_double = log_exposure.astype<double>();

        // Create interp1d interpolator: density -> log_exposure (inverse interpolation)
        // This matches Python's interp1d(density_curves[:, i], log_exposure, kind='cubic')
        scipy::interpolate::interp1d interp(density_channel_double, log_exposure_double,
                                     scipy::interpolate::interp1d::Kind::Cubic);

        // Interpolate to find log exposure values at the target densities
        double loge0 = interp(static_cast<double>(density_0));
        double loge1 = interp(static_cast<double>(density_1));

        // Calculate gamma
        gamma[i] = static_cast<float>((density_1 - density_0) / (loge1 - loge0));
    }

    return gamma;
}

} // namespace utils
} // namespace agx

int main() {
    // Use the exact same data as Python
    std::vector<float> log_exposure_vec = {
        -2.0f, -1.78947368f, -1.57894737f, -1.36842105f, -1.15789474f, -0.94736842f,
        -0.73684211f, -0.52631579f, -0.31578947f, -0.10526316f, 0.10526316f, 0.31578947f,
        0.52631579f, 0.73684211f, 0.94736842f, 1.15789474f, 1.36842105f, 1.57894737f,
        1.78947368f, 2.0f
    };
    
    std::vector<float> density_curves_vec = {
        0.54944814f, 1.24085717f, 0.97839273f,
        0.81839018f, 0.28722237f, 0.28719342f,
        0.16970033f, 1.13941137f, 0.82133801f,
        0.94968709f, 0.12470139f, 1.26389182f,
        1.09893117f, 0.35480693f, 0.31818996f,
        0.32008541f, 0.46509069f, 0.72970772f,
        0.61833402f, 0.44947497f, 0.83422347f,
        0.26739263f, 0.45057358f, 0.53963421f,
        0.64728398f, 1.04221115f, 0.33960854f,
        0.71708133f, 0.81089748f, 0.1557405f,
        0.82905382f, 0.30462895f, 0.17806191f,
        1.23866264f, 1.25875844f, 1.07007682f,
        0.46553652f, 0.21720654f, 0.92107963f,
        0.62818299f, 0.24644588f, 0.69421229f,
        0.14126623f, 1.19118448f, 0.41053598f,
        0.89502674f, 0.47405329f, 0.72408163f,
        0.75605234f, 0.32182535f, 1.26350155f,
        1.03015939f, 1.22739873f, 1.17379282f,
        0.81747997f, 1.20624908f, 0.206191f,
        0.33517943f, 0.15427275f, 0.4903964f
    };
    
    // Convert to NumCpp arrays
    nc::NdArray<float> log_exposure(log_exposure_vec);
    nc::NdArray<float> density_curves(20, 3);
    
    for (int i = 0; i < 20; ++i) {
        for (int j = 0; j < 3; ++j) {
            density_curves(i, j) = density_curves_vec[i * 3 + j];
        }
    }
    
    float density_0 = 0.25f;
    float density_1 = 1.0f;
    
    std::cout << "=== Testing Updated measure.cpp vs Python ===" << std::endl;
    std::cout << "Target densities: " << density_0 << ", " << density_1 << std::endl;
    
    // Test our C++ measure_gamma function
    nc::NdArray<float> gamma_cpp = agx::utils::measure_gamma(log_exposure, density_curves, density_0, density_1);
    
    std::cout << std::fixed << std::setprecision(12);
    std::cout << "\nC++ measure_gamma results:" << std::endl;
    std::cout << "  gamma[0] = " << gamma_cpp[0] << std::endl;
    std::cout << "  gamma[1] = " << gamma_cpp[1] << std::endl;
    std::cout << "  gamma[2] = " << gamma_cpp[2] << std::endl;
    
    // Expected Python results (from the Python test above)
    float expected_gamma_0 = 0.374881737243f;
    float expected_gamma_1 = -0.919590255039f;
    float expected_gamma_2 = 0.812494934093f;
    
    std::cout << "\nExpected Python results:" << std::endl;
    std::cout << "  gamma[0] = " << expected_gamma_0 << std::endl;
    std::cout << "  gamma[1] = " << expected_gamma_1 << std::endl;
    std::cout << "  gamma[2] = " << expected_gamma_2 << std::endl;
    
    std::cout << "\nDifferences:" << std::endl;
    std::cout << "  gamma[0] diff: " << std::abs(gamma_cpp[0] - expected_gamma_0) << std::endl;
    std::cout << "  gamma[1] diff: " << std::abs(gamma_cpp[1] - expected_gamma_1) << std::endl;
    std::cout << "  gamma[2] diff: " << std::abs(gamma_cpp[2] - expected_gamma_2) << std::endl;
    
    float max_diff = std::max({std::abs(gamma_cpp[0] - expected_gamma_0),
                              std::abs(gamma_cpp[1] - expected_gamma_1),
                              std::abs(gamma_cpp[2] - expected_gamma_2)});
    
    std::cout << "\nMax difference: " << max_diff << std::endl;
    
    // Check if differences are within acceptable range (< 0.001)
    bool success = (max_diff < 0.001f);
    std::cout << "Test " << (success ? "PASSED" : "FAILED") << std::endl;
    
    return success ? 0 : 1;
} 