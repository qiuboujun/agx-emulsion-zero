#include <iostream>
#include <fstream>
#include <string>
#include <iomanip>
#include "io.hpp"
#include "config.hpp"

void save_array_to_file(const nc::NdArray<float>& arr, const std::string& filename, const std::string& name) {
    std::ofstream file(filename);
    file << "# " << name << std::endl;
    file << "# Shape: (" << arr.shape().rows << ", " << arr.shape().cols << ")" << std::endl;
    file << "# dtype: float" << std::endl;
    file << "# Data:" << std::endl;
    file << std::fixed << std::setprecision(10);
    
    // For NumCpp arrays, always save as 2D format for consistency
    for (nc::uint32 i = 0; i < arr.shape().rows; ++i) {
        for (nc::uint32 j = 0; j < arr.shape().cols; ++j) {
            file << arr(i, j);
            if (j < arr.shape().cols - 1) file << " ";
        }
        file << std::endl;
    }
}

int main() {
    std::cout << "Testing C++ load_agx_emulsion_data..." << std::endl;
    
    // Initialize the global configuration
    agx::config::initialize_config();
    
    // Test parameters - using kodak_vision3_500t which has all required files
    std::string stock = "kodak_vision3_500t";
    std::string log_sensitivity_donor = "";
    std::string density_curves_donor = "";
    std::string dye_density_cmy_donor = "";
    std::string dye_density_min_mid_donor = "";
    std::string type = "negative";
    bool color = true;
    
    try {
        // Call the function
        AgxEmulsionData result = agx::utils::load_agx_emulsion_data(
            stock,
            log_sensitivity_donor,
            density_curves_donor,
            dye_density_cmy_donor,
            dye_density_min_mid_donor,
            type,
            color
        );
        
        std::cout << "Log sensitivity shape: (" << result.log_sensitivity.shape().rows 
                  << ", " << result.log_sensitivity.shape().cols << ")" << std::endl;
        std::cout << "Dye density shape: (" << result.dye_density.shape().rows 
                  << ", " << result.dye_density.shape().cols << ")" << std::endl;
        std::cout << "Wavelengths shape: (" << result.wavelengths.shape().rows 
                  << ", " << result.wavelengths.shape().cols << ")" << std::endl;
        std::cout << "Density curves shape: (" << result.density_curves.shape().rows 
                  << ", " << result.density_curves.shape().cols << ")" << std::endl;
        std::cout << "Log exposure shape: (" << result.log_exposure.shape().rows 
                  << ", " << result.log_exposure.shape().cols << ")" << std::endl;
        
        // Save results to files
        save_array_to_file(result.log_sensitivity, "cpp_log_sensitivity.txt", "Log Sensitivity");
        save_array_to_file(result.dye_density, "cpp_dye_density.txt", "Dye Density");
        save_array_to_file(result.wavelengths, "cpp_wavelengths.txt", "Wavelengths");
        save_array_to_file(result.density_curves, "cpp_density_curves.txt", "Density Curves");
        save_array_to_file(result.log_exposure, "cpp_log_exposure.txt", "Log Exposure");
        
        std::cout << "C++ results saved to files with 'cpp_' prefix" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
} 