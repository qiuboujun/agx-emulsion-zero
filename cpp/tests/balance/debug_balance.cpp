#include <iostream>
#include <vector>
#include "balance.hpp"
#include "config.hpp"
#include "profile_io.hpp"
#include "illuminants.hpp"

int main() {
    try {
        // Initialize config
        agx::config::initialize_config();
        
        // Load actual profile data
        std::cout << "Loading profile..." << std::endl;
        auto profile = agx::profiles::ProfileIO::load_from_file("agx_emulsion/data/profiles/kodak_portra_400_au.json");
        
        std::cout << "Profile loaded successfully" << std::endl;
        std::cout << "Dye density shape: " << profile.data.dye_density.shape().rows 
                  << "x" << profile.data.dye_density.shape().cols << std::endl;
        
        // Get viewing illuminant (use a default for now)
        auto illuminant = agx::model::standard_illuminant("D65");
        std::cout << "Illuminant shape: " << illuminant.shape().rows 
                  << "x" << illuminant.shape().cols << std::endl;
        
        std::cout << "Calling balance_metameric_neutral_with_illuminant..." << std::endl;
        
        auto result = agx::profiles::balance_metameric_neutral_with_illuminant(
            profile.data.dye_density, illuminant, 0.184f);
        
        std::cout << "Success! Result:" << std::endl;
        std::cout << "d_cmy_metameric: [" << result.d_cmy_metameric[0] << ", " 
                  << result.d_cmy_metameric[1] << ", " << result.d_cmy_metameric[2] << "]" << std::endl;
        std::cout << "d_cmy_scale: [" << result.d_cmy_scale[0] << ", " 
                  << result.d_cmy_scale[1] << ", " << result.d_cmy_scale[2] << "]" << std::endl;
        std::cout << "dye_density_out shape: " << result.dye_density_out.shape().rows 
                  << "x" << result.dye_density_out.shape().cols << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    } catch (...) {
        std::cerr << "Unknown exception occurred" << std::endl;
        return 1;
    }
    
    return 0;
}
