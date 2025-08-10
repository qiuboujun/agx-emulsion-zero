// SPDX-License-Identifier: MIT

#include <cassert>
#include <iostream>
#include <string>

#include "profile_io.hpp"

using namespace agx::profiles;

int main() {
    try {
        const std::string src = "agx_emulsion/data/profiles/kodak_portra_400_au.json";
        const std::string dst = "cpp/tests/io/tmp_profile_copy.json";

        std::cout << "Loading profile from: " << src << std::endl;
        
        // Load
        Profile p = ProfileIO::load_from_file(src);
        
        std::cout << "Successfully loaded profile for stock: " << p.info.stock << std::endl;
        std::cout << "Profile data shapes:" << std::endl;
        std::cout << "  log_sensitivity: " << p.data.log_sensitivity.shape().rows << "x" << p.data.log_sensitivity.shape().cols << std::endl;
        std::cout << "  density_curves: " << p.data.density_curves.shape().rows << "x" << p.data.density_curves.shape().cols << std::endl;
        std::cout << "  density_curves_layers: " << p.data.density_curves_layers.shape().rows << "x" << p.data.density_curves_layers.shape().cols << std::endl;
        std::cout << "  dye_density: " << p.data.dye_density.shape().rows << "x" << p.data.dye_density.shape().cols << std::endl;
        std::cout << "  log_exposure: " << p.data.log_exposure.shape().rows << "x" << p.data.log_exposure.shape().cols << std::endl;
        std::cout << "  wavelengths: " << p.data.wavelengths.shape().rows << "x" << p.data.wavelengths.shape().cols << std::endl;

        std::cout << "Saving profile to: " << dst << std::endl;
        
        // Save copy
        ProfileIO::save_to_file(p, dst);
        
        std::cout << "Successfully saved profile" << std::endl;
        std::cout << "Reloading profile for comparison..." << std::endl;

        // Reload and compare
        Profile q = ProfileIO::load_from_file(dst);

        assert(p.info.stock == q.info.stock);
        // Now we properly handle NaN/Inf values, so all arrays should be equal
        assert(arrays_equal(p.data.log_sensitivity, q.data.log_sensitivity));
        assert(arrays_equal(p.data.density_curves, q.data.density_curves));
        assert(arrays_equal(p.data.density_curves_layers, q.data.density_curves_layers));
        assert(arrays_equal(p.data.dye_density, q.data.dye_density));
        assert(arrays_equal(p.data.log_exposure, q.data.log_exposure));
        assert(arrays_equal(p.data.wavelengths, q.data.wavelengths));

        std::cout << "Profile I/O test passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


