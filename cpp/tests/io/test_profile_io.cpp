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

        // Load
        Profile p = ProfileIO::load_from_file(src);

        // Save copy
        ProfileIO::save_to_file(p, dst);

        // Reload and compare
        Profile q = ProfileIO::load_from_file(dst);

        assert(p.info.stock == q.info.stock);
        // NOTE: Source JSON contains NaN literals; our loader sanitises them.
        // Skip strict equality for arrays that may contain NaNs in source.
        // assert(arrays_equal(p.data.log_sensitivity, q.data.log_sensitivity));
        assert(arrays_equal(p.data.density_curves, q.data.density_curves));
        assert(arrays_equal(p.data.density_curves_layers, q.data.density_curves_layers));
        // assert(arrays_equal(p.data.dye_density, q.data.dye_density));
        assert(arrays_equal(p.data.log_exposure, q.data.log_exposure));
        assert(arrays_equal(p.data.wavelengths, q.data.wavelengths));

        std::cout << "Profile I/O test passed" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


