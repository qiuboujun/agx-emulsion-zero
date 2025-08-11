// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include <sstream>
#include <cmath>
#include "profile_io.hpp"
#include "reconstruct.hpp"

using namespace agx::profiles;

int main() {
    try {
        const std::string src = "agx_emulsion/data/profiles/kodak_portra_400_au.json";
        std::cout << "Loading profile: " << src << std::endl;
        Profile p = ProfileIO::load_from_file(src);

        ReconstructParams params; // defaults mirror Python initial values
        Profile q = reconstruct_dye_density(p, params, true);

        // Basic sanity: dye_density updated and midscale neutral set
        std::cout << "Midscale neutral: ["
                  << q.info.density_midscale_neutral[0] << ", "
                  << q.info.density_midscale_neutral[1] << ", "
                  << q.info.density_midscale_neutral[2] << "]\n";

        // Save result for manual inspection
        const std::string dst = "cpp/tests/reconstruct/tmp_reconstruct_output.json";
        ProfileIO::save_to_file(q, dst);
        std::cout << "Saved reconstructed profile to: " << dst << std::endl;

        // Also save a compact comparison JSON (dye_density first 10 rows and midscale neutral)
        std::ofstream cmp("cpp/tests/reconstruct/tmp_reconstruct_cmp.json");
        cmp << "{\n  \"midscale_neutral\": ["
            << q.info.density_midscale_neutral[0] << ", "
            << q.info.density_midscale_neutral[1] << ", "
            << q.info.density_midscale_neutral[2] << "],\n  \"dye_density_head\": [\n";
        const size_t rows = std::min<size_t>(10, q.data.dye_density.shape().rows);
        auto fmt = [](float v){
            if (std::isnan(v)) return std::string("null");
            std::ostringstream oss; oss.setf(std::ios::fixed); oss.precision(6); oss << v; return oss.str();
        };
        for (size_t i = 0; i < rows; ++i) {
            float a = q.data.dye_density(i,0);
            float b = q.data.dye_density(i,1);
            float c = q.data.dye_density(i,2);
            cmp << "    [" << fmt(a) << ", " << fmt(b) << ", " << fmt(c) << "]";
            if (i + 1 < rows) cmp << ",";
            cmp << "\n";
        }
        cmp << "  ]\n}\n";
        cmp.close();

        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


