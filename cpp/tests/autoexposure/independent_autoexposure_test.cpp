// SPDX-License-Identifier: MIT

#include <iostream>
#include "autoexposure.hpp"

int main() {
    try {
        // Fixed random-ish image: gradient to be deterministic
        const int H = 128, W = 192;
        nc::NdArray<float> img(H, W * 3);
        for (int y = 0; y < H; ++y) {
            for (int x = 0; x < W; ++x) {
                float v = (float)y / (H - 1) * 0.8f + (float)x / (W - 1) * 0.2f; // 0..1
                img(y, x * 3 + 0) = v;
                img(y, x * 3 + 1) = v * 0.9f;
                img(y, x * 3 + 2) = v * 1.1f;
            }
        }
        float ev_cpu = agx::utils::measure_autoexposure_ev(img, true, "center_weighted");
        float ev_auto = agx::utils::measure_autoexposure_ev_auto(img, true, "center_weighted");
        std::cout << "EV CPU:  " << ev_cpu << "\n";
        std::cout << "EV AUTO: " << ev_auto << "\n";
        std::cout << "Abs diff: " << std::abs(ev_cpu - ev_auto) << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


