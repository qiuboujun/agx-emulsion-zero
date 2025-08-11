// SPDX-License-Identifier: MIT

#include <iostream>
#include <cmath>
#include <fstream>
#include "crop_resize.hpp"

int main() {
    try {
        const int H = 100, W = 150;
        nc::NdArray<float> img(H, W * 3);
        for (int y = 0; y < H; ++y) for (int x = 0; x < W; ++x) {
            float v = (float)y / (H - 1);
            img(y, x * 3 + 0) = v;
            img(y, x * 3 + 1) = v;
            img(y, x * 3 + 2) = v;
        }
        auto cropped = agx::utils::crop_image(img, {0.5f, 0.5f}, {0.3f, 0.2f});
        auto resized_cpu = agx::utils::resize_image_bilinear(cropped, 64, 96);
        auto resized_auto = agx::utils::resize_image_bilinear_auto(cropped, 64, 96, true);
        double max_abs = 0.0;
        for (size_t i = 0; i < resized_cpu.size(); ++i) {
            max_abs = std::max(max_abs, (double)std::abs(resized_cpu[i] - resized_auto[i]));
        }
        std::cout << "Crop size: " << cropped.shape().rows << "x" << cropped.shape().cols / 3 << "\n";
        std::cout << "Resize diff (CPU vs AUTO): " << max_abs << "\n";

        // Write cropped result to a fixed JSON path for Python comparison
        const char* out_path = "/home/jimmyqiu/cursor/agx-emulsion-zero/cpp/tests/crop_resize/tmp_crop_cpp.json";
        std::ofstream ofs(out_path, std::ios::out | std::ios::trunc);
        ofs << "{\n";
        ofs << "  \"H\": " << cropped.shape().rows << ",\n";
        ofs << "  \"W\": " << (cropped.shape().cols / 3) << ",\n";
        ofs << "  \"data\": [\n";
        for (size_t y = 0; y < cropped.shape().rows; ++y) {
            ofs << "    [";
            for (size_t x = 0; x < cropped.shape().cols; ++x) {
                ofs << cropped(y, x);
                if (x + 1 < cropped.shape().cols) ofs << ", ";
            }
            ofs << "]";
            if (y + 1 < cropped.shape().rows) ofs << ",";
            ofs << "\n";
        }
        ofs << "  ]\n";
        ofs << "}\n";
        ofs.close();
        std::cout << "Wrote: " << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


