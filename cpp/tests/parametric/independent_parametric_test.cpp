// SPDX-License-Identifier: MIT

#include <iostream>
#include <cmath>
#include <fstream>
#include <sstream>
#include <unistd.h>
#include <limits.h>
#include "parametric.hpp"

static std::string get_cwd() {
    char buf[PATH_MAX];
    if (getcwd(buf, sizeof(buf)) != nullptr) return std::string(buf);
    return std::string(".");
}

int main() {
    try {
        // Fixed input matching the Python function signature
        const int N = 256;
        nc::NdArray<float> log_exposure(1, N);
        for (int i = 0; i < N; ++i) log_exposure[i] = -2.0f + i * (4.0f / (N - 1));

        std::array<float,3> gamma = {0.65f, 0.62f, 0.68f};
        std::array<float,3> loge0 = {0.0f, 0.0f, 0.0f};
        std::array<float,3> dmax  = {2.2f, 2.1f, 2.3f};
        std::array<float,3> toe   = {0.25f, 0.25f, 0.25f};
        std::array<float,3> shldr = {0.5f, 0.5f, 0.5f};

        auto cpu = agx::model::parametric_density_curves_model(log_exposure, gamma, loge0, dmax, toe, shldr);
        auto auto_accel = agx::model::parametric_density_curves_model_auto(log_exposure, gamma, loge0, dmax, toe, shldr, true);

        // Compare CPU vs auto (CUDA if available)
        float max_abs_diff = 0.0f;
        for (size_t i = 0; i < cpu.size(); ++i) {
            max_abs_diff = std::max(max_abs_diff, std::abs(cpu[i] - auto_accel[i]));
        }
        std::cout << "Max abs diff (CPU vs AUTO): " << max_abs_diff << "\n";

        // Emit compact JSON for Python comparison
        std::string out_path = std::string("/home/jimmyqiu/cursor/agx-emulsion-zero/cpp/tests/parametric/tmp_parametric_cpp.json");
        std::ofstream ofs(out_path, std::ios::out | std::ios::trunc);
        ofs << "{\n";
        ofs << "  \"N\": " << N << ",\n";
        ofs << "  \"log_exposure\": [";
        for (int i = 0; i < N; ++i) { ofs << log_exposure[i]; if (i+1<N) ofs << ", "; }
        ofs << "],\n";
        auto dump_arr = [&](const char* name, const std::array<float,3>& a){
            ofs << "  \"" << name << "\": [" << a[0] << ", " << a[1] << ", " << a[2] << "],\n";
        };
        dump_arr("gamma", gamma);
        dump_arr("log_exposure_0", loge0);
        dump_arr("density_max", dmax);
        dump_arr("toe_size", toe);
        dump_arr("shoulder_size", shldr);
        ofs << "  \"density_curves\": [\n";
        for (int i = 0; i < N; ++i) {
            ofs << "    [" << auto_accel(i,0) << ", " << auto_accel(i,1) << ", " << auto_accel(i,2) << "]";
            if (i+1<N) ofs << ",";
            ofs << "\n";
        }
        ofs << "  ]\n";
        ofs << "}\n";
        ofs.flush();
        ofs.close();
        std::cout << "Wrote: " << out_path << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}


