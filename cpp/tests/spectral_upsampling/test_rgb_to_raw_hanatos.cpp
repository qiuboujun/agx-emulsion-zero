// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include "spectral_upsampling.hpp"
#include "fast_interp_lut.hpp"
#include "colour.hpp"

static std::vector<float> bilinear_interp_flat_LxL_K(const nc::NdArray<float>& lut_flat, int L, float x, float y) {
    const int K = lut_flat.shape().cols;
    x = std::max(0.0f, std::min(x, float(L - 1)));
    y = std::max(0.0f, std::min(y, float(L - 1)));
    int x0 = int(std::floor(x)); int y0 = int(std::floor(y));
    int x1 = std::min(x0 + 1, L - 1); int y1 = std::min(y0 + 1, L - 1);
    float fx = x - x0, fy = y - y0;
    float w00 = (1 - fx) * (1 - fy);
    float w10 = fx * (1 - fy);
    float w01 = (1 - fx) * fy;
    float w11 = fx * fy;
    int idx00 = x0 * L + y0;
    int idx10 = x1 * L + y0;
    int idx01 = x0 * L + y1;
    int idx11 = x1 * L + y1;
    std::vector<float> out(K, 0.0f);
    for (int k = 0; k < K; ++k) {
        float v00 = lut_flat(idx00, k);
        float v10 = lut_flat(idx10, k);
        float v01 = lut_flat(idx01, k);
        float v11 = lut_flat(idx11, k);
        out[k] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
    }
    return out;
}

int main(){
    try{
        // Fixed input pixel grid
        const int N=5;
        nc::NdArray<float> rgb(N,3);
        for(int i=0;i<N;++i){
            float v=0.1f*i;
            rgb(i,0)=v; rgb(i,1)=v*0.8f; rgb(i,2)=v*1.2f;
        }
        // Load the real Hanatos LUT for parity
        std::string repo = std::string(AGX_SOURCE_DIR) + "/..";
        std::string npy = repo + "/agx_emulsion/data/luts/spectral_upsampling/irradiance_xy_tc.npy";
        auto lut = agx::utils::load_hanatos_spectra_lut_npy(npy);
        // Sensitivity: use K from LUT columns to build a trivial sensitivity of ones
        nc::NdArray<float> sens(lut.shape().cols, 3); sens.fill(1.0f);

        // Compute tc and b
        auto [tc, b] = agx::utils::rgb_to_tc_b_cpp(rgb, "sRGB", true, "D65");
        std::cout << "tc last: [" << tc(N-1,0) << ", " << tc(N-1,1) << "]\n";
        std::cout << "b last: " << b(N-1,0) << "\n";

        // Derive XYZ and xy for last sample
        nc::NdArray<float> last_rgb(1,3); last_rgb(0,0)=rgb(N-1,0); last_rgb(0,1)=rgb(N-1,1); last_rgb(0,2)=rgb(N-1,2);
        nc::NdArray<float> illum_xy(1,2); illum_xy(0,0)=0.3127f; illum_xy(0,1)=0.3290f; // D65
        auto xyz = colour::RGB_to_XYZ(last_rgb, "sRGB", true, illum_xy, "CAT02");
        float X=xyz(0,0), Y=xyz(0,1), Z=xyz(0,2);
        float S = std::max(1e-10f, X+Y+Z);
        std::cout << "XYZ last: ["<<X<<", "<<Y<<", "<<Z<<"] xy: ["<< (X/S) << ", " << (Y/S) << "]\n";

        // Preproject LUT by sensitivity -> (L*L,3)
        const int rows = lut.shape().rows; const int K = lut.shape().cols;
        nc::NdArray<float> lut_proj(rows, 3);
        for (int r=0;r<rows;++r){ for(int c=0;c<3;++c){ float acc=0; for(int k=0;k<K;++k) acc += lut(r,k)*sens(k,c); lut_proj(r,c)=acc; }}
        const int L = int(std::round(std::sqrt(rows)));

        // Sample projected LUT at last tc and multiply by b (CPU cubic)
        float x_in = tc(N-1,0) * (L - 1);
        float y_in = tc(N-1,1) * (L - 1);
        std::vector<float> raw_pre_vec = agx::cubic_interp_lut_at_2d(lut_proj, x_in, y_in);
        nc::NdArray<float> raw_pre(1,3); raw_pre(0,0)=raw_pre_vec[0]; raw_pre(0,1)=raw_pre_vec[1]; raw_pre(0,2)=raw_pre_vec[2];
        for(int c=0;c<3;++c) raw_pre(0,c) *= b(N-1,0);
        std::cout << "raw_pre_norm last: [" << raw_pre(0,0) << ", " << raw_pre(0,1) << ", " << raw_pre(0,2) << "]\n";

        // Midgray normalization factor
        nc::NdArray<float> mid(1,3); mid(0,0)=0.184f; mid(0,1)=0.184f; mid(0,2)=0.184f;
        auto [tc_m, b_m] = agx::utils::rgb_to_tc_b_cpp(mid, "sRGB", false, "D65");
        float xm = tc_m(0,0)*(L-1), ym = tc_m(0,1)*(L-1);
        auto spec_mid = bilinear_interp_flat_LxL_K(lut, L, xm, ym);
        float raw_mid[3]={0,0,0};
        for(int c=0;c<3;++c){ float acc=0; for(int k=0;k<K;++k) acc += spec_mid[k]*sens(k,c); raw_mid[c]=acc*b_m(0,0); }
        float scale = 1.0f/std::max(1e-10f, raw_mid[1]);
        std::cout << "midgray scale: " << scale << "\n";
        std::cout << "raw_post_norm last (pred): [" << raw_pre(0,0)*scale << ", " << raw_pre(0,1)*scale << ", " << raw_pre(0,2)*scale << "]\n";

        auto raw = agx::utils::rgb_to_raw_hanatos2025(rgb, sens, "sRGB", true, "D65", lut);
        std::cout<<"RAW first row: ["<<raw(0,0)<<", "<<raw(0,1)<<", "<<raw(0,2)<<"]\n";
        std::cout<<"RAW last row: ["<<raw(N-1,0)<<", "<<raw(N-1,1)<<", "<<raw(N-1,2)<<"]\n";
        // emit CSV to compare with Python (absolute path within cpp/build)
        std::string out_csv = std::string(AGX_SOURCE_DIR) + "/build/tmp_rgb_to_raw_cpp.csv";
        std::ofstream ofs(out_csv);
        for(int i=0;i<N;++i){
            ofs<<raw(i,0)<<","<<raw(i,1)<<","<<raw(i,2)<<"\n";
        }
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<"\n"; return 1;
    }
}


