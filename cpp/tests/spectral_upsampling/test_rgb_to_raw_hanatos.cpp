// SPDX-License-Identifier: MIT

#include <iostream>
#include <fstream>
#include "spectral_upsampling.hpp"

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
        auto raw = agx::utils::rgb_to_raw_hanatos2025(rgb, sens, "sRGB", true, "D65", lut);
        std::cout<<"RAW first row: ["<<raw(0,0)<<", "<<raw(0,1)<<", "<<raw(0,2)<<"]\n";
        std::cout<<"RAW last row: ["<<raw(N-1,0)<<", "<<raw(N-1,1)<<", "<<raw(N-1,2)<<"]\n";
        // emit CSV to compare with Python
        std::ofstream ofs("../build/tmp_rgb_to_raw_cpp.csv");
        for(int i=0;i<N;++i){
            ofs<<raw(i,0)<<","<<raw(i,1)<<","<<raw(i,2)<<"\n";
        }
        return 0;
    }catch(const std::exception& e){
        std::cerr<<"Error: "<<e.what()<<"\n"; return 1;
    }
}


