#include <iostream>
#include <fstream>
#include "process.hpp"
#include "config.hpp"

int main(){
    // Initialize global spectral shape, CMFS, and log exposure grids
    agx::config::initialize_config();
    using namespace agx::process;
    // Build params matching Python defaults
    Params params;
    params.profiles.negative = "kodak_portra_400_auc";
    params.profiles.print_paper = "kodak_portra_endura_uc";
    params.io.input_color_space = "sRGB";
    params.io.input_cctf_decoding = false;
    params.io.output_color_space = "sRGB";

    // Create a small fixed image (5x5 gradient)
    const int H=5, W=5;
    nc::NdArray<float> img(H, W*3);
    for (int i=0;i<H;++i){
        for (int j=0;j<W;++j){
            float v = float(i*W+j)/float(H*W-1);
            img(i, 3*j+0) = v;
            img(i, 3*j+1) = 0.8f*v;
            img(i, 3*j+2) = 1.2f*v;
        }
    }

    Process proc(params);
    auto out = proc.run(img);

    std::string out_csv = std::string(AGX_SOURCE_DIR) + "/build/tmp_process_cpp.csv";
    std::ofstream ofs(out_csv);
    for (uint32_t i=0;i<out.shape().rows;++i){
        for (uint32_t j=0;j<out.shape().cols; ++j){
            ofs << out(i,j);
            if (j+1<out.shape().cols) ofs << ",";
        }
        ofs << "\n";
    }
    std::cout << "Wrote " << out_csv << "\n";
    return 0;
}
