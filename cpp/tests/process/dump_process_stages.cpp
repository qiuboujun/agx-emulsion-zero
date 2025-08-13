#include <iostream>
#include <fstream>
#include <nlohmann/json.hpp>

#include "process.hpp"
#include "config.hpp"
#include "profile_io.hpp"
#include "spectral_upsampling.hpp"
#include "illuminants.hpp"
#include "color_filters.hpp"
#include "density_spectral.hpp"
#include "density_curves.hpp"
#include "conversions.hpp"
#include "autoexposure.hpp"
#include "fast_interp_lut.hpp"

using json = nlohmann::json;

static nc::NdArray<float> hw3_to_hw_by3(const nc::NdArray<float>& img_hw3){
    const int H = static_cast<int>(img_hw3.shape().rows);
    const int W3 = static_cast<int>(img_hw3.shape().cols);
    const int W = W3 / 3;
    nc::NdArray<float> out(H * W, 3);
    for (int i = 0; i < H; ++i){
        for (int w = 0; w < W; ++w){
            for (int c = 0; c < 3; ++c){
                out(i*W + w, c) = img_hw3(i, w*3 + c);
            }
        }
    }
    return out;
}

static nc::NdArray<float> hw_by3_to_hw3(const nc::NdArray<float>& img_hw_by3, int H, int W){
    nc::NdArray<float> out(H, W*3);
    for (int i = 0; i < H; ++i){
        for (int w = 0; w < W; ++w){
            for (int c = 0; c < 3; ++c){
                out(i, w*3 + c) = img_hw_by3(i*W + w, c);
            }
        }
    }
    return out;
}

static nc::NdArray<float> dot_blocks_K3(const nc::NdArray<float>& A, const nc::NdArray<float>& B, int W){
    const int H = static_cast<int>(A.shape().rows);
    const int K = static_cast<int>(B.shape().rows);
    nc::NdArray<float> out(H, W*3);
    for (int w = 0; w < W; ++w){
        for (int i = 0; i < H; ++i){
            for (int c = 0; c < 3; ++c){
                float acc = 0.0f;
                for (int k = 0; k < K; ++k){
                    acc += A(i, w*K + k) * B(k, c);
                }
                out(i, w*3 + c) = acc;
            }
        }
    }
    return out;
}

int main(){
    agx::config::initialize_config();
    using namespace agx::process;
    using agx::profiles::ProfileIO;

    // Params
    Params params;
    params.profiles.negative = "kodak_portra_400_auc";
    params.profiles.print_paper = "kodak_portra_endura_uc";
    params.io.input_color_space = "sRGB";
    params.io.input_cctf_decoding = false;
    params.io.output_color_space = "sRGB";
    params.camera.auto_exposure = true;

    // Load profiles
    std::string root = std::string(AGX_SOURCE_DIR) + "/cpp/data/profiles/";
    auto neg = ProfileIO::load_from_file(root + params.profiles.negative + ".json");
    auto paper = ProfileIO::load_from_file(root + params.profiles.print_paper + ".json");

    // Image
    const int H=5, W=5;
    nc::NdArray<float> image(H, W*3);
    for (int i=0;i<H;++i){
        for (int j=0;j<W;++j){
            float v = float(i*W+j)/float(H*W-1);
            image(i, 3*j+0) = v;
            image(i, 3*j+1) = 0.8f*v;
            image(i, 3*j+2) = 1.2f*v;
        }
    }

    // Auto exposure
    float ev = agx::utils::measure_autoexposure_ev_auto(image, params.io.input_cctf_decoding, params.camera.auto_exposure_method);

    // Sensitivity (neg)
    nc::NdArray<float> sensitivity(neg.data.log_sensitivity.shape().rows, neg.data.log_sensitivity.shape().cols);
    for (uint32_t i=0;i<neg.data.log_sensitivity.shape().rows;++i)
        for (uint32_t j=0;j<neg.data.log_sensitivity.shape().cols;++j)
            sensitivity(i,j) = std::pow(10.0f, neg.data.log_sensitivity(i,j));
    sensitivity = nc::nan_to_num(sensitivity);

    // Hanatos LUT
    auto lut = agx::utils::load_hanatos_spectra_lut_npy(std::string(AGX_SOURCE_DIR) + "/cpp/data/luts/spectral_upsampling/irradiance_xy_tc.npy");
    auto image_hw_by3 = hw3_to_hw_by3(image);

    // Center pixel index and RGB
    int ci = H/2, cj = W/2;
    nc::NdArray<float> center_rgb(1,3);
    center_rgb(0,0) = image(ci, cj*3+0);
    center_rgb(0,1) = image(ci, cj*3+1);
    center_rgb(0,2) = image(ci, cj*3+2);

    // Compute tc and b for center
    auto tb = agx::utils::rgb_to_tc_b_cpp(center_rgb, params.io.input_color_space, params.io.input_cctf_decoding, neg.info.reference_illuminant);
    auto tc_center = tb.first; // (1,2)
    auto b_center = tb.second; // (1,1)

    // Preproject LUT by sensitivity (L*L, K) dot (K,3) -> (L*L,3)
    const int rows = lut.shape().rows;
    const int K = lut.shape().cols;
    nc::NdArray<float> lut_proj(rows, 3);
    for (int r=0;r<rows;++r){
        for (int c=0;c<3;++c){
            float acc = 0.0f;
            for (int k=0;k<K;++k) acc += lut(r,k) * sensitivity(k,c);
            lut_proj(r,c) = acc;
        }
    }
    const int L = static_cast<int>(std::round(std::sqrt(rows)));

    // Apply 2D LUT at tc_center using cubic (Mitchell)
    nc::NdArray<float> tc_img(1,2);
    tc_img(0,0) = tc_center(0,0);
    tc_img(0,1) = tc_center(0,1);
    auto raw_pre_center_nc = agx::apply_lut_cubic_2d(lut_proj, tc_img, /*height*/1, /*width*/1); // (1,3)

    // Scale by b
    nc::NdArray<float> raw_pre_scaled(1,3);
    for (int c=0;c<3;++c) raw_pre_scaled(0,c) = raw_pre_center_nc(0,c) * b_center(0,0);

    // Midgray normalization (linear RGI equivalent)
    nc::NdArray<float> mid_rgb(1,3); mid_rgb(0,0)=0.184f; mid_rgb(0,1)=0.184f; mid_rgb(0,2)=0.184f;
    auto tb_mid = agx::utils::rgb_to_tc_b_cpp(mid_rgb, params.io.input_color_space, false, neg.info.reference_illuminant);
    const float x_m = tb_mid.first(0,0) * (L - 1);
    const float y_m = tb_mid.first(0,1) * (L - 1);
    // Bilinear over original spectra LUT
    // local helper
    auto bilinear = [&](float x, float y){
        const int x0 = static_cast<int>(std::floor(x));
        const int y0 = static_cast<int>(std::floor(y));
        const int x1 = std::min(x0 + 1, L - 1);
        const int y1 = std::min(y0 + 1, L - 1);
        const float fx = x - static_cast<float>(x0);
        const float fy = y - static_cast<float>(y0);
        const float w00 = (1.0f - fx) * (1.0f - fy);
        const float w10 = fx * (1.0f - fy);
        const float w01 = (1.0f - fx) * fy;
        const float w11 = fx * fy;
        std::vector<float> out(K, 0.0f);
        const int idx00 = x0 * L + y0;
        const int idx10 = x1 * L + y0;
        const int idx01 = x0 * L + y1;
        const int idx11 = x1 * L + y1;
        for (int k=0;k<K;++k){
            float v00 = lut(idx00, k);
            float v10 = lut(idx10, k);
            float v01 = lut(idx01, k);
            float v11 = lut(idx11, k);
            out[k] = w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11;
        }
        return out;
    };
    auto mid_spec = bilinear(x_m, y_m);
    // scale spectrum by b_mid
    for (int k=0;k<K;++k) mid_spec[k] *= tb_mid.second(0,0);
    // project with sensitivity
    float raw_mid_center[3] = {0,0,0};
    for (int c=0;c<3;++c){
        float acc = 0.0f; for (int k=0;k<K;++k) acc += mid_spec[k] * sensitivity(k,c);
        raw_mid_center[c] = acc;
    }

    auto raw_hw_by3 = agx::utils::rgb_to_raw_hanatos2025(image_hw_by3, sensitivity, params.io.input_color_space, params.io.input_cctf_decoding, neg.info.reference_illuminant, lut);
    raw_hw_by3 *= std::pow(2.0f, ev);
    auto raw = hw_by3_to_hw3(raw_hw_by3, H, W);

    // Develop film
    auto log_raw_hw3 = nc::log10(raw + 1e-10f);
    auto log_raw = hw3_to_hw_by3(log_raw_hw3);
    auto dc_neg = agx_emulsion::Matrix(neg.data.density_curves.shape().rows, neg.data.density_curves.shape().cols);
    for (uint32_t i=0;i<neg.data.density_curves.shape().rows;++i)
        for (uint32_t j=0;j<neg.data.density_curves.shape().cols;++j)
            dc_neg(i,j)=neg.data.density_curves(i,j);
    std::vector<double> le_neg(neg.data.log_exposure.size());
    for (uint32_t i=0;i<neg.data.log_exposure.size();++i){ le_neg[i]=neg.data.log_exposure[i]; }
    std::array<double,3> gamma = {1.0,1.0,1.0};
    agx_emulsion::Matrix log_raw_mat(log_raw.shape().rows, log_raw.shape().cols);
    for (uint32_t i=0;i<log_raw.shape().rows;++i)
        for (uint32_t j=0;j<log_raw.shape().cols;++j)
            log_raw_mat(i,j)=log_raw(i,j);
    auto density_cmy_hw_by3 = agx_emulsion::interpolate_exposure_to_density(log_raw_mat, dc_neg, le_neg, gamma);
    // Convert back to NdArray
    nc::NdArray<float> density_cmy_flat(log_raw.shape().rows, 3);
    for (uint32_t i=0;i<density_cmy_flat.shape().rows;++i)
        for (uint32_t j=0;j<3;++j)
            density_cmy_flat(i,j) = static_cast<float>(density_cmy_hw_by3(i,j));
    auto density_cmy = hw_by3_to_hw3(density_cmy_flat, H, W);

    // Enlarge and print
    auto light_src = agx::model::standard_illuminant(params.enlarger.illuminant);
    auto enlarger_ill = agx::model::color_enlarger(light_src,
        params.enlarger.y_filter_neutral * agx::config::ENLARGER_STEPS + params.enlarger.y_filter_shift,
        params.enlarger.m_filter_neutral * agx::config::ENLARGER_STEPS + params.enlarger.m_filter_shift,
        params.enlarger.c_filter_neutral * agx::config::ENLARGER_STEPS);

    auto density_spec = agx::utils::compute_density_spectral(density_cmy, neg.data.dye_density, 1.0f);
    auto light = agx::utils::density_to_light(density_spec, enlarger_ill);

    nc::NdArray<float> paper_sens(paper.data.log_sensitivity.shape().rows, paper.data.log_sensitivity.shape().cols);
    for (uint32_t i=0;i<paper.data.log_sensitivity.shape().rows;++i)
        for (uint32_t j=0;j<paper.data.log_sensitivity.shape().cols;++j)
            paper_sens(i,j) = std::pow(10.0f, paper.data.log_sensitivity(i,j));
    paper_sens = nc::nan_to_num(paper_sens);
    auto cmy = dot_blocks_K3(light, paper_sens, W);
    cmy *= params.enlarger.print_exposure;

    // midgray factor
    nc::NdArray<float> rgb_mid(1,3); rgb_mid(0,0)=0.184f; rgb_mid(0,1)=0.184f; rgb_mid(0,2)=0.184f;
    auto raw_mid = agx::utils::rgb_to_raw_hanatos2025(rgb_mid, sensitivity, "sRGB", false, neg.info.reference_illuminant, lut);
    auto log_raw_mid = nc::log10(raw_mid + 1e-10f);
    agx_emulsion::Matrix log_raw_mid_mat(1,3);
    for (int j=0;j<3;++j) log_raw_mid_mat(0,j)=log_raw_mid(0,j);
    auto density_cmy_mid_mat = agx_emulsion::interpolate_exposure_to_density(log_raw_mid_mat, dc_neg, le_neg, gamma);
    nc::NdArray<float> density_cmy_mid(1,3);
    for (int j=0;j<3;++j) density_cmy_mid(0,j)=static_cast<float>(density_cmy_mid_mat(0,j));
    auto density_spec_mid = agx::utils::compute_density_spectral(density_cmy_mid, neg.data.dye_density, 1.0f);
    auto light_mid = agx::utils::density_to_light(density_spec_mid, enlarger_ill);
    auto raw_mid_print = nc::dot(light_mid, paper_sens);
    float midgray_factor = 1.0f / std::max(1e-10f, raw_mid_print(0,1));
    cmy *= midgray_factor;

    // Develop paper
    auto log_cmy_hw3 = nc::log10(cmy + 1e-10f);
    auto log_cmy_flat = hw3_to_hw_by3(log_cmy_hw3);
    auto dc_paper = agx_emulsion::Matrix(paper.data.density_curves.shape().rows, paper.data.density_curves.shape().cols);
    for (uint32_t i=0;i<paper.data.density_curves.shape().rows;++i)
        for (uint32_t j=0;j<paper.data.density_curves.shape().cols;++j)
            dc_paper(i,j)=paper.data.density_curves(i,j);
    std::vector<double> le_paper(paper.data.log_exposure.size());
    for (uint32_t i=0;i<paper.data.log_exposure.size();++i) le_paper[i]=paper.data.log_exposure[i];
    agx_emulsion::Matrix log_cmy_flat_mat(log_cmy_flat.shape().rows, log_cmy_flat.shape().cols);
    for (uint32_t i=0;i<log_cmy_flat.shape().rows;++i)
        for (uint32_t j=0;j<log_cmy_flat.shape().cols;++j)
            log_cmy_flat_mat(i,j)=log_cmy_flat(i,j);
    auto density_print_flat_mat = agx_emulsion::interpolate_exposure_to_density(log_cmy_flat_mat, dc_paper, le_paper, gamma);
    nc::NdArray<float> density_print_flat(log_cmy_flat.shape().rows, 3);
    for (uint32_t i=0;i<density_print_flat.shape().rows;++i)
        for (uint32_t j=0;j<3;++j)
            density_print_flat(i,j)=static_cast<float>(density_print_flat_mat(i,j));
    auto density_print = hw_by3_to_hw3(density_print_flat, H, W);

    // Scan
    auto scan_ill = agx::model::standard_illuminant(paper.info.viewing_illuminant).flatten();
    auto dens_spec_scan = agx::utils::compute_density_spectral(density_print, paper.data.dye_density, 1.0f);
    auto light_scan = agx::utils::density_to_light(dens_spec_scan, scan_ill);
    float norm = 0.0f; for(uint32_t i=0;i<agx::config::SPECTRAL_SHAPE.wavelengths.size();++i) norm += agx::config::STANDARD_OBSERVER_CMFS(i,1) * scan_ill[i];
    auto xyz_hw3 = dot_blocks_K3(light_scan, agx::config::STANDARD_OBSERVER_CMFS, W) / norm;
    auto xyz_flat = hw3_to_hw_by3(xyz_hw3);
    nc::NdArray<float> illuminant_xy(1,2); illuminant_xy(0,0)=0.3127f; illuminant_xy(0,1)=0.3290f;
    auto rgb_flat = colour::XYZ_to_RGB(xyz_flat, params.io.output_color_space, params.io.output_cctf_encoding, illuminant_xy, "CAT02");
    auto rgb = hw_by3_to_hw3(rgb_flat, H, W);

    // Center pixel index
    // int ci = H/2, cj = W/2; (already declared)
    int idx_center = cj; // within row

    // dump JSON
    json j;
    auto at3 = [&](const nc::NdArray<float>& hw3){ return std::array<float,3>{ hw3(ci, cj*3+0), hw3(ci, cj*3+1), hw3(ci, cj*3+2)}; };
    auto extractK = [&](const nc::NdArray<float>& hwK){ std::vector<float> out; int Kloc = agx::config::SPECTRAL_SHAPE.wavelengths.size(); out.reserve(Kloc); for (int k=0;k<Kloc;++k) out.push_back(hwK(ci, cj*Kloc + k)); return out; };

    j["image_center_rgb"] = at3(image);
    j["tc_center"] = std::vector<float>{ tc_center(0,0), tc_center(0,1) };
    j["b_center"] = b_center(0,0);
    j["raw_pre_center"] = std::vector<float>{ raw_pre_center_nc(0,0), raw_pre_center_nc(0,1), raw_pre_center_nc(0,2) };
    j["raw_pre_scaled_center"] = std::vector<float>{ raw_pre_scaled(0,0), raw_pre_scaled(0,1), raw_pre_scaled(0,2) };
    j["midgray_raw_center"] = std::vector<float>{ raw_mid_center[0], raw_mid_center[1], raw_mid_center[2] };
    j["ev"] = ev;
    j["raw_center"] = at3(raw);
    j["density_cmy_center"] = at3(density_cmy);
    j["density_spectral_center"] = extractK(density_spec);
    auto enlarger_ill_flat = enlarger_ill.flatten();
    j["print_illuminant"] = std::vector<float>(enlarger_ill_flat.begin(), enlarger_ill_flat.end());
    j["light_enlarger_center"] = extractK(light);
    j["cmy_pre_paper_center"] = at3(cmy);
    j["midgray_factor"] = midgray_factor;
    j["density_print_center"] = at3(density_print);
    j["scan_illuminant"] = std::vector<float>(scan_ill.begin(), scan_ill.end());
    j["light_scan_center"] = extractK(light_scan);
    j["xyz_center"] = at3(xyz_hw3);
    j["rgb_center"] = at3(rgb);

    std::string out_json = std::string(AGX_SOURCE_DIR) + "/build/tmp_process_cpp_stages.json";
    std::ofstream ofs(out_json);
    ofs << j.dump(2);
    std::cout << "Wrote stages JSON: " << out_json << "\n";
    return 0;
}


