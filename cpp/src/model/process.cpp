#include "process.hpp"
#include "autoexposure.hpp"
#include "crop_resize.hpp"
#include "profile_io.hpp"
#include "spectral_upsampling.hpp"
#include "illuminants.hpp"
#include "color_filters.hpp"
#include "config.hpp"
#include "density_spectral.hpp"
#include "density_curves.hpp"
#include "diffusion.hpp"
#include "colour.hpp"
#include "conversions.hpp"
#include "io.hpp"

using agx::profiles::Profile;
using agx::profiles::ProfileIO;

namespace agx { namespace process {

Process::Process(const Params& params) : params_(params) {}

static nc::NdArray<float> make_full(const nc::NdArray<float>& like, float value){
    nc::NdArray<float> out(like.shape().rows, like.shape().cols);
    for(uint32_t i=0;i<out.shape().rows;++i){ for(uint32_t j=0;j<out.shape().cols;++j){ out(i,j)=value; }}
    return out;
}

static agx_emulsion::Matrix toMat(const nc::NdArray<float>& a){
    agx_emulsion::Matrix m(a.shape().rows, a.shape().cols);
    for(uint32_t i=0;i<a.shape().rows;++i){ for(uint32_t j=0;j<a.shape().cols;++j){ m(i,j)=a(i,j); }}
    return m;
}

static nc::NdArray<float> fromMat(const agx_emulsion::Matrix& m){
    nc::NdArray<float> a(m.rows, m.cols);
    for(uint32_t i=0;i<m.rows;++i){ for(uint32_t j=0;j<m.cols;++j){ a(i,j)=static_cast<float>(m(i,j)); }}
    return a;
}

// Reshape H x (W*3) to (H*W) x 3
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

// Reshape (H*W) x 3 back to H x (W*3)
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

// Blocked dot: A is H x (W*K), B is K x 3 -> returns H x (W*3)
static nc::NdArray<float> dot_blocks_K3(const nc::NdArray<float>& A, const nc::NdArray<float>& B, int W){
    const int H = static_cast<int>(A.shape().rows);
    const int K = static_cast<int>(B.shape().rows);
    nc::NdArray<float> out(H, W*3);
    for (int w = 0; w < W; ++w){
        // compute out block (H x 3) = (H x K) dot (K x 3)
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

nc::NdArray<float> Process::run(const nc::NdArray<float>& image_in) {
    // Load profiles
    std::string root = std::string(AGX_SOURCE_DIR) + "/cpp/data/profiles/";
    Profile neg = ProfileIO::load_from_file(root + params_.profiles.negative + ".json");
    Profile paper = ProfileIO::load_from_file(root + params_.profiles.print_paper + ".json");

    // Auto exposure
    float ev = 0.0f;
    if (params_.camera.auto_exposure) {
        ev = agx::utils::measure_autoexposure_ev_auto(image_in, params_.io.input_cctf_decoding, params_.camera.auto_exposure_method);
        ev += params_.camera.exposure_compensation_ev;
    } else {
        ev = params_.camera.exposure_compensation_ev;
    }

    // Resize and pixel size
    float film_format_mm = params_.camera.film_format_mm;
    float pixel_size_um = film_format_mm * 1000.0f / static_cast<float>(std::max(image_in.shape().rows, image_in.shape().cols));
    nc::NdArray<float> image = image_in;
    std::cout << "Input image shape: " << image.shape().rows << "x" << image.shape().cols << "\n";
    const int Himg = static_cast<int>(image.shape().rows);
    const int Wimg = static_cast<int>(image.shape().cols) / 3;
    float prf = params_.io.preview_resize_factor * params_.io.upscale_factor;
    if (prf != 1.0f) {
        int newH = static_cast<int>(std::round(image.shape().rows * prf));
        int newW = static_cast<int>(std::round(image.shape().cols * prf));
        image = agx::utils::resize_image_bilinear_auto(image, newH, newW);
        pixel_size_um /= prf;
        std::cout << "Resized image shape: " << image.shape().rows << "x" << image.shape().cols << "\n";
    }

    // Apply band-pass filter to sensitivity
    nc::NdArray<float> sensitivity(neg.data.log_sensitivity.shape().rows, neg.data.log_sensitivity.shape().cols);
    for (uint32_t i=0;i<neg.data.log_sensitivity.shape().rows;++i){
        for (uint32_t j=0;j<neg.data.log_sensitivity.shape().cols;++j){
            sensitivity(i,j) = std::pow(10.0f, neg.data.log_sensitivity(i,j));
        }
    }
    sensitivity = nc::nan_to_num(sensitivity);
    std::cout << "Neg sensitivity shape: " << sensitivity.shape().rows << "x" << sensitivity.shape().cols << "\n";
    if (params_.camera.filter_uv_amp>0 || params_.camera.filter_ir_amp>0) {
        auto bpf = agx::model::compute_band_pass_filter({params_.camera.filter_uv_amp, params_.camera.filter_uv_wl, params_.camera.filter_uv_width},
                                                         {params_.camera.filter_ir_amp, params_.camera.filter_ir_wl, params_.camera.filter_ir_width});
        for (uint32_t i=0;i<sensitivity.shape().rows && i<bpf.size();++i){
            for (int c=0;c<3;++c) sensitivity(i,c) *= bpf[i];
        }
    }

    // RGB->RAW (Hanatos 2025) and exposure
    auto lut = agx::utils::load_hanatos_spectra_lut_npy(std::string(AGX_SOURCE_DIR) + "/cpp/data/luts/spectral_upsampling/irradiance_xy_tc.npy");
    std::cout << "Spectra LUT shape: " << lut.shape().rows << "x" << lut.shape().cols << "\n";
    // Reshape to (H*W) x 3 for per-pixel processing
    auto image_hw_by3 = hw3_to_hw_by3(image);
    nc::NdArray<float> raw_hw_by3 = agx::utils::rgb_to_raw_hanatos2025(image_hw_by3, sensitivity, params_.io.input_color_space, params_.io.input_cctf_decoding, neg.info.reference_illuminant, lut);
    std::cout << "Raw (film) shape (HWx3): " << raw_hw_by3.shape().rows << "x" << raw_hw_by3.shape().cols << "\n";
    // Back to H x (W*3)
    nc::NdArray<float> raw = hw_by3_to_hw3(raw_hw_by3, Himg, Wimg);
    raw *= std::pow(2.0f, ev);

    // Lens blur (Gaussian)
    if (params_.camera.lens_blur_um > 0.0f) {
        std::vector<float> raw_vec(raw.size()); std::copy(raw.begin(), raw.end(), raw_vec.begin());
        std::vector<float> out(raw.size());
        agx_emulsion::Diffusion::apply_gaussian_blur_um(raw_vec, image.shape().rows, image.shape().cols, params_.camera.lens_blur_um, pixel_size_um, out);
        std::copy(out.begin(), out.end(), raw.begin());
    }

    // Develop film (log_raw -> density_cmy)
    // Develop film per pixel: reshape to (H*W) x 3
    nc::NdArray<float> log_raw_nc = nc::log10(raw + 1e-10f);
    auto log_raw = toMat(hw3_to_hw_by3(log_raw_nc));
    auto dc_neg = toMat(neg.data.density_curves);
    std::array<double,3> gamma = {1.0, 1.0, 1.0};
    std::vector<double> le_neg(neg.data.log_exposure.size());
    for (uint32_t i=0;i<neg.data.log_exposure.size();++i){ le_neg[i] = neg.data.log_exposure[i]; }
    auto density_cmy_mat = agx_emulsion::interpolate_exposure_to_density(log_raw, dc_neg, le_neg, gamma);
    auto density_cmy = hw_by3_to_hw3(fromMat(density_cmy_mat), Himg, Wimg);
    std::cout << "Density CMY shape: " << density_cmy.shape().rows << "x" << density_cmy.shape().cols << "\n";

    // Enlarge and print
    auto light_src_K = agx::model::standard_illuminant(params_.enlarger.illuminant); // already 1xK
    const int Kspec = static_cast<int>(agx::config::SPECTRAL_SHAPE.wavelengths.size());
    // Load neutral Y/M/C filter settings from DB to match Python
    float y_neutral = params_.enlarger.y_filter_neutral;
    float m_neutral = params_.enlarger.m_filter_neutral;
    float c_neutral = params_.enlarger.c_filter_neutral;
    try {
        auto ymc = agx::utils::read_neutral_ymc_filter_values();
        // Access [print_paper][illuminant][negative] -> [y, m, c]
        auto arr = ymc[paper.info.stock][params_.enlarger.illuminant][neg.info.stock];
        if (arr.is_array() && arr.size() >= 3) {
            y_neutral = arr[0].get<float>();
            m_neutral = arr[1].get<float>();
            c_neutral = arr[2].get<float>();
        }
    } catch (...) {
        // Fallback to params if DB not found
    }
    auto enlarger_ill = agx::model::color_enlarger(light_src_K,
        y_neutral * agx::config::ENLARGER_STEPS + params_.enlarger.y_filter_shift,
        m_neutral * agx::config::ENLARGER_STEPS + params_.enlarger.m_filter_shift,
        c_neutral * agx::config::ENLARGER_STEPS);

    // Spectral density from film density
    auto density_spec = agx::utils::compute_density_spectral(density_cmy, neg.data.dye_density, /*min factor*/ neg.data.dye_density_min_factor);
    std::cout << "Density spectral shape: " << density_spec.shape().rows << "x" << density_spec.shape().cols << "\n";
    auto light = agx::utils::density_to_light(density_spec, enlarger_ill);
    std::cout << "Light (enlarger) shape: " << light.shape().rows << "x" << light.shape().cols << "\n";
    nc::NdArray<float> paper_sens(paper.data.log_sensitivity.shape().rows, paper.data.log_sensitivity.shape().cols);
    for (uint32_t i=0;i<paper.data.log_sensitivity.shape().rows;++i){
        for (uint32_t j=0;j<paper.data.log_sensitivity.shape().cols;++j){
            paper_sens(i,j) = std::pow(10.0f, paper.data.log_sensitivity(i,j));
        }
    }
    paper_sens = nc::nan_to_num(paper_sens);
    // Ensure paper_sens is shape (K,3) and light has H x (W*K)
    if (!(paper_sens.shape().cols == 3 && (light.shape().cols % paper_sens.shape().rows == 0))) {
        throw std::runtime_error("paper_sens shape mismatch");
    }
    auto cmy = dot_blocks_K3(light, paper_sens, Wimg);
    std::cout << "CMY (pre paper) shape: " << cmy.shape().rows << "x" << cmy.shape().cols << "\n";
    cmy *= params_.enlarger.print_exposure;

    // Midgray normalization for print exposure (match Python):
    // rgb_midgray = [0.184,0.184,0.184] * 2**neg_exp_comp_ev if print_exposure_compensation
    float neg_exp_comp_ev = params_.enlarger.print_exposure_compensation ? params_.camera.exposure_compensation_ev : 0.0f;
    float mid_scale = std::pow(2.0f, neg_exp_comp_ev);
    nc::NdArray<float> rgb_mid(1,3); rgb_mid(0,0)=0.184f*mid_scale; rgb_mid(0,1)=0.184f*mid_scale; rgb_mid(0,2)=0.184f*mid_scale;
    auto raw_mid = agx::utils::rgb_to_raw_hanatos2025(rgb_mid, sensitivity, "sRGB", false, neg.info.reference_illuminant, lut);
    auto log_raw_mid_nc = nc::log10(raw_mid + 1e-10f);
    auto log_raw_mid_mat = toMat(log_raw_mid_nc);
    auto density_cmy_mid_mat = agx_emulsion::interpolate_exposure_to_density(log_raw_mid_mat, dc_neg, le_neg, gamma);
    auto density_cmy_mid = fromMat(density_cmy_mid_mat);
    auto density_spec_mid = agx::utils::compute_density_spectral(density_cmy_mid, neg.data.dye_density, /*min factor*/ neg.data.dye_density_min_factor);
    auto light_mid = agx::utils::density_to_light(density_spec_mid, enlarger_ill);
    auto raw_mid_print = nc::dot(light_mid, paper_sens);
    std::cout << "Midgray raw print: " << raw_mid_print(0,0) << ", " << raw_mid_print(0,1) << ", " << raw_mid_print(0,2) << "\n";
    float mid_g = raw_mid_print(0,1);
    if (!std::isfinite(mid_g)) mid_g = 1e-10f;
    float factor = 1.0f / std::max(1e-10f, mid_g);
    cmy *= factor;

    // Preflash if any (approximate)
    if (params_.enlarger.preflash_exposure > 0.0f) {
        auto preflash_ill = agx::model::color_enlarger(light_src_K,
            y_neutral * agx::config::ENLARGER_STEPS + params_.enlarger.preflash_y_filter_shift,
            m_neutral * agx::config::ENLARGER_STEPS + params_.enlarger.preflash_m_filter_shift,
            c_neutral * agx::config::ENLARGER_STEPS);
        nc::NdArray<float> base_den(1, neg.data.dye_density.shape().rows);
        for (uint32_t k=0;k<neg.data.dye_density.shape().rows;++k){ base_den(0,k)=neg.data.dye_density(k,3); }
        auto light_pre = agx::utils::density_to_light(base_den, preflash_ill);
        auto raw_pre = nc::dot(light_pre, paper_sens); // 1x3
        // Broadcast add across W blocks
        for (int i=0;i<cmy.shape().rows;++i){
            for (int w=0; w<Wimg; ++w){
                for (int c=0;c<3;++c){
                    cmy(i, w*3 + c) += raw_pre(0,c) * params_.enlarger.preflash_exposure;
                }
            }
        }
    }

    // Develop print: log10 and apply density curves of paper
    auto log_cmy_nc = nc::log10(cmy + 1e-10f);
    auto log_cmy_mat = toMat(hw3_to_hw_by3(log_cmy_nc));
    auto dc_paper = toMat(paper.data.density_curves);
    std::vector<double> le_paper(paper.data.log_exposure.size());
    for (uint32_t i=0;i<paper.data.log_exposure.size();++i){ le_paper[i]=paper.data.log_exposure[i]; }
    auto density_print_mat = agx_emulsion::interpolate_exposure_to_density(log_cmy_mat, dc_paper, le_paper, gamma);
    auto density_print = hw_by3_to_hw3(fromMat(density_print_mat), Himg, Wimg);
    std::cout << "Density print shape: " << density_print.shape().rows << "x" << density_print.shape().cols << "\n";

    // Scan to RGB (XYZ)
    std::cout << "Paper dye_density shape: " << paper.data.dye_density.shape().rows << "x" << paper.data.dye_density.shape().cols << "\n";
    auto scan_ill = agx::model::standard_illuminant(paper.info.viewing_illuminant); // 1xK
    std::cout << "Scan illuminant shape: " << scan_ill.shape().rows << "x" << scan_ill.shape().cols << "\n";
    float norm = 0.0f; for(uint32_t i=0;i<agx::config::SPECTRAL_SHAPE.wavelengths.size();++i) norm += agx::config::STANDARD_OBSERVER_CMFS(i,1) * scan_ill[i];
    std::cout << "Computing density spectral for scan...\n";
    auto dens_spec_scan = agx::utils::compute_density_spectral(density_print, paper.data.dye_density, /*min factor*/ 1.0f);
    std::cout << "dens_spec_scan shape: " << dens_spec_scan.shape().rows << "x" << dens_spec_scan.shape().cols << "\n";
    auto light_scan = agx::utils::density_to_light(dens_spec_scan, scan_ill);
    std::cout << "Light (scan) shape: " << light_scan.shape().rows << "x" << light_scan.shape().cols << "\n";
    auto xyz_hw3 = dot_blocks_K3(light_scan, agx::config::STANDARD_OBSERVER_CMFS, Wimg) / norm;
    xyz_hw3 = nc::nan_to_num(xyz_hw3);
    std::cout << "XYZ (H x W*3) shape: " << xyz_hw3.shape().rows << "x" << xyz_hw3.shape().cols << "\n";

    // Convert XYZ -> output RGB, compute viewing illuminant xy from scan SPD to match Python
    nc::NdArray<float> illuminant_xyz(1,3);
    illuminant_xyz(0,0)=0.0f; illuminant_xyz(0,1)=0.0f; illuminant_xyz(0,2)=0.0f;
    auto scan_flat = scan_ill.flatten();
    for (uint32_t i=0;i<agx::config::SPECTRAL_SHAPE.wavelengths.size();++i) {
        float e = scan_flat[i];
        illuminant_xyz(0,0) += e * agx::config::STANDARD_OBSERVER_CMFS(i,0);
        illuminant_xyz(0,1) += e * agx::config::STANDARD_OBSERVER_CMFS(i,1);
        illuminant_xyz(0,2) += e * agx::config::STANDARD_OBSERVER_CMFS(i,2);
    }
    illuminant_xyz /= norm;
    nc::NdArray<float> illuminant_xy(1,2);
    float Ssum = illuminant_xyz(0,0) + illuminant_xyz(0,1) + illuminant_xyz(0,2);
    if (Ssum > 0.0f) { illuminant_xy(0,0) = illuminant_xyz(0,0)/Ssum; illuminant_xy(0,1) = illuminant_xyz(0,1)/Ssum; }
    else { illuminant_xy(0,0)=0.3127f; illuminant_xy(0,1)=0.3290f; }
    auto xyz_hw_by3 = hw3_to_hw_by3(xyz_hw3);
    auto rgb_hw_by3 = colour::XYZ_to_RGB(xyz_hw_by3, params_.io.output_color_space, params_.io.output_cctf_encoding, illuminant_xy, "CAT02");
    auto rgb_hw3 = hw_by3_to_hw3(rgb_hw_by3, Himg, Wimg);
    std::cout << "Output RGB shape: " << rgb_hw3.shape().rows << "x" << rgb_hw3.shape().cols << "\n";
    return rgb_hw3;
}

} } // namespace agx::process
