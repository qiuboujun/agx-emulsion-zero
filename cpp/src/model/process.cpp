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
#include "couplers.hpp"
#include "grain.hpp"
#include "colour.hpp"
#include "conversions.hpp"
#include "io.hpp"
#include "lut.hpp"

using agx::profiles::Profile;
using agx::profiles::ProfileIO;

namespace agx { namespace process {

Process::Process(const Params& params) : params_(params) {}

static inline int reflect_index_1d(int idx, int N) {
    if (N <= 1) return 0;
    while (idx < 0 || idx >= N) {
        if (idx < 0) idx = -idx - 1;
        else if (idx >= N) idx = 2 * N - idx - 1;
    }
    return idx;
}

static float interp_scalar(float x, const nc::NdArray<float>& xp_col, const nc::NdArray<float>& fp_col) {
    // xp_col: N x 1 strictly monotonic increasing in typical profiles
    auto xp = xp_col.flatten();
    auto fp = fp_col.flatten();
    const int N = static_cast<int>(xp.size());
    if (N == 0) return 0.0f;
    if (x <= xp[0]) return fp[0];
    if (x >= xp[N-1]) return fp[N-1];
    // binary search for interval
    int lo = 0, hi = N - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (x >= xp[mid]) lo = mid; else hi = mid;
    }
    float t = (x - xp[lo]) / (xp[hi] - xp[lo]);
    return fp[lo] * (1.0f - t) + fp[hi] * t;
}

static nc::NdArray<float> interp_array(const nc::NdArray<float>& xs_col,
                                       const nc::NdArray<float>& xp_col,
                                       const nc::NdArray<float>& fp_col) {
    const int N = static_cast<int>(xs_col.size());
    nc::NdArray<float> out(1, N);
    for (int i=0;i<N;++i) out[i] = interp_scalar(xs_col[i], xp_col, fp_col);
    return out.transpose();
}

static void gaussian_blur_1d_reflect(nc::NdArray<float>& x, double sigma) {
    if (sigma <= 0.0) return;
    const int N = static_cast<int>(x.size());
    if (N <= 1) return;
    int radius = static_cast<int>(std::ceil(3.0 * sigma));
    if (radius < 1) radius = 1;
    const int K = 2 * radius + 1;
    std::vector<double> w(K);
    double sumw = 0.0;
    for (int k=-radius; k<=radius; ++k) {
        double g = std::exp(-0.5 * (k*k) / (sigma*sigma));
        w[k + radius] = g;
        sumw += g;
    }
    for (int i=0;i<K;++i) w[i] /= sumw;
    nc::NdArray<float> y(1, N);
    for (int i=0;i<N;++i) {
        double acc = 0.0;
        for (int k=-radius; k<=radius; ++k) {
            int j = reflect_index_1d(i + k, N);
            acc += w[k + radius] * static_cast<double>(x[j]);
        }
        y[i] = static_cast<float>(acc);
    }
    // copy back
    for (int i=0;i<N;++i) x[i] = y[i];
}

static nc::NdArray<float> remove_viewing_glare_comp(const nc::NdArray<float>& le_col,
                                                    const nc::NdArray<float>& dc, // N x 3
                                                    float factor,
                                                    float density,
                                                    float transition) {
    if (factor <= 0.0f) return dc;
    const int N = static_cast<int>(dc.shape().rows);
    if (N == 0) return dc;
    // mean density across channels
    nc::NdArray<float> dc_mean(1, N);
    for (int i=0;i<N;++i) {
        double m = 0.0; for (int c=0;c<3;++c) m += dc(i,c);
        dc_mean[i] = static_cast<float>(m / 3.0);
    }
    dc_mean = dc_mean.transpose(); // N x 1
    // center le where mean density equals given density
    float le_center = interp_scalar(density, dc_mean, le_col);
    // slope around le_center measured over +/- 0.5 EV in log10 domain
    double le_delta = std::log10(std::pow(2.0, 1.0)) / 2.0; // log10(2)/2
    float le0 = le_center - static_cast<float>(le_delta);
    float le1 = le_center + static_cast<float>(le_delta);
    float d0 = interp_scalar(le0, le_col, dc_mean);
    float d1 = interp_scalar(le1, le_col, dc_mean);
    float slope = (d1 - d0) / (le1 - le0);
    if (!std::isfinite(slope) || std::abs(slope) < 1e-12f) slope = 1.0f;
    // le step
    double le_step_acc = 0.0;
    auto le_flat = le_col.flatten();
    for (int i=1;i<N;++i) le_step_acc += static_cast<double>(le_flat[i] - le_flat[i-1]);
    double le_step = le_step_acc / std::max(1, N-1);
    if (le_step <= 0.0) le_step = 1.0;
    // Build le_nl and blur
    nc::NdArray<float> le_nl = le_col.copy();
    for (int i=0;i<N;++i) {
        float le_i = le_flat[i];
        if (le_i > le_center) le_nl[i] = le_i - (le_i - le_center) * factor; else le_nl[i] = le_i;
    }
    double le_transition = static_cast<double>(transition) / static_cast<double>(slope);
    double sigma = le_transition / le_step;
    gaussian_blur_1d_reflect(le_nl, sigma);
    // remap each channel via interpolation
    nc::NdArray<float> out(N, 3);
    for (int c=0;c<3;++c) {
        auto fp = dc(nc::Slice(), nc::Slice(c, c+1)).copy();
        auto yi = interp_array(le_nl, le_col, fp);
        for (int i=0;i<N;++i) out(i,c) = yi(i,0);
    }
    return out;
}

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
                double acc = 0.0;
                for (int k = 0; k < K; ++k){
                    acc += static_cast<double>(A(i, w*K + k)) * static_cast<double>(B(k, c));
                }
                out(i, w*3 + c) = static_cast<float>(acc);
            }
        }
    }
    return out;
}

nc::NdArray<float> Process::run(const nc::NdArray<float>& image_in) {
    // Apply debug switches (subset parity with Python)
    if (params_.debug.deactivate_spatial_effects) {
        // Lens/enlarger/scanner blur and unsharp
        params_.camera.lens_blur_um = 0.0f;
        params_.enlarger.lens_blur = 0.0f;
        params_.scanner.lens_blur = 0.0f;
        params_.scanner.unsharp_sigma = 0.0f;
        params_.scanner.unsharp_amount = 0.0f;
    }
    // Load profiles (resolve via runtime data path with repo fallback)
    std::string root = agx::utils::get_data_path() + "agx_emulsion/data/profiles/";
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
    auto lut = agx::utils::load_hanatos_spectra_lut_npy(agx::utils::get_data_path() + "agx_emulsion/data/luts/spectral_upsampling/irradiance_xy_tc.npy");
    std::cout << "Spectra LUT shape: " << lut.shape().rows << "x" << lut.shape().cols << "\n";
    // Reshape to (H*W) x 3 for per-pixel processing
    auto image_hw_by3 = hw3_to_hw_by3(image);
    nc::NdArray<float> raw_hw_by3;
    if (params_.settings.use_camera_lut) {
        // Build a 3D LUT over input RGB -> raw using our spectral upsampling model, then apply
        auto [raw_lut_out, camera_lut] = agx::utils::compute_camera_with_lut(
            image_hw_by3,
            Himg,
            Wimg,
            sensitivity,
            params_.io.input_color_space,
            params_.io.input_cctf_decoding,
            neg.info.reference_illuminant,
            lut,
            params_.settings.lut_resolution);
        raw_hw_by3 = raw_lut_out;
        (void)camera_lut; // can be stored for debug if needed
    } else {
        raw_hw_by3 = agx::utils::rgb_to_raw_hanatos2025(image_hw_by3, sensitivity, params_.io.input_color_space, params_.io.input_cctf_decoding, neg.info.reference_illuminant, lut);
    }
    std::cout << "Raw (film) shape (HWx3): " << raw_hw_by3.shape().rows << "x" << raw_hw_by3.shape().cols << "\n";
    // Back to H x (W*3)
    nc::NdArray<float> raw = hw_by3_to_hw3(raw_hw_by3, Himg, Wimg);
    raw *= std::pow(2.0f, ev);

    // Lens blur (Gaussian) and halation/scattering
    std::cout << "Lens blur (um): " << params_.camera.lens_blur_um
              << ", Halation active: " << (params_.io.full_image && neg.halation.active ? 1 : 0)
              << ", pixel_size_um: " << pixel_size_um << "\n";
    if (params_.camera.lens_blur_um > 0.0f) {
        std::vector<float> raw_vec(raw.size()); std::copy(raw.begin(), raw.end(), raw_vec.begin());
        std::vector<float> out(raw.size());
        agx_emulsion::Diffusion::apply_gaussian_blur_um(raw_vec, image.shape().rows, image.shape().cols, params_.camera.lens_blur_um, pixel_size_um, out);
        std::copy(out.begin(), out.end(), raw.begin());
    }
    // Apply halation/scattering if active in profile and full image mode (match Python debug switches)
    if (params_.io.full_image && neg.halation.active) {
        std::vector<float> raw_vec(raw.size()); std::copy(raw.begin(), raw.end(), raw_vec.begin());
        agx_emulsion::HalationParams hp;
        hp.active = true;
        hp.size_um = neg.halation.size_um;
        hp.strength = neg.halation.strength;
        hp.scattering_size_um = neg.halation.scattering_size_um;
        hp.scattering_strength = neg.halation.scattering_strength;
        agx_emulsion::Diffusion::apply_halation_um(raw_vec, image.shape().rows, image.shape().cols, hp, pixel_size_um);
        std::copy(raw_vec.begin(), raw_vec.end(), raw.begin());
    }

    // Develop film (log_raw -> density_cmy)
    std::cout << "Develop film: interpolate exposure->density (neg)" << "\n";
    // Compute log10 in double precision and reshape to (H*W) x 3
    auto raw_hw_by3_for_log = hw3_to_hw_by3(raw);
    agx_emulsion::Matrix log_raw(raw_hw_by3_for_log.shape().rows, 3);
    for (uint32_t i=0;i<raw_hw_by3_for_log.shape().rows;++i){
        for (uint32_t j=0;j<3;++j){
            double v = static_cast<double>(raw_hw_by3_for_log(i,j));
            if (v < 0.0) v = 0.0; // match np.fmax(raw, 0.0)
            log_raw(i,j) = std::log10(v + 1e-10);
        }
    }
    auto dc_neg = toMat(neg.data.density_curves);
    std::array<double,3> gamma = {static_cast<double>(neg.data.gamma_factor[0]), static_cast<double>(neg.data.gamma_factor[1]), static_cast<double>(neg.data.gamma_factor[2])};
    std::vector<double> le_neg(neg.data.log_exposure.size());
    for (uint32_t i=0;i<neg.data.log_exposure.size();++i){ le_neg[i] = neg.data.log_exposure[i]; }
    agx_emulsion::Matrix density_cmy_mat;
    {
        std::cout << "  P (pixels): " << log_raw.rows << ", N (curve points): " << le_neg.size() << "\n";
        auto t0 = std::chrono::steady_clock::now();
        bool gpu_ok = false;
        try {
            gpu_ok = agx_emulsion::gpu_interpolate_exposure_to_density(log_raw, dc_neg, le_neg, gamma, density_cmy_mat);
        } catch (const std::exception& e) {
            std::cout << "  GPU interpolate threw: " << e.what() << "\n";
            gpu_ok = false;
        } catch (...) {
            std::cout << "  GPU interpolate threw unknown exception" << "\n";
            gpu_ok = false;
        }
        auto t1 = std::chrono::steady_clock::now();
        auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count();
        std::cout << "  GPU interpolate done: ok=" << (gpu_ok?1:0) << ", time=" << ms << " ms" << "\n";
        if (!gpu_ok) {
            auto c0 = std::chrono::steady_clock::now();
            density_cmy_mat = agx_emulsion::interpolate_exposure_to_density(log_raw, dc_neg, le_neg, gamma);
            auto c1 = std::chrono::steady_clock::now();
            auto cms = std::chrono::duration_cast<std::chrono::milliseconds>(c1 - c0).count();
            std::cout << "  CPU interpolate time=" << cms << " ms" << "\n";
        }
    }
    // DIR couplers: compute corrected curves and corrected log_raw, then re-interpolate (Python parity)
    // Only apply if profile has dir_couplers active
    if (true) {
        try {
            auto j = agx::profiles::parse_json_with_specials(root + params_.profiles.negative + ".json");
            if (j.contains("dir_couplers") && j["dir_couplers"].contains("active") && j["dir_couplers"]["active"].get<bool>()) {
                auto dc_cpp = neg.data.density_curves;
                std::vector<std::vector<double>> dc_vec(dc_cpp.shape().rows, std::vector<double>(3));
                for (uint32_t i=0;i<dc_cpp.shape().rows;++i){ for(int c=0;c<3;++c) dc_vec[i][c]=dc_cpp(i,c); }
                std::vector<double> le_vec(neg.data.log_exposure.size()); for(uint32_t i=0;i<le_vec.size();++i) le_vec[i]=neg.data.log_exposure[i];
                std::array<double,3> amount_rgb{1.0,1.0,1.0};
                double layer_diffusion = 1.0;
                double diffusion_size_um = 0.0;
                double high_shift = 0.0;
                if (j["dir_couplers"].contains("amount")) {
                    double a = j["dir_couplers"]["amount"].get<double>();
                    std::array<double,3> ratio{1.0,1.0,1.0};
                    if (j["dir_couplers"].contains("ratio_rgb")) {
                        for (int c=0;c<3;++c) ratio[c] = j["dir_couplers"]["ratio_rgb"][c].get<double>();
                    }
                    for (int c=0;c<3;++c) amount_rgb[c] = a * ratio[c];
                }
                if (j["dir_couplers"].contains("diffusion_interlayer")) layer_diffusion = j["dir_couplers"]["diffusion_interlayer"].get<double>();
                if (j["dir_couplers"].contains("diffusion_size_um")) diffusion_size_um = j["dir_couplers"]["diffusion_size_um"].get<double>();
                if (j["dir_couplers"].contains("high_exposure_shift")) high_shift = j["dir_couplers"]["high_exposure_shift"].get<double>();
                auto M = agx_emulsion::Couplers::compute_dir_couplers_matrix(amount_rgb, layer_diffusion);
                auto dc0 = agx_emulsion::Couplers::compute_density_curves_before_dir_couplers(dc_vec, le_vec, M, high_shift);
                // density_max per channel
                std::array<double,3> dmax{0.0,0.0,0.0};
                for (int c=0;c<3;++c){ double m=-1e30; for(uint32_t i=0;i<dc_cpp.shape().rows;++i){ m = std::max(m, (double)dc_cpp(i,c)); } dmax[c]=m; }
                int diffusion_px = (int)std::round(diffusion_size_um / pixel_size_um);
                // reshape inputs for correction: (H*W) x 3 to H x W x 3 grid
                int H = Himg, W = Wimg;
                std::vector<std::vector<std::array<double,3>>> log_raw_grid(H, std::vector<std::array<double,3>>(W));
                std::vector<std::vector<std::array<double,3>>> dens_grid(H, std::vector<std::array<double,3>>(W));
                for (int i=0;i<H;++i){ for(int w=0; w<W; ++w){ for(int c=0;c<3;++c){
                    log_raw_grid[i][w][c] = log_raw(i*W + w, c);
                    dens_grid[i][w][c]    = density_cmy_mat(i*W + w, c);
                } } }
                auto log_raw_corr = agx_emulsion::Couplers::compute_exposure_correction_dir_couplers(log_raw_grid, dens_grid, dmax, M, diffusion_px, high_shift);
                // flatten back to (H*W) x 3
                for (int i=0;i<H;++i){ for(int w=0; w<W; ++w){ for(int c=0;c<3;++c){
                    log_raw(i*W + w, c) = (double)log_raw_corr[i][w][c];
                } } }
                // interpolate with corrected curves
                agx_emulsion::Matrix dc0_mat(dc_cpp.shape().rows, 3);
                for (uint32_t i=0;i<dc_cpp.shape().rows;++i){ for(int c=0;c<3;++c) dc0_mat(i,c) = (double)dc0[i][c]; }
                agx_emulsion::Matrix density_cmy_mat2;
                if (!agx_emulsion::gpu_interpolate_exposure_to_density(log_raw, dc0_mat, le_neg, gamma, density_cmy_mat2)) {
                    density_cmy_mat2 = agx_emulsion::interpolate_exposure_to_density(log_raw, dc0_mat, le_neg, gamma);
                }
                density_cmy_mat = density_cmy_mat2;
            }
        } catch(...) { /* ignore couplers on error */ }
    }
    auto density_cmy = hw_by3_to_hw3(fromMat(density_cmy_mat), Himg, Wimg);
    // Apply grain only in full-image mode (matches Python behavior). Deterministic by default for tests.
    if (params_.io.full_image) {
        try {
            auto jneg = agx::profiles::parse_json_with_specials(root + params_.profiles.negative + ".json");
            if (jneg.contains("grain") && jneg["grain"].contains("active") && jneg["grain"]["active"].get<bool>()) {
                // Build Image2D from density_cmy
                agx_emulsion::Image2D img(Wimg, Himg, 3);
                for (int i=0;i<Himg;++i){ for(int w=0; w<Wimg; ++w){ for(int c=0;c<3;++c){ img.at(w,i,c) = density_cmy(i, w*3+c); } } }
                float agx_particle_area_um2 = jneg["grain"].contains("agx_particle_area_um2") ? jneg["grain"]["agx_particle_area_um2"].get<float>() : 0.2f;
                std::array<float,3> agx_particle_scale{1.0f,0.8f,3.0f};
                if (jneg["grain"].contains("agx_particle_scale")) for(int c=0;c<3;++c) agx_particle_scale[c] = jneg["grain"]["agx_particle_scale"][c].get<float>();
                std::array<float,3> density_min{0.03f,0.06f,0.04f};
                if (jneg["grain"].contains("density_min")) for(int c=0;c<3;++c) density_min[c] = jneg["grain"]["density_min"][c].get<float>();
                std::array<float,3> density_max_curves{2.2f,2.2f,2.2f};
                // use max from curves if not present
                for (int c=0;c<3;++c){ float m=-1e30f; for(uint32_t i=0;i<neg.data.density_curves.shape().rows;++i) m = std::max(m, neg.data.density_curves(i,c)); density_max_curves[c]=m; }
                std::array<float,3> grain_uniformity{0.98f,0.98f,0.98f};
                if (jneg["grain"].contains("uniformity")) for(int c=0;c<3;++c) grain_uniformity[c] = jneg["grain"]["uniformity"][c].get<float>();
                float grain_blur = jneg["grain"].contains("blur") ? jneg["grain"]["blur"].get<float>() : 0.0f;
                int n_sub_layers = jneg["grain"].contains("n_sub_layers") ? jneg["grain"]["n_sub_layers"].get<int>() : 1;
                auto out_img = agx_emulsion::Grain::apply_grain_to_density(img, pixel_size_um, agx_particle_area_um2, agx_particle_scale, density_min, density_max_curves, grain_uniformity, grain_blur, n_sub_layers, /*fixed_seed*/true);
                for (int i=0;i<Himg;++i){ for(int w=0; w<Wimg; ++w){ for(int c=0;c<3;++c){ density_cmy(i, w*3+c) = out_img.at(w,i,c); } } }
            }
        } catch (...) { /* ignore grain on error */ }
    }
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

    // Apply masking couplers to spectral dye densities if configured (erf model, effectiveness=1.0)
    if (params_.settings.apply_masking_couplers && neg.masking_couplers.active) {
        auto wl = neg.data.wavelengths.flatten();
        const double effectiveness = 1.0;
        const double denom = 2.0 + effectiveness; // Python: (erf + 1 + eff) / (2 + eff)
        for (uint32_t k=0;k<neg.data.dye_density.shape().rows;++k){
            double w = wl[k];
            for (int c=0;c<3;++c){
                double z = (w - (double)neg.masking_couplers.cross_over_points[c]) / (double)neg.masking_couplers.transition_widths[c];
                double s = (std::erf(z) + 1.0 + effectiveness) / denom;
                neg.data.dye_density(k, c) = (float)(neg.data.dye_density(k, c) * s);
            }
        }
    }

    // ---------------------------------------------------------------------------------
    // Enlarge and compute print log exposure either directly or via LUT
    // ---------------------------------------------------------------------------------
    agx_emulsion::Matrix log_cmy_mat; // (H*W) x 3
    if (params_.settings.use_enlarger_lut) {
        std::cout << "Enlarger path: using LUT, resolution=" << params_.settings.lut_resolution << "\n";
        // Precompute paper sensitivity and midgray factor and preflash raw once
        nc::NdArray<float> paper_sens(paper.data.log_sensitivity.shape().rows, paper.data.log_sensitivity.shape().cols);
        for (uint32_t i=0;i<paper.data.log_sensitivity.shape().rows;++i){
            for (uint32_t j=0;j<paper.data.log_sensitivity.shape().cols;++j){
                paper_sens(i,j) = std::pow(10.0f, paper.data.log_sensitivity(i,j));
            }
        }
        paper_sens = nc::nan_to_num(paper_sens);

        // Midgray normalization
        float neg_exp_comp_ev = params_.enlarger.print_exposure_compensation ? params_.camera.exposure_compensation_ev : 0.0f;
        float mid_scale = std::pow(2.0f, neg_exp_comp_ev);
        nc::NdArray<float> rgb_mid(1,3); rgb_mid(0,0)=0.184f*mid_scale; rgb_mid(0,1)=0.184f*mid_scale; rgb_mid(0,2)=0.184f*mid_scale;
        auto raw_mid = agx::utils::rgb_to_raw_hanatos2025(rgb_mid, sensitivity, "sRGB", false, neg.info.reference_illuminant, lut);
        agx_emulsion::Matrix log_raw_mid_mat(1,3);
        for (int j=0;j<3;++j){ double v = static_cast<double>(raw_mid(0,j)); if (v<0.0) v=0.0; log_raw_mid_mat(0,j)=std::log10(v+1e-10); }
        agx_emulsion::Matrix density_cmy_mid_mat;
        if (!agx_emulsion::gpu_interpolate_exposure_to_density(log_raw_mid_mat, dc_neg, le_neg, gamma, density_cmy_mid_mat)) {
            density_cmy_mid_mat = agx_emulsion::interpolate_exposure_to_density(log_raw_mid_mat, dc_neg, le_neg, gamma);
        }
        auto density_cmy_mid = fromMat(density_cmy_mid_mat);
        nc::NdArray<float> density_spec_mid;
        if (!agx::utils::compute_density_spectral_gpu(density_cmy_mid, neg.data.dye_density, /*min factor*/ neg.data.dye_density_min_factor, density_spec_mid)) {
            density_spec_mid = agx::utils::compute_density_spectral(density_cmy_mid, neg.data.dye_density, /*min factor*/ neg.data.dye_density_min_factor);
        }
        nc::NdArray<float> light_mid;
        if (!agx::utils::density_to_light_gpu(density_spec_mid, enlarger_ill, light_mid)) {
            light_mid = agx::utils::density_to_light(density_spec_mid, enlarger_ill);
        }
        auto raw_mid_print = nc::dot(light_mid, paper_sens);
        float mid_g = raw_mid_print(0,1);
        if (!std::isfinite(mid_g)) mid_g = 1e-10f;
        float exposure_factor = 1.0f / std::max(1e-10f, mid_g);

        // Precompute preflash raw
        nc::NdArray<float> raw_pre(1,3); raw_pre.fill(0.0f);
        if (params_.enlarger.preflash_exposure > 0.0f) {
            auto preflash_ill = agx::model::color_enlarger(light_src_K,
                y_neutral * agx::config::ENLARGER_STEPS + params_.enlarger.preflash_y_filter_shift,
                m_neutral * agx::config::ENLARGER_STEPS + params_.enlarger.preflash_m_filter_shift,
                c_neutral * agx::config::ENLARGER_STEPS);
            nc::NdArray<float> base_den(1, neg.data.dye_density.shape().rows);
            for (uint32_t k=0;k<neg.data.dye_density.shape().rows;++k){ base_den(0,k)=neg.data.dye_density(k,3); }
            auto light_pre = agx::utils::density_to_light(base_den, preflash_ill);
            raw_pre = nc::dot(light_pre, paper_sens); // 1x3
            for (int c=0;c<3;++c) raw_pre(0,c) *= params_.enlarger.preflash_exposure;
        }

        // Compute normalization for film densities: (d + dmin)/dmax
        std::array<float,3> density_min{0.03f,0.06f,0.04f};
        try {
            auto jneg = agx::profiles::parse_json_with_specials(root + params_.profiles.negative + ".json");
            if (jneg.contains("grain") && jneg["grain"].contains("density_min")) {
                for (int c=0;c<3;++c) density_min[c] = jneg["grain"]["density_min"][c].get<float>();
            }
        } catch (...) {}
        std::array<float,3> density_max_curves{0.f,0.f,0.f};
        for (int c=0;c<3;++c){ float m=-1e30f; for(uint32_t i=0;i<neg.data.density_curves.shape().rows;++i) m = std::max(m, neg.data.density_curves(i,c)); density_max_curves[c]=m + density_min[c]; }

        // Build function mapping normalized CMY -> log_raw (per-pixel)
        auto spectral_fn = [&](const nc::NdArray<float>& dens_n)->nc::NdArray<float>{
            const int N = (int)dens_n.shape().rows;
            // Denormalize
            nc::NdArray<float> dens(N,3);
            for (int i=0;i<N;++i){
                for (int c=0;c<3;++c){
                    float dn = dens_n(i,c);
                    float d = dn * density_max_curves[c] - density_min[c];
                    dens(i,c) = d;
                }
            }
            // Compute spectral density and light
            nc::NdArray<float> dens_spec;
            if (!agx::utils::compute_density_spectral_gpu(dens, neg.data.dye_density, neg.data.dye_density_min_factor, dens_spec)) {
                dens_spec = agx::utils::compute_density_spectral(dens, neg.data.dye_density, neg.data.dye_density_min_factor);
            }
            auto light_loc = agx::utils::density_to_light(dens_spec, enlarger_ill);
            auto raw_loc = nc::dot(light_loc, paper_sens);
            // Apply exposure scaling
            if (!params_.enlarger.just_preflash) {
                for (int i=0;i<N;++i){ for(int c=0;c<3;++c){ raw_loc(i,c) *= (params_.enlarger.print_exposure * exposure_factor); } }
            } else {
                for (int i=0;i<N;++i){ for(int c=0;c<3;++c){ raw_loc(i,c) = 0.0f; } }
            }
            // Add preflash
            for (int i=0;i<N;++i){ for(int c=0;c<3;++c){ raw_loc(i,c) += raw_pre(0,c); } }
            // Log10
            nc::NdArray<float> out(N,3);
            for (int i=0;i<N;++i){ for(int c=0;c<3;++c){ double v = std::max(0.0, (double)raw_loc(i,c)); out(i,c) = (float)std::log10(v + 1e-10); } }
            return out;
        };

        // Normalize current density_cmy
        nc::NdArray<float> dnorm = density_cmy.copy();
        for (int i=0;i<Himg;++i){ for (int w=0; w<Wimg; ++w){
            for (int c=0;c<3;++c){ float d = dnorm(i, w*3 + c); dnorm(i, w*3 + c) = (d + density_min[c]) / std::max(1e-10f, density_max_curves[c]); }
        }}

        // Evaluate via LUT
        auto dnorm_flat = hw3_to_hw_by3(dnorm);
        std::cout << "  compute_with_lut (enlarger) start" << std::endl;
        auto [log_raw_flat, lut3d] = agx::utils::compute_with_lut(dnorm_flat, spectral_fn, 0.0f, 1.0f, params_.settings.lut_resolution, Himg, Wimg);
        std::cout << "  compute_with_lut (enlarger) done" << std::endl;
        
        // Check if LUT computation failed
        if (log_raw_flat.shape().rows == 0 || log_raw_flat.shape().cols == 0) {
            std::cout << "  ERROR: LUT computation failed, falling back to direct spectral path" << std::endl;
            // Fallback to direct spectral computation
            auto density_spec = agx::utils::compute_density_spectral(dnorm, neg.data.dye_density, neg.data.dye_density_min_factor);
            auto light = agx::utils::density_to_light(density_spec, enlarger_ill);
            auto raw = nc::dot(light, paper_sens);
            if (!params_.enlarger.just_preflash) {
                for (int i=0;i<raw.shape().rows;++i){ for(int c=0;c<3;++c){ raw(i,c) *= (params_.enlarger.print_exposure * exposure_factor); } }
            } else {
                for (int i=0;i<raw.shape().rows;++i){ for(int c=0;c<3;++c){ raw(i,c) = 0.0f; } }
            }
            for (int i=0;i<raw.shape().rows;++i){ for(int c=0;c<3;++c){ raw(i,c) += raw_pre(0,c); } }
            nc::NdArray<float> log_raw_fallback(raw.shape().rows,3);
            for (int i=0;i<raw.shape().rows;++i){ for (int c=0;c<3;++c){ double v = std::max(0.0, (double)raw(i,c)); log_raw_fallback(i,c) = (float)std::log10(v + 1e-10); } }
            log_raw_flat = log_raw_fallback;
        }
        
        (void)lut3d;
        // Fill log_cmy_mat
        log_cmy_mat = agx_emulsion::Matrix(log_raw_flat.shape().rows, 3);
        for (uint32_t i=0;i<log_raw_flat.shape().rows;++i){ for(int c=0;c<3;++c) log_cmy_mat(i,c) = log_raw_flat(i,c); }
    } else {
        std::cout << "Enlarger path: direct spectral (no LUT)\n";
        // Direct spectral path (existing implementation)
        nc::NdArray<float> density_spec;
        if (!agx::utils::compute_density_spectral_gpu(density_cmy, neg.data.dye_density, /*min factor*/ neg.data.dye_density_min_factor, density_spec)) {
            density_spec = agx::utils::compute_density_spectral(density_cmy, neg.data.dye_density, /*min factor*/ neg.data.dye_density_min_factor);
        }
        std::cout << "Density spectral shape: " << density_spec.shape().rows << "x" << density_spec.shape().cols << "\n";
        nc::NdArray<float> light;
        if (!agx::utils::density_to_light_gpu(density_spec, enlarger_ill, light)) {
            light = agx::utils::density_to_light(density_spec, enlarger_ill);
        }
        std::cout << "Light (enlarger) shape: " << light.shape().rows << "x" << light.shape().cols << "\n";
        nc::NdArray<float> paper_sens(paper.data.log_sensitivity.shape().rows, paper.data.log_sensitivity.shape().cols);
        for (uint32_t i=0;i<paper.data.log_sensitivity.shape().rows;++i){ for (uint32_t j=0;j<paper.data.log_sensitivity.shape().cols;++j){ paper_sens(i,j) = std::pow(10.0f, paper.data.log_sensitivity(i,j)); }}
        paper_sens = nc::nan_to_num(paper_sens);
        if (!(paper_sens.shape().cols == 3 && (light.shape().cols % paper_sens.shape().rows == 0))) {
            throw std::runtime_error("paper_sens shape mismatch");
        }
        nc::NdArray<float> cmy;
        if (!agx::utils::dot_blocks_K3_gpu(light, paper_sens, Wimg, cmy)) { cmy = dot_blocks_K3(light, paper_sens, Wimg); }
        std::cout << "CMY (pre paper) shape: " << cmy.shape().rows << "x" << cmy.shape().cols << "\n";
        cmy *= params_.enlarger.print_exposure;
        float neg_exp_comp_ev = params_.enlarger.print_exposure_compensation ? params_.camera.exposure_compensation_ev : 0.0f;
        float mid_scale = std::pow(2.0f, neg_exp_comp_ev);
        nc::NdArray<float> rgb_mid(1,3); rgb_mid(0,0)=0.184f*mid_scale; rgb_mid(0,1)=0.184f*mid_scale; rgb_mid(0,2)=0.184f*mid_scale;
        auto raw_mid = agx::utils::rgb_to_raw_hanatos2025(rgb_mid, sensitivity, "sRGB", false, neg.info.reference_illuminant, lut);
        agx_emulsion::Matrix log_raw_mid_mat(1,3);
        for (int j=0;j<3;++j){ double v = static_cast<double>(raw_mid(0,j)); if (v < 0.0) v = 0.0; log_raw_mid_mat(0,j) = std::log10(v + 1e-10); }
        agx_emulsion::Matrix density_cmy_mid_mat;
        if (!agx_emulsion::gpu_interpolate_exposure_to_density(log_raw_mid_mat, dc_neg, le_neg, gamma, density_cmy_mid_mat)) {
            density_cmy_mid_mat = agx_emulsion::interpolate_exposure_to_density(log_raw_mid_mat, dc_neg, le_neg, gamma);
        }
        auto density_cmy_mid = fromMat(density_cmy_mid_mat);
        nc::NdArray<float> density_spec_mid;
        if (!agx::utils::compute_density_spectral_gpu(density_cmy_mid, neg.data.dye_density, /*min factor*/ neg.data.dye_density_min_factor, density_spec_mid)) {
            density_spec_mid = agx::utils::compute_density_spectral(density_cmy_mid, neg.data.dye_density, /*min factor*/ neg.data.dye_density_min_factor);
        }
        nc::NdArray<float> light_mid;
        if (!agx::utils::density_to_light_gpu(density_spec_mid, enlarger_ill, light_mid)) { light_mid = agx::utils::density_to_light(density_spec_mid, enlarger_ill); }
        auto raw_mid_print = nc::dot(light_mid, paper_sens);
        std::cout << "Midgray raw print: " << raw_mid_print(0,0) << ", " << raw_mid_print(0,1) << ", " << raw_mid_print(0,2) << "\n";
        if (!params_.enlarger.just_preflash) {
            float mid_g = raw_mid_print(0,1);
            if (!std::isfinite(mid_g)) mid_g = 1e-10f;
            float factor = 1.0f / std::max(1e-10f, mid_g);
            cmy *= factor;
        } else { for (uint32_t i=0;i<cmy.shape().rows;++i) for (uint32_t j=0;j<cmy.shape().cols;++j) cmy(i,j) = 0.0f; }
        if (params_.enlarger.preflash_exposure > 0.0f) {
            auto preflash_ill = agx::model::color_enlarger(light_src_K,
                y_neutral * agx::config::ENLARGER_STEPS + params_.enlarger.preflash_y_filter_shift,
                m_neutral * agx::config::ENLARGER_STEPS + params_.enlarger.preflash_m_filter_shift,
                c_neutral * agx::config::ENLARGER_STEPS);
            nc::NdArray<float> base_den(1, neg.data.dye_density.shape().rows);
            for (uint32_t k=0;k<neg.data.dye_density.shape().rows;++k){ base_den(0,k)=neg.data.dye_density(k,3); }
            auto light_pre = agx::utils::density_to_light(base_den, preflash_ill);
            auto raw_pre = nc::dot(light_pre, paper_sens); // 1x3
            for (int i=0;i<cmy.shape().rows;++i){ for (int w=0; w<Wimg; ++w){ for (int c=0;c<3;++c){ cmy(i, w*3 + c) += raw_pre(0,c) * params_.enlarger.preflash_exposure; } } }
        }
        if (params_.enlarger.lens_blur > 0.0f) { std::vector<float> cmy_vec(cmy.size()); std::copy(cmy.begin(), cmy.end(), cmy_vec.begin()); std::vector<float> cmy_blur; agx_emulsion::Diffusion::apply_gaussian_blur(cmy_vec, Himg, Wimg, params_.enlarger.lens_blur, cmy_blur, /*truncate*/4.0f, /*try_cuda*/true); std::copy(cmy_blur.begin(), cmy_blur.end(), cmy.begin()); }
        // Compute log10(cmy)
        auto cmy_hw_by3_for_log = hw3_to_hw_by3(cmy);
        log_cmy_mat = agx_emulsion::Matrix(cmy_hw_by3_for_log.shape().rows, 3);
        for (uint32_t i=0;i<cmy_hw_by3_for_log.shape().rows;++i){ for (uint32_t j=0;j<3;++j){ double v = static_cast<double>(cmy_hw_by3_for_log(i,j)); if (v<0.0) v=0.0; log_cmy_mat(i,j) = std::log10(v + 1e-10); } }
    }
    // Apply viewing glare compensation removal to paper curves if requested
    nc::NdArray<float> paper_dc = paper.data.density_curves;
    if (paper.glare.compensation_removal_factor > 0.0f) {
        paper_dc = remove_viewing_glare_comp(paper.data.log_exposure, paper_dc,
                                             paper.glare.compensation_removal_factor,
                                             paper.glare.compensation_removal_density,
                                             paper.glare.compensation_removal_transition);
    }
    auto dc_paper = toMat(paper_dc);
    std::array<double,3> gamma_paper = {static_cast<double>(paper.data.gamma_factor[0]), static_cast<double>(paper.data.gamma_factor[1]), static_cast<double>(paper.data.gamma_factor[2])};
    std::vector<double> le_paper(paper.data.log_exposure.size());
    for (uint32_t i=0;i<paper.data.log_exposure.size();++i){ le_paper[i]=paper.data.log_exposure[i]; }
    agx_emulsion::Matrix density_print_mat;
    if (!agx_emulsion::gpu_interpolate_exposure_to_density(log_cmy_mat, dc_paper, le_paper, gamma_paper, density_print_mat)) {
        density_print_mat = agx_emulsion::interpolate_exposure_to_density(log_cmy_mat, dc_paper, le_paper, gamma_paper);
    }
    auto density_print = hw_by3_to_hw3(fromMat(density_print_mat), Himg, Wimg);
    std::cout << "Density print shape: " << density_print.shape().rows << "x" << density_print.shape().cols << "\n";

    // Scan to RGB (XYZ)
    std::cout << "Paper dye_density shape: " << paper.data.dye_density.shape().rows << "x" << paper.data.dye_density.shape().cols << "\n";
    auto scan_ill = agx::model::standard_illuminant(paper.info.viewing_illuminant); // 1xK
    std::cout << "Scan illuminant shape: " << scan_ill.shape().rows << "x" << scan_ill.shape().cols << "\n";
    float norm = 0.0f; for(uint32_t i=0;i<agx::config::SPECTRAL_SHAPE.wavelengths.size();++i) norm += agx::config::STANDARD_OBSERVER_CMFS(i,1) * scan_ill[i];
    std::cout << "Computing density spectral for scan...\n";
    nc::NdArray<float> xyz_hw3;
    if (params_.settings.use_scanner_lut) {
        // Normalize density_print for LUT input
        std::array<float,3> dmax{0.f,0.f,0.f};
        for (int c=0;c<3;++c){ float m=-1e30f; for(uint32_t i=0;i<paper.data.density_curves.shape().rows;++i) m = std::max(m, paper.data.density_curves(i,c)); dmax[c]=m; }
        nc::NdArray<float> dnorm = density_print.copy();
        for (int i=0;i<Himg;++i){ for(int w=0; w<Wimg; ++w){ for(int c=0;c<3;++c){ float d = dnorm(i,w*3+c); dnorm(i,w*3+c) = d / std::max(1e-10f, dmax[c]); }}}
        auto dnorm_flat = hw3_to_hw_by3(dnorm);
        // Spectral function: density -> log10(xyz)
        auto spec_fn = [&](const nc::NdArray<float>& dens)->nc::NdArray<float>{
            const int N = (int)dens.shape().rows;
            nc::NdArray<float> d(N,3);
            for (int i=0;i<N;++i){ for (int c=0;c<3;++c){ d(i,c) = dens(i,c) * dmax[c]; }}
            nc::NdArray<float> dens_spec;
            if (!agx::utils::compute_density_spectral_gpu(d, paper.data.dye_density, paper.data.dye_density_min_factor, dens_spec)) {
                dens_spec = agx::utils::compute_density_spectral(d, paper.data.dye_density, paper.data.dye_density_min_factor);
            }
            auto light_scan_loc = agx::utils::density_to_light(dens_spec, scan_ill);
            auto xyz_flat = dot_blocks_K3(light_scan_loc, agx::config::STANDARD_OBSERVER_CMFS, /*W*/1);
            for (uint32_t i=0;i<xyz_flat.shape().rows;++i){ for (int c=0;c<3;++c){ xyz_flat(i,c) /= norm; }}
            nc::NdArray<float> out(N,3);
            for (int i=0;i<N;++i){ for (int c=0;c<3;++c){ double v = std::max(0.0, (double)xyz_flat(i,c)); out(i,c) = (float)std::log10(v + 1e-10); } }
            return out;
        };
        auto [log_xyz_flat, lut3d_scan] = agx::utils::compute_with_lut(dnorm_flat, spec_fn, 0.0f, 1.0f, params_.settings.lut_resolution, Himg, Wimg);
        (void)lut3d_scan;
        
        // Check if LUT computation failed
        if (log_xyz_flat.shape().rows == 0 || log_xyz_flat.shape().cols == 0) {
            std::cout << "  ERROR: Scanner LUT computation failed, falling back to direct spectral path" << std::endl;
            // Fallback to direct spectral computation
            auto dens_spec_scan = agx::utils::compute_density_spectral(density_print, paper.data.dye_density, paper.data.dye_density_min_factor);
            auto light_scan = agx::utils::density_to_light(dens_spec_scan, scan_ill);
            auto xyz_hw_by3_local = dot_blocks_K3(light_scan, agx::config::STANDARD_OBSERVER_CMFS, Wimg);
            for (uint32_t i=0;i<xyz_hw_by3_local.shape().rows;++i){ for (int c=0;c<3;++c){ xyz_hw_by3_local(i,c) /= norm; }}
            log_xyz_flat = xyz_hw_by3_local;
            for (uint32_t i=0;i<log_xyz_flat.shape().rows;++i){ for (int c=0;c<3;++c){ log_xyz_flat(i,c) = std::pow(10.0f, log_xyz_flat(i,c)); }}
        } else {
            for (uint32_t i=0;i<log_xyz_flat.shape().rows;++i){ for (int c=0;c<3;++c){ log_xyz_flat(i,c) = std::pow(10.0f, log_xyz_flat(i,c)); }}
        }
        auto xyz_hw_by3_local = log_xyz_flat;
        xyz_hw3 = hw_by3_to_hw3(xyz_hw_by3_local, Himg, Wimg);
        xyz_hw3 = nc::nan_to_num(xyz_hw3);
    } else {
        nc::NdArray<float> dens_spec_scan;
        if (!agx::utils::compute_density_spectral_gpu(density_print, paper.data.dye_density, /*min factor*/ paper.data.dye_density_min_factor, dens_spec_scan)) {
            dens_spec_scan = agx::utils::compute_density_spectral(density_print, paper.data.dye_density, /*min factor*/ paper.data.dye_density_min_factor);
        }
        std::cout << "dens_spec_scan shape: " << dens_spec_scan.shape().rows << "x" << dens_spec_scan.shape().cols << "\n";
        nc::NdArray<float> light_scan;
        if (!agx::utils::density_to_light_gpu(dens_spec_scan, scan_ill, light_scan)) {
            light_scan = agx::utils::density_to_light(dens_spec_scan, scan_ill);
        }
        std::cout << "Light (scan) shape: " << light_scan.shape().rows << "x" << light_scan.shape().cols << "\n";
        if (!agx::utils::dot_blocks_K3_gpu(light_scan, agx::config::STANDARD_OBSERVER_CMFS, Wimg, xyz_hw3)) {
            xyz_hw3 = dot_blocks_K3(light_scan, agx::config::STANDARD_OBSERVER_CMFS, Wimg);
        }
        xyz_hw3 = xyz_hw3 / norm;
        xyz_hw3 = nc::nan_to_num(xyz_hw3);
    }
    std::cout << "XYZ (H x W*3) shape: " << xyz_hw3.shape().rows << "x" << xyz_hw3.shape().cols << std::endl;
        
        // Convert XYZ -> output RGB, compute viewing illuminant xy from scan SPD to match Python
        std::cout << "Computing viewing illuminant..." << std::endl;
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
        std::cout << "Viewing illuminant computed" << std::endl;
        
        auto xyz_hw_by3 = hw3_to_hw_by3(xyz_hw3);
        std::cout << "XYZ reshaped to " << xyz_hw_by3.shape().rows << "x" << xyz_hw_by3.shape().cols << std::endl;
        
        // Add stochastic glare if requested (matches Python compute_random_glare_amount)
        if (params_.settings.apply_paper_glare && paper.glare.active && paper.glare.percent > 0.0f) {
            std::cout << "Applying paper glare..." << std::endl;
            xyz_hw_by3 = agx::utils::add_random_glare(
                xyz_hw_by3,
                illuminant_xyz,
                paper.glare.percent / 100.0f,
                paper.glare.roughness,
                paper.glare.blur,
                Himg,
                Wimg
            );
            std::cout << "Paper glare applied" << std::endl;
        }
        
        std::cout << "Converting XYZ to RGB..." << std::endl;
        auto rgb_hw_by3 = colour::XYZ_to_RGB(xyz_hw_by3, params_.io.output_color_space, params_.io.output_cctf_encoding, illuminant_xy, "CAT02");
        std::cout << "XYZ to RGB conversion completed" << std::endl;
        
        auto rgb_hw3 = hw_by3_to_hw3(rgb_hw_by3, Himg, Wimg);
        std::cout << "RGB reshaped to " << rgb_hw3.shape().rows << "x" << rgb_hw3.shape().cols << std::endl;
        
        // Scanner lens blur and unsharp (post-conversion), matching Python order
        if (params_.scanner.lens_blur > 0.0f) {
            std::cout << "Applying scanner lens blur..." << std::endl;
            std::vector<float> rgb_vec(rgb_hw3.size()); std::copy(rgb_hw3.begin(), rgb_hw3.end(), rgb_vec.begin());
            std::vector<float> rgb_blur;
            agx_emulsion::Diffusion::apply_gaussian_blur(rgb_vec, Himg, Wimg, params_.scanner.lens_blur, rgb_blur, /*truncate*/4.0f, /*try_cuda*/true);
            std::copy(rgb_blur.begin(), rgb_blur.end(), rgb_hw3.begin());
            std::cout << "Scanner lens blur applied" << std::endl;
        }
        if (params_.scanner.unsharp_sigma > 0.0f && params_.scanner.unsharp_amount > 0.0f) {
            std::cout << "Applying scanner unsharp mask..." << std::endl;
            std::vector<float> rgb_vec(rgb_hw3.size()); std::copy(rgb_hw3.begin(), rgb_hw3.end(), rgb_vec.begin());
            std::vector<float> rgb_unsharp;
            agx_emulsion::Diffusion::apply_unsharp_mask(rgb_vec, Himg, Wimg, params_.scanner.unsharp_sigma, params_.scanner.unsharp_amount, rgb_unsharp);
            std::copy(rgb_unsharp.begin(), rgb_unsharp.end(), rgb_hw3.begin());
            std::cout << "Scanner unsharp mask applied" << std::endl;
        }
        
        if (params_.debug.return_negative_density_cmy) return density_cmy;
        if (params_.debug.return_print_density_cmy) return density_print;
        
        // Final clamp to [0,1] to match Python's np.clip after optional CCTF encoding
        std::cout << "Applying final clamp..." << std::endl;
        for (auto &v : rgb_hw3) {
            if (v < 0.0f) v = 0.0f;
            else if (v > 1.0f) v = 1.0f;
        }
        std::cout << "Final clamp completed" << std::endl;
        
        std::cout << "Output RGB shape: " << rgb_hw3.shape().rows << "x" << rgb_hw3.shape().cols << std::endl;
        std::cout << "Process::run() completed successfully" << std::endl;
        return rgb_hw3;
}

} } // namespace agx::process
