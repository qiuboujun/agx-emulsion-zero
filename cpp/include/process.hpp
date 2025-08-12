#pragma once

#include "NumCpp.hpp"
#include <string>
#include <memory>

namespace agx {
namespace process {

struct IOSettings {
    std::string input_color_space = "ProPhoto RGB";
    bool input_cctf_decoding = false;
    std::string output_color_space = "sRGB";
    bool output_cctf_encoding = true;
    bool crop = false;
    float crop_center_x = 0.5f;
    float crop_center_y = 0.5f;
    float crop_size_x = 0.1f;
    float crop_size_y = 1.0f;
    float preview_resize_factor = 1.0f;
    float upscale_factor = 1.0f;
    bool full_image = false;
    bool compute_negative = false;
    bool compute_film_raw = false;
};

struct CameraSettings {
    float exposure_compensation_ev = 0.0f;
    bool auto_exposure = true;
    std::string auto_exposure_method = "center_weighted";
    float lens_blur_um = 0.0f;
    float film_format_mm = 35.0f;
    float filter_uv_amp = 1.0f; float filter_uv_wl = 410.0f; float filter_uv_width = 8.0f;
    float filter_ir_amp = 1.0f; float filter_ir_wl = 675.0f; float filter_ir_width = 15.0f;
};

struct EnlargerSettings {
    std::string illuminant = "TH-KG3-L";
    float print_exposure = 1.0f;
    bool print_exposure_compensation = true;
    float y_filter_shift = 0.0f;
    float m_filter_shift = 0.0f;
    float y_filter_neutral = 0.9f;
    float m_filter_neutral = 0.5f;
    float c_filter_neutral = 0.35f;
    float lens_blur = 0.0f;
    float preflash_exposure = 0.0f;
    float preflash_y_filter_shift = 0.0f;
    float preflash_m_filter_shift = 0.0f;
    bool just_preflash = false;
};

struct ScannerSettings {
    float lens_blur = 0.55f;
    float unsharp_sigma = 0.7f;
    float unsharp_amount = 1.0f;
};

struct ProcessSettings {
    std::string rgb_to_raw_method = "hanatos2025";
    bool use_camera_lut = false;
    bool use_enlarger_lut = false;
    bool use_scanner_lut = false;
    int lut_resolution = 17;
    bool use_fast_stats = false;
};

struct ProfileSet {
    std::string negative = "kodak_portra_400_auc";
    std::string print_paper = "kodak_portra_endura_uc";
};

struct Params {
    IOSettings io;
    CameraSettings camera;
    EnlargerSettings enlarger;
    ScannerSettings scanner;
    ProcessSettings settings;
    ProfileSet profiles;
};

// Orchestrator
class Process {
public:
    explicit Process(const Params& params);
    // image: HxWx3 double in [0,1]
    nc::NdArray<float> run(const nc::NdArray<float>& image);

private:
    Params params_;
};

} // namespace process
} // namespace agx
