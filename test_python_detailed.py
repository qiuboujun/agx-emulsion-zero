#!/usr/bin/env python3
"""
Detailed Python test to show intermediate values at each stage
for comparison with C++ debug output
"""

import numpy as np
import sys
import os

# Add agx_emulsion to path
sys.path.insert(0, 'agx_emulsion')

from agx_emulsion.model.process import AgXPhoto, photo_params
from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025
from agx_emulsion.model.emulsion import Film
from agx_emulsion.utils.autoexposure import measure_autoexposure_ev

def test_python_detailed():
    """Test Python implementation with detailed intermediate values"""
    
    print("=== Python Detailed Implementation Test ===")
    print("Testing with same input as C++: ACES2065-1 linear [0.107023, 0.107023, 0.107023]")
    
    # Create small test image for easier debugging
    height, width = 100, 100
    center_y, center_x = height // 2, width // 2
    
    # Create solid color image matching your input
    image = np.full((height, width, 3), 0.107023, dtype=np.float64)
    
    print(f"Input image shape: {image.shape}")
    print(f"Input RGB center ({center_x},{center_y}): {image[center_y, center_x]}")
    
    # Set up parameters to match C++ test
    params = photo_params(
        negative='kodak_vision3_250d_uc',  # Same as C++
        print_paper='kodak_2383_uc'        # Same as C++
    )
    
    # Match C++ settings exactly
    params.io.input_color_space = 'ACES2065-1'  # ACES2065-1 linear
    params.io.input_cctf_decoding = False       # No CCTF decoding for ACES linear
    params.io.output_color_space = 'ACES2065-1' # ACES2065-1 linear output
    params.io.output_cctf_encoding = False      # No CCTF encoding for ACES linear
    params.io.full_image = True                 # Enable all effects
    
    # Disable autoexposure to match C++ plugin
    params.camera.auto_exposure = False
    params.camera.exposure_compensation_ev = 0.0
    
    # Disable LUTs to match your C++ test (camera LUT disabled)
    params.settings.use_camera_lut = False
    params.settings.use_enlarger_lut = True     # Keep enlarger LUT as in your test
    params.settings.use_scanner_lut = True      # Keep scanner LUT as in your test
    params.settings.lut_resolution = 17         # Same as C++
    
    # Disable halation and grain to match your C++ test
    params.negative.halation.active = False
    params.negative.grain.active = False
    
    # Disable DIR couplers to match your C++ test
    params.negative.dir_couplers.active = False
    
    # Disable lens blur to match your C++ test
    params.camera.lens_blur_um = 0.0
    params.enlarger.lens_blur = 0.0
    params.scanner.lens_blur = 0.0
    params.scanner.unsharp_mask = (0.0, 0.0)
    
    print(f"Film stock: {params.negative.info.stock}")
    print(f"Print paper: {params.print_paper.info.stock}")
    print(f"Input color space: {params.io.input_color_space}")
    print(f"Output color space: {params.io.output_color_space}")
    print(f"LUTs: camera={params.settings.use_camera_lut}, enlarger={params.settings.use_enlarger_lut}, scanner={params.settings.use_scanner_lut}")
    print(f"LUT resolution: {params.settings.lut_resolution}")
    
    # Create processor
    processor = AgXPhoto(params)
    
    # Step 1: Auto exposure
    print("\n=== Step 1: Auto Exposure ===")
    exposure_ev = processor._auto_exposure(image)
    print(f"Exposure EV: {exposure_ev}")
    
    # Step 2: RGB to RAW (camera exposure)
    print("\n=== Step 2: RGB to RAW (Camera Exposure) ===")
    sensitivity = 10**params.negative.data.log_sensitivity
    sensitivity = np.nan_to_num(sensitivity)
    
    # Apply band-pass filter like C++ defaults (UV/IR filters active)
    from agx_emulsion.model.color_filters import compute_band_pass_filter
    bpf = compute_band_pass_filter(params.camera.filter_uv, params.camera.filter_ir)
    sensitivity = sensitivity * bpf[:, None]
    
    # Apply exposure via direct hanatos2025 path to mirror C++ logic
    from agx_emulsion.utils.spectral_upsampling import rgb_to_tc_b, HANATOS2025_SPECTRA_LUT
    from agx_emulsion.utils.fast_interp_lut import apply_lut_cubic_2d
    
    # tc, b for input
    tc_raw, b = rgb_to_tc_b(image, color_space=params.io.input_color_space,
                             apply_cctf_decoding=params.io.input_cctf_decoding,
                             reference_illuminant=params.negative.info.reference_illuminant)
    print(f"tc center ({center_x},{center_y}): {tc_raw[center_y, center_x]}")
    print(f"b center ({center_x},{center_y}): {b[center_y, center_x]}")
    
    # Preproject LUT by sensitivity and interpolate
    tc_lut = np.tensordot(HANATOS2025_SPECTRA_LUT, sensitivity, axes=([2],[0]))  # (L,L,K) dot (K,3) -> (L,L,3)
    raw_unscaled = apply_lut_cubic_2d(tc_lut, tc_raw)
    raw_pre_norm = raw_unscaled * b[..., None]
    print(f"RAW before midgray norm center ({center_x},{center_y}): {raw_pre_norm[center_y, center_x]}")
    
    # Use library function for final raw (includes midgray normalization)
    raw = rgb_to_raw_hanatos2025(
        image,
        sensitivity,
        color_space=params.io.input_color_space,
        apply_cctf_decoding=params.io.input_cctf_decoding,
        reference_illuminant=params.negative.info.reference_illuminant
    )
    raw *= 2**0.0
    
    print(f"RAW (film) shape: {raw.shape}")
    print(f"RAW after EV center ({center_x},{center_y}): {raw[center_y, center_x]}")
    
    # Step 3: Film development
    print("\n=== Step 3: Film Development ===")
    log_raw = np.log10(np.fmax(raw, 0.0) + 1e-10)
    film = Film(params.negative)
    density_cmy = film.develop(log_raw, pixel_size_um=6.07639, use_fast_stats=False)
    
    print(f"Density CMY shape: {density_cmy.shape}")
    print(f"Density CMY before DIR center ({center_x},{center_y}): {density_cmy[center_y, center_x]}")
    
    # Step 4: Print exposure (enlarger)
    print("\n=== Step 4: Print Exposure (Enlarger) ===")
    # Normalize density for LUT
    film_density_cmy_normalized = processor._normalize_film_density(density_cmy)
    
    def spectral_calculation(density_cmy_n):
        density_cmy = processor._denormalize_film_density(density_cmy_n)
        return processor._film_density_cmy_to_print_log_raw(density_cmy)
    
    log_raw_print = processor._spectral_lut_compute(
        film_density_cmy_normalized, 
        spectral_calculation,
        use_lut=params.settings.use_enlarger_lut,
        save_enlarger_lut=True
    )
    
    print(f"Print log_raw shape: {log_raw_print.shape}")
    print(f"Print log_raw center ({center_x},{center_y}): {log_raw_print[center_y, center_x]}")
    
    # Step 5: Print development
    print("\n=== Step 5: Print Development ===")
    from agx_emulsion.model.emulsion import develop_simple
    density_print = develop_simple(params.print_paper, log_raw_print)
    
    print(f"Density print shape: {density_print.shape}")
    print(f"Density print center ({center_x},{center_y}): {density_print[center_y, center_x]}")
    
    # Step 6: Scan
    print("\n=== Step 6: Scan ===")
    rgb = processor._density_cmy_to_rgb(density_print, use_lut=params.settings.use_scanner_lut)
    
    print(f"RGB after scan shape: {rgb.shape}")
    print(f"RGB after scan center ({center_x},{center_y}): {rgb[center_y, center_x]}")
    
    # Step 7: Final output
    print("\n=== Step 7: Final Output ===")
    result = processor.process(image)
    
    print(f"Final output shape: {result.shape}")
    print(f"Final output center ({center_x},{center_y}): {result[center_y, center_x]}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_python_detailed()
