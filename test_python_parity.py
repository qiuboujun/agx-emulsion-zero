#!/usr/bin/env python3
"""
Test script to run Python implementation with same input as C++ test
for 1:1 parity comparison
"""

import numpy as np
import sys
import os

# Add agx_emulsion to path
sys.path.insert(0, 'agx_emulsion')

from agx_emulsion.model.process import AgXPhoto, photo_params

def test_python_parity():
    """Test Python implementation with same input as C++ test"""
    
    print("=== Python Implementation Test ===")
    print("Testing with same input as C++: ACES2065-1 linear [0.107023, 0.107023, 0.107023]")
    
    # Create test image: 1920x1080 with center pixel [0.107023, 0.107023, 0.107023]
    # Same dimensions as your C++ test
    height, width = 1080, 1920
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
    
    # Process image
    print("\nProcessing image...")
    result = processor.process(image)
    
    print(f"Output image shape: {result.shape}")
    print(f"Output RGB center ({center_x},{center_y}): {result[center_y, center_x]}")
    
    # Print intermediate values if available
    if hasattr(processor, 'debug') and hasattr(processor.debug, 'luts'):
        if hasattr(processor.debug.luts, 'enlarger_lut'):
            print(f"Enlarger LUT created: {processor.debug.luts.enlarger_lut.shape if processor.debug.luts.enlarger_lut is not None else 'None'}")
        if hasattr(processor.debug.luts, 'scanner_lut'):
            print(f"Scanner LUT created: {processor.debug.luts.scanner_lut.shape if processor.debug.luts.scanner_lut is not None else 'None'}")
    
    # Also test with a smaller image to see if we can get more detailed debug info
    print("\n=== Testing with smaller image for detailed debug ===")
    small_height, small_width = 100, 100
    small_center_y, small_center_x = small_height // 2, small_width // 2
    small_image = np.full((small_height, small_width, 3), 0.107023, dtype=np.float64)
    
    print(f"Small image shape: {small_image.shape}")
    print(f"Small image center ({small_center_x},{small_center_y}): {small_image[small_center_y, small_center_x]}")
    
    small_result = processor.process(small_image)
    print(f"Small output shape: {small_result.shape}")
    print(f"Small output center ({small_center_x},{small_center_y}): {small_result[small_center_y, small_center_x]}")
    
    print("\n=== Test completed ===")

if __name__ == "__main__":
    test_python_parity()
