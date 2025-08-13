#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the agx_emulsion module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from agx_emulsion.utils.spectral_upsampling import load_spectra_lut, rgb_to_raw_hanatos2025
from agx_emulsion.profiles.io import load_profile

def debug_rgb_to_raw():
    """Debug rgb_to_raw_hanatos2025 on the same center pixel as C++"""
    
    # Load the same profiles as C++
    neg_profile = load_profile('kodak_portra_400_auc')
    
    # Create the same test image as C++
    H, W = 5, 5
    img = np.zeros((H, W*3), dtype=np.float32)
    
    for i in range(H):
        for j in range(W):
            v = float(i*W + j) / float(H*W - 1)
            img[i, 3*j + 0] = v
            img[i, 3*j + 1] = 0.8 * v
            img[i, 3*j + 2] = 1.2 * v
    
    # Get center pixel (same as C++)
    center_pixel = img[2, 6:9]  # Row 2, columns 6-8
    print(f"Center pixel RGB: {center_pixel}")
    
    # Load spectra LUT
    spectra_lut = load_spectra_lut()
    print(f"Spectra LUT shape: {spectra_lut.shape}")
    
    # Get sensitivity from profile
    sensitivity = neg_profile.data.log_sensitivity
    print(f"Sensitivity shape: {sensitivity.shape}")
    
    # Run rgb_to_raw_hanatos2025 on center pixel
    print(f"\nRunning rgb_to_raw_hanatos2025 on center pixel...")
    
    # Reshape to single pixel for the function
    center_pixel_reshaped = center_pixel.reshape(1, 3)
    
    try:
        raw_output = rgb_to_raw_hanatos2025(center_pixel_reshaped, sensitivity, 
                                           color_space='sRGB', apply_cctf_decoding=False, 
                                           reference_illuminant='D65')
        print(f"Python raw output: {raw_output}")
        print(f"Python raw output shape: {raw_output.shape}")
        
        # Compare with C++ output
        cpp_raw = np.array([0.82472819, 0.83753932, 1.63281977])
        diff = np.abs(raw_output.flatten() - cpp_raw)
        print(f"\nComparison with C++:")
        print(f"C++ raw_center: {cpp_raw}")
        print(f"Python raw output: {raw_output.flatten()}")
        print(f"Differences: {diff}")
        print(f"Max abs diff: {np.max(diff):.6f}")
        print(f"Mean abs diff: {np.mean(diff):.6f}")
        
    except Exception as e:
        print(f"Error running rgb_to_raw_hanatos2025: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_rgb_to_raw()
