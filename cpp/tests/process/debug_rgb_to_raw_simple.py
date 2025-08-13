#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the agx_emulsion module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from agx_emulsion.utils.spectral_upsampling import rgb_to_tc_b
from agx_emulsion.profiles.io import load_profile

def debug_simple():
    """Debug individual functions to see intermediate values"""
    
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
    
    # Get sensitivity from profile
    sensitivity = neg_profile.data.log_sensitivity
    print(f"Sensitivity shape: {sensitivity.shape}")
    
    # Test rgb_to_tc_b function
    print(f"\nTesting rgb_to_tc_b function...")
    try:
        tc, b = rgb_to_tc_b(center_pixel.reshape(1, 3), 
                           color_space='sRGB', 
                           apply_cctf_decoding=False, 
                           reference_illuminant='D65')
        print(f"tc shape: {tc.shape}, tc values: {tc}")
        print(f"b shape: {b.shape}, b values: {b}")
        
        # Compare with C++ values from stages
        print(f"\nC++ values from stages:")
        print(f"raw_center: [0.82472819, 0.83753932, 1.63281977]")
        print(f"This suggests the issue is in the LUT interpolation or midgray normalization")
        
    except Exception as e:
        print(f"Error in rgb_to_tc_b: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    debug_simple()

