#!/usr/bin/env python3

import numpy as np
import json
import sys
import os

# Add the agx_emulsion module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from agx_emulsion.model.process import AgXPhoto
from agx_emulsion.profiles.io import load_profile
from agx_emulsion.utils.spectral_upsampling import load_spectra_lut

def create_test_image():
    H, W = 5, 5
    img = np.zeros((H, W*3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            v = float(i*W + j) / float(H*W - 1)
            img[i, 3*j + 0] = v
            img[i, 3*j + 1] = 0.8 * v
            img[i, 3*j + 2] = 1.2 * v
    return img

def run_python_stages():
    neg_profile = load_profile('kodak_portra_400_auc')
    pos_profile = load_profile('kodak_portra_endura_uc')
    image = create_test_image()

    from agx_emulsion.utils.autoexposure import measure_autoexposure_ev
    image_3d = image.reshape(5, 5, 3)
    ev = measure_autoexposure_ev(image_3d, color_space='sRGB', apply_cctf_decoding=False)

    center_pixel = image[2, 6:9]
    spectra_lut = load_spectra_lut()
    sensitivity = neg_profile.data.log_sensitivity
    return {
        'autoexposure_ev': ev,
        'center_pixel_rgb': center_pixel.tolist(),
        'spectra_lut_shape': list(spectra_lut.shape),
        'sensitivity_shape': list(sensitivity.shape)
    }

def compare_stages():
    cpp_stages_path = os.path.join(os.path.dirname(__file__), '../../../build/tmp_process_cpp_stages.json')
    if not os.path.exists(cpp_stages_path):
        print(f"Error: C++ stages file not found at {cpp_stages_path}")
        return
    with open(cpp_stages_path, 'r') as f:
        cpp_stages = json.load(f)

    print("=== C++ Stage Outputs ===")
    for stage, value in cpp_stages.items():
        if isinstance(value, (list, dict)):
            print(f"{stage}: {type(value).__name__} with {len(value)} elements")
        else:
            print(f"{stage}: {value}")

    print("\n=== Python Stage Outputs ===")
    python_stages = run_python_stages()
    for stage, value in python_stages.items():
        if isinstance(value, (list, dict)):
            print(f"{stage}: {type(value).__name__} with {len(value)} elements")
        else:
            print(f"{stage}: {value}")

    print("\n=== Stage-by-Stage Comparison ===")
    if 'image_center_rgb' in cpp_stages:
        cpp_rgb = np.array(cpp_stages['image_center_rgb'])
        py_rgb = np.array(python_stages['center_pixel_rgb'])
        diff = np.abs(cpp_rgb - py_rgb)
        print(f"Input RGB (center pixel) - C++: {cpp_rgb}, Python: {py_rgb}")
        print(f"RGB differences: {diff}, Max diff: {np.max(diff):.6f}")

    if 'raw_center' in cpp_stages:
        cpp_raw = np.array(cpp_stages['raw_center'])
        print(f"C++ raw_center (rgb_to_raw_hanatos2025): {cpp_raw}")

    # More direct checks of shapes we fixed
    if 'print_illuminant' in cpp_stages:
        print(f"print_illuminant size: {len(cpp_stages['print_illuminant'])} (expect 81)")
    if 'scan_illuminant' in cpp_stages:
        print(f"scan_illuminant size: {len(cpp_stages['scan_illuminant'])} (expect 81)")

    # No stale numbers
    print("\nFinal output comparison is now printed by compare_process_py.py")

if __name__ == "__main__":
    compare_stages()
