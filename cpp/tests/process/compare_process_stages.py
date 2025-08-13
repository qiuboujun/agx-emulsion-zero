#!/usr/bin/env python3

import numpy as np
import json
import sys
import os

# Add the agx_emulsion module to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from agx_emulsion.model.process import photo_params
from agx_emulsion.profiles.io import load_profile
from agx_emulsion.model.illuminants import standard_illuminant
from agx_emulsion.model.color_filters import color_enlarger
from agx_emulsion.utils.conversions import density_to_light
from agx_emulsion.model.emulsion import compute_density_spectral, develop_simple
from agx_emulsion.utils.autoexposure import measure_autoexposure_ev
from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025

H, W = 5, 5

def create_test_image():
    img = np.zeros((H, W, 3), dtype=np.float64)
    for i in range(H):
        for j in range(W):
            v = float(i*W + j) / float(H*W - 1)
            img[i, j, 0] = v
            img[i, j, 1] = 0.8 * v
            img[i, j, 2] = 1.2 * v
    return img

def run_python_stages():
    neg = load_profile('kodak_portra_400_auc')
    paper = load_profile('kodak_portra_endura_uc')
    img = create_test_image()
    ev = measure_autoexposure_ev(img, color_space='sRGB', apply_cctf_decoding=False)

    # Disable glare/blur to match C++
    params = photo_params(negative='kodak_portra_400_auc', print_paper='kodak_portra_endura_uc')
    params.print_paper.glare.active = False
    params.scanner.lens_blur = 0.0
    params.scanner.unsharp_mask = (0.0, 0.0)

    # Film develop
    sens_neg = np.nan_to_num(10**neg.data.log_sensitivity)
    raw = rgb_to_raw_hanatos2025(img, sens_neg, color_space='sRGB', apply_cctf_decoding=False, reference_illuminant=neg.info.reference_illuminant)
    raw *= 2**ev
    log_raw = np.log10(np.fmax(raw, 0.0) + 1e-10)
    density_cmy = develop_simple(neg, log_raw)

    # Enlarger and pre-paper CMY
    light_src = standard_illuminant(params.enlarger.illuminant)
    yval = params.enlarger.y_filter_neutral * 170 + params.enlarger.y_filter_shift
    mval = params.enlarger.m_filter_neutral * 170 + params.enlarger.m_filter_shift
    cval = params.enlarger.c_filter_neutral * 170
    print_ill = color_enlarger(light_src, yval, mval, cval)
    dens_spec = compute_density_spectral(neg, density_cmy)
    light = density_to_light(dens_spec, print_ill)
    sens_paper = np.nan_to_num(10**paper.data.log_sensitivity)
    cmy_pre = np.einsum('ijk,kl->ijl', light, sens_paper)

    # Midgray factor
    neg_exp_comp_ev = params.camera.exposure_compensation_ev if params.enlarger.print_exposure_compensation else 0.0
    mid_scale = 2**neg_exp_comp_ev
    rgb_mid = np.array([[[0.184*mid_scale, 0.184*mid_scale, 0.184*mid_scale]]], dtype=np.float64)
    raw_mid = rgb_to_raw_hanatos2025(rgb_mid, sens_neg, color_space='sRGB', apply_cctf_decoding=False, reference_illuminant=neg.info.reference_illuminant)
    log_raw_mid = np.log10(np.fmax(raw_mid, 0.0) + 1e-10)
    dens_mid = develop_simple(neg, log_raw_mid)
    dens_spec_mid = compute_density_spectral(neg, dens_mid)
    light_mid = density_to_light(dens_spec_mid, print_ill)
    raw_mid_print = np.einsum('ijk,kl->ijl', light_mid, sens_paper)
    midgray_factor = 1.0 / np.clip(raw_mid_print[:, :, 1], 1e-10, None)

    return {
        'cmy_pre_paper_center': cmy_pre[H//2, W//2, :].tolist(),
        'midgray_factor': float(midgray_factor[0,0])
    }

def compare_stages():
    cpp_stages_path = os.path.join(os.path.dirname(__file__), '../../../build/tmp_process_cpp_stages.json')
    if not os.path.exists(cpp_stages_path):
        print(f"Error: C++ stages file not found at {cpp_stages_path}")
        return
    with open(cpp_stages_path, 'r') as f:
        cpp_stages = json.load(f)

    py = run_python_stages()

    def show_diff(name, a, b):
        a = np.array(a)
        b = np.array(b)
        d = np.abs(a - b)
        print(f"{name} C++: {a}, Python: {b}, max abs diff: {d.max() if d.size else 0}")

    show_diff('cmy_pre_paper_center', cpp_stages.get('cmy_pre_paper_center', []), py['cmy_pre_paper_center'])
    print('midgray_factor C++:', cpp_stages.get('midgray_factor'), 'Python:', py['midgray_factor'])

if __name__ == "__main__":
    compare_stages()
