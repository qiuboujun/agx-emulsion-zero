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

    center_sens = sens_paper
    center_light = light[H//2, W//2, :]
    dot_center = np.array([
        np.sum(center_light * center_sens[:,0]),
        np.sum(center_light * center_sens[:,1]),
        np.sum(center_light * center_sens[:,2])
    ])

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
        'print_illuminant': print_ill.tolist(),
        'light_enlarger_center': center_light.tolist(),
        'log_raw_center': np.log10(np.fmax(raw[H//2, W//2, :], 0.0)+1e-10).tolist(),
        'density_cmy_center': density_cmy[H//2, W//2, :].tolist(),
        'density_spectral_center': dens_spec[H//2, W//2, :].tolist(),
        'paper_sensitivity': sens_paper.flatten().tolist(),
        'paper_dot_center': dot_center.tolist(),
        'cmy_pre_paper_center': cmy_pre[H//2, W//2, :].tolist(),
        'midgray_factor': float(midgray_factor[0,0])
    }

def compare_stages():
    cpp_stages_path = os.path.join(os.path.dirname(__file__), '../../../build/tmp_process_cpp_stages.json')
    if not os.path.exists(cpp_stages_path):
        print(f"Error: C++ stages file not found at {cpp_stages_path}")
        return
    with open(cpp_stages_path, 'r') as f:
        cpp = json.load(f)

    py = run_python_stages()

    def to_float_array(xs):
        if xs is None:
            return np.array([], dtype=np.float64)
        flat = []
        for v in xs:
            if v is None:
                flat.append(np.nan)
            else:
                try:
                    flat.append(float(v))
                except Exception:
                    flat.append(np.nan)
        return np.array(flat, dtype=np.float64)

    def show_diff(name, a, b):
        a = to_float_array(a)
        b = to_float_array(b)
        if a.size == 0 or b.size == 0:
            print(f"{name} diff max: 0")
            return
        # match lengths
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
        d = np.abs(a - b)
        print(f"{name} diff max: {np.nanmax(d) if d.size else 0}")

    show_diff('print_illuminant', cpp.get('print_illuminant', []), py['print_illuminant'])
    show_diff('light_enlarger_center', cpp.get('light_enlarger_center', []), py['light_enlarger_center'])
    show_diff('log_raw_center', cpp.get('log_raw_center', []), py['log_raw_center'])
    show_diff('density_cmy_center', cpp.get('density_cmy_center', []), py['density_cmy_center'])
    show_diff('density_spectral_center', cpp.get('density_spectral_center', []), py['density_spectral_center'])
    show_diff('paper_sensitivity', cpp.get('paper_sensitivity', []), py['paper_sensitivity'])
    show_diff('paper_dot_center', cpp.get('paper_dot_center', []), py['paper_dot_center'])
    # If we have per-wavelength parts from C++, print them and the summed diffs
    parts_r = cpp.get('paper_dot_center_parts_r', [])
    parts_g = cpp.get('paper_dot_center_parts_g', [])
    parts_b = cpp.get('paper_dot_center_parts_b', [])
    if parts_r and parts_g and parts_b:
        center_light = np.array(py['light_enlarger_center'])
        K = center_light.shape[0]
        sens = np.array(py['paper_sensitivity']).reshape(K,3)
        py_r = center_light * sens[:,0]
        py_g = center_light * sens[:,1]
        py_b = center_light * sens[:,2]
        show_diff('paper_dot_center_parts_r', parts_r, py_r.tolist())
        show_diff('paper_dot_center_parts_g', parts_g, py_g.tolist())
        show_diff('paper_dot_center_parts_b', parts_b, py_b.tolist())

    print('cmy_pre_paper_center C++:', cpp.get('cmy_pre_paper_center'), 'Python:', py['cmy_pre_paper_center'])
    print('midgray_factor C++:', cpp.get('midgray_factor'), 'Python:', py['midgray_factor'])

if __name__ == "__main__":
    compare_stages()
