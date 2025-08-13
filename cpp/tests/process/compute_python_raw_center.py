#!/usr/bin/env python3

import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../..'))

from agx_emulsion.utils.spectral_upsampling import (
    load_spectra_lut, rgb_to_tc_b, tri2quad
)
from agx_emulsion.profiles.io import load_profile
from agx_emulsion.config import SPECTRAL_SHAPE
from opt_einsum import contract
from agx_emulsion.utils.fast_interp_lut import apply_lut_cubic_2d
import scipy.interpolate


def build_cpp_like_image(H=5, W=5):
    img = np.zeros((H, W*3), dtype=np.float32)
    for i in range(H):
        for j in range(W):
            v = float(i*W + j) / float(H*W - 1)
            img[i, 3*j + 0] = v
            img[i, 3*j + 1] = 0.8 * v
            img[i, 3*j + 2] = 1.2 * v
    return img


ess_rate = None

def main():
    neg = load_profile('kodak_portra_400_auc')
    # Sensitivity must be 10**log_sensitivity and NaNs replaced, same as C++
    sens = np.nan_to_num(10 ** neg.data.log_sensitivity)
    img = build_cpp_like_image()
    center_rgb = img[2, 6:9].reshape(1,3)
    print(f"center_rgb: {center_rgb}")

    # 1) tc, b
    # Use profile reference illuminant (Portra 400 uses D55)
    ref_illu = 'D55'
    tc, b = rgb_to_tc_b(center_rgb, color_space='sRGB', apply_cctf_decoding=False, reference_illuminant=ref_illu)
    print(f"tc: {tc}")
    print(f"b: {b}")

    # 2) Preproject spectra LUT with sensitivity: (L,L,81) x (81,3) -> (L,L,3)
    spectra_lut = load_spectra_lut()  # (L,L,81)
    lut_proj = contract('ijl,lm->ijm', spectra_lut, sens)
    print(f"lut_proj finite? {np.isfinite(lut_proj).all()}")

    # 3) Apply 2D cubic LUT at tc using Mitchell–Netravali (Numba) like C++
    img_tc = np.zeros((1,1,2), dtype=np.float64)
    img_tc[0,0,0] = float(tc[0,0])
    img_tc[0,0,1] = float(tc[0,1])
    raw_pre = apply_lut_cubic_2d(lut_proj.astype(np.float64), img_tc)
    raw_pre = raw_pre.reshape(1,3)
    print(f"raw_pre (lut result before scale): {raw_pre}")

    # 4) Scale by b
    raw_scaled = raw_pre * b.reshape(1,1)
    print(f"raw_scaled (after b): {raw_scaled}")

    # 5) Midgray normalization: linear RGI over original spectra LUT
    mid_rgb = np.array([[0.184, 0.184, 0.184]], dtype=np.float64)
    tc_m, b_m = rgb_to_tc_b(mid_rgb, color_space='sRGB', apply_cctf_decoding=False, reference_illuminant=ref_illu)
    v = np.linspace(0, 1, spectra_lut.shape[0])
    rgi = scipy.interpolate.RegularGridInterpolator((v, v), spectra_lut.astype(np.float64), method='linear')
    mid_spec = rgi(tc_m).reshape(-1)
    mid_spec *= float(b_m)  # scale spectrum by brightness
    raw_mid = np.einsum('k,km->m', mid_spec, sens)
    mid_norm = raw_mid[1]
    print(f"mid_spec len: {len(mid_spec)} raw_mid: {raw_mid} mid_norm (G): {mid_norm}")

    raw_final = raw_scaled / mid_norm
    print(f"raw_final (python): {raw_final}")

    # 6) Compare to C++
    cpp_raw = np.array([0.82472819, 0.83753932, 1.63281977])
    diff = np.abs(raw_final.flatten() - cpp_raw)
    print(f"\nCompare to C++ raw_center: {cpp_raw}")
    print(f"python raw: {raw_final.flatten()}")
    print(f"abs diff: {diff}; max: {diff.max():.6g}, mean: {diff.mean():.6g}")


if __name__ == '__main__':
    main()
