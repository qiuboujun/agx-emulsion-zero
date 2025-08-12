import os
import numpy as np
from agx_emulsion.utils.spectral_upsampling import rgb_to_tc_b, load_spectra_lut, rgb_to_raw_hanatos2025
from agx_emulsion.utils.fast_interp_lut import apply_lut_cubic_2d
from scipy.interpolate import RegularGridInterpolator


def main():
    # Same inputs as C++ test
    N = 5
    rgb = np.zeros((N, 3), dtype=float)
    for i in range(N):
        v = 0.1 * i
        rgb[i] = [v, v * 0.8, v * 1.2]

    lut = load_spectra_lut()  # shape (L, L, K)
    L, _, K = lut.shape
    sensitivity = np.ones((K, 3), dtype=float)

    # Compute tc, b
    tc, b = rgb_to_tc_b(rgb[None, ...], color_space='sRGB', apply_cctf_decoding=True, reference_illuminant='D65')
    tc = tc[0]
    b = b[0]
    print(f"py tc last: {tc[-1]}")
    print(f"py b last: {b[-1]}")

    # Preproject LUT by sensitivity -> (L,L,3)
    proj = np.tensordot(lut, sensitivity, axes=([2], [0]))  # (L, L, 3)

    # Sample projected LUT at tc (cubic Mitchell)
    tc_img = tc[None, ...]
    raw_proj = apply_lut_cubic_2d(proj, tc_img)[0]  # (N, 3)

    # Multiply by b per pixel
    raw_pre = raw_proj * b[..., None]
    print(f"py raw_pre_norm last: {raw_pre[-1]}")

    # Midgray normalization using linear interpolation of spectra lut
    midgray_rgb = np.array([[[0.184, 0.184, 0.184]]], dtype=float)
    tc_m, b_m = rgb_to_tc_b(midgray_rgb, color_space='sRGB', apply_cctf_decoding=False, reference_illuminant='D65')
    v = np.linspace(0, 1, L)
    interp = RegularGridInterpolator((v, v), lut, method='linear', bounds_error=False, fill_value=None)
    spec_mid = interp(tc_m[0, 0]) * b_m[0, 0]
    spec_mid = spec_mid.reshape(-1)
    raw_mid = np.einsum('k,km->m', spec_mid, sensitivity)
    scale = 1.0 / max(1e-10, raw_mid[1])
    print(f"py midgray scale: {scale}")

    print(f"py raw_post_norm last (pred): {raw_pre[-1] * scale}")

    # Full function output for confirmation
    raw_full = rgb_to_raw_hanatos2025(rgb[None, ...], sensitivity,
                                      color_space='sRGB', apply_cctf_decoding=True,
                                      reference_illuminant='D65')[0]
    print(f"py RAW last row (full): {raw_full[-1]}")


if __name__ == '__main__':
    main()
