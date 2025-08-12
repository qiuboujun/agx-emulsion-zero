import os
import numpy as np
from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025, load_spectra_lut

def main():
    cpp_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'tmp_rgb_to_raw_cpp.csv')
    if not os.path.exists(cpp_csv):
        print('Missing C++ CSV:', cpp_csv)
        return 2
    raw_cpp = np.loadtxt(cpp_csv, delimiter=',')

    # Recreate the same RGB used in C++
    N = raw_cpp.shape[0]
    rgb = np.zeros((N,3), dtype=float)
    for i in range(N):
        v = 0.1*i
        rgb[i] = [v, v*0.8, v*1.2]

    # Load sensitivity: trivial ones like C++ (K inferred from LUT)
    lut = load_spectra_lut()
    K = lut.shape[-1]
    sensitivity = np.ones((K,3), dtype=float)

    raw_py = rgb_to_raw_hanatos2025(rgb[None, ...], sensitivity,
                                    color_space='sRGB', apply_cctf_decoding=True,
                                    reference_illuminant='D65')[0]

    diff = np.abs(raw_cpp - raw_py)
    print('max abs diff:', diff.max(), 'mean abs diff:', diff.mean())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())


