import os
import numpy as np
from agx_emulsion.utils.spectral_upsampling import rgb_to_raw_hanatos2025, load_spectra_lut

def main():
    # Prefer repo-root build path where the C++ test writes now
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    root_build_csv = os.path.join(repo_root, 'build', 'tmp_rgb_to_raw_cpp.csv')
    # Fallback to previous cpp/build location
    cpp_build_csv = os.path.join(os.path.dirname(__file__), '..', '..', 'build', 'tmp_rgb_to_raw_cpp.csv')

    if os.path.exists(root_build_csv):
        cpp_csv = root_build_csv
    elif os.path.exists(cpp_build_csv):
        cpp_csv = cpp_build_csv
    else:
        print('Missing C++ CSV in both locations:', root_build_csv, 'and', cpp_build_csv)
        return 2

    raw_cpp = np.loadtxt(cpp_csv, delimiter=',')
    
    print(f"C++ output shape: {raw_cpp.shape}")
    print(f"C++ first row: {raw_cpp[0]}")
    print(f"C++ last row: {raw_cpp[-1]}")

    # Recreate the same RGB used in C++
    N = raw_cpp.shape[0]
    rgb = np.zeros((N,3), dtype=float)
    for i in range(N):
        v = 0.1*i
        rgb[i] = [v, v*0.8, v*1.2]
    
    print(f"Python RGB first row: {rgb[0]}")
    print(f"Python RGB last row: {rgb[-1]}")

    # Load sensitivity: trivial ones like C++ (K inferred from LUT)
    lut = load_spectra_lut()
    K = lut.shape[-1]
    sensitivity = np.ones((K,3), dtype=float)
    
    print(f"LUT shape: {lut.shape}")
    print(f"Sensitivity shape: {sensitivity.shape}")

    raw_py = rgb_to_raw_hanatos2025(rgb[None, ...], sensitivity,
                                    color_space='sRGB', apply_cctf_decoding=True,
                                    reference_illuminant='D65')[0]
    
    print(f"Python output shape: {raw_py.shape}")
    print(f"Python first row: {raw_py[0]}")
    print(f"Python last row: {raw_py[-1]}")

    diff = np.abs(raw_cpp - raw_py)
    print(f'max abs diff: {diff.max()}')
    print(f'mean abs diff: {diff.mean()}')
    
    # Show where the largest differences occur
    max_diff_idx = np.unravel_index(diff.argmax(), diff.shape)
    print(f'Max diff at position {max_diff_idx}: C++={raw_cpp[max_diff_idx]}, Python={raw_py[max_diff_idx]}')
    
    # Show differences by channel
    for c in range(3):
        channel_diff = diff[:, c]
        print(f'Channel {c}: max={channel_diff.max():.8f}, mean={channel_diff.mean():.8f}')
    
    return 0

if __name__ == '__main__':
    raise SystemExit(main())


