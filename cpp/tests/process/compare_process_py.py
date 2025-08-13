import os
import numpy as np
from agx_emulsion.model.process import photo_params, photo_process
from agx_emulsion.utils.autoexposure import measure_autoexposure_ev

def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
    cpp_csv = os.path.join(repo_root, 'build', 'tmp_process_cpp.csv')
    if not os.path.exists(cpp_csv):
        print('Missing C++ CSV:', cpp_csv)
        return 2
    out_cpp = np.loadtxt(cpp_csv, delimiter=',')

    # Build same fixed image (5x5 gradient)
    H, W = 5, 5
    img = np.zeros((H, W, 3), dtype=float)
    for i in range(H):
        for j in range(W):
            v = float(i*W+j)/float(H*W-1)
            img[i,j,0] = v
            img[i,j,1] = 0.8*v
            img[i,j,2] = 1.2*v

    # Params matching C++ defaults
    params = photo_params(negative='kodak_portra_400_auc', print_paper='kodak_portra_endura_uc')
    params.io.input_color_space = 'sRGB'
    params.io.input_cctf_decoding = False
    params.io.output_color_space = 'sRGB'
    params.io.output_cctf_encoding = True
    params.camera.auto_exposure = True

    # Align Python EV with C++: use identical setting of apply_cctf_decoding=False
    ev_py = measure_autoexposure_ev(img, color_space='sRGB', apply_cctf_decoding=False, method='center_weighted')
    print('Python EV (apply_cctf_decoding=False):', ev_py)

    out_py = photo_process(img, params)

    # Flatten Python output to match CSV rows (H x W*3) style
    out_py_flat = out_py.reshape(H, -1)

    diff = np.abs(out_cpp - out_py_flat)
    print('max abs diff:', diff.max(), 'mean abs diff:', diff.mean())
    return 0

if __name__ == '__main__':
    raise SystemExit(main())
