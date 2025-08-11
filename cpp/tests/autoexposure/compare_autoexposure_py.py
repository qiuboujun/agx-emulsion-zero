import os, json, numpy as np
from agx_emulsion.utils.autoexposure import measure_autoexposure_ev

def main():
    H, W = 128, 192
    img = np.zeros((H, W, 3), dtype=float)
    for y in range(H):
        for x in range(W):
            v = y / (H - 1) * 0.8 + x / (W - 1) * 0.2
            img[y, x, 0] = v
            img[y, x, 1] = v * 0.9
            img[y, x, 2] = v * 1.1

    # Python EV
    ev_py = measure_autoexposure_ev(img, color_space='sRGB', apply_cctf_decoding=True, method='center_weighted')

    # Load EVs reported by C++ test from stdout capture (optional) or recompute via file exchange
    print('EV Python:', float(ev_py))

if __name__ == '__main__':
    main()


