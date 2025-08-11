import os
import re
import subprocess
import numpy as np
from agx_emulsion.utils.autoexposure import measure_autoexposure_ev


def make_image(H=128, W=192):
    img = np.zeros((H, W, 3), dtype=float)
    for y in range(H):
        for x in range(W):
            v = y / (H - 1) * 0.8 + x / (W - 1) * 0.2
            img[y, x, 0] = v
            img[y, x, 1] = v * 0.9
            img[y, x, 2] = v * 1.1
    return img


def main():
    # Python EV
    img = make_image()
    ev_py = measure_autoexposure_ev(img, color_space='sRGB', apply_cctf_decoding=True, method='center_weighted')

    # C++ EV: run the binary (it generates the same pattern internally)
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    bin_path = os.path.join(root, 'build', 'independent_autoexposure_test')
    if not os.path.exists(bin_path):
        raise SystemExit(f"Missing binary: {bin_path}")
    out = subprocess.check_output([bin_path], text=True)
    # Parse lines like: EV CPU:  -0.303601, EV AUTO: -0.303601
    m_cpu = re.search(r"EV CPU:\s*([-+eE0-9\.-]+)", out)
    m_auto = re.search(r"EV AUTO:\s*([-+eE0-9\.-]+)", out)
    if not (m_cpu and m_auto):
        raise SystemExit(f"Unexpected binary output:\n{out}")
    ev_cpu = float(m_cpu.group(1))
    ev_auto = float(m_auto.group(1))

    print("EV Python:", float(ev_py))
    print("EV C++ (CPU):", ev_cpu)
    print("EV C++ (AUTO):", ev_auto)
    print("Abs diff Python vs C++ CPU:", abs(float(ev_py) - ev_cpu))
    print("Abs diff Python vs C++ AUTO:", abs(float(ev_py) - ev_auto))


if __name__ == '__main__':
    main()


