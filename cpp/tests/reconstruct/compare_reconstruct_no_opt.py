import os
import json
import numpy as np

from agx_emulsion.profiles.io import load_profile
from agx_emulsion.profiles.reconstruct import (
    make_reconstruct_dye_density_params,
    density_mid_min_model,
)


def run_python_no_opt(stock: str = "kodak_portra_400_au"):
    profile = load_profile(stock)

    cmy_model = profile.data.dye_density[:, :3]
    wl = profile.data.wavelengths.flatten()
    params = make_reconstruct_dye_density_params('dmid_dmin')

    # Forward model with default params (no optimiser)
    cmy, dye, filters, dmin = density_mid_min_model(params, wl, cmy_model, 'dmid_dmin')

    # Normalise and compute midscale neutral as reconstruct.py does
    ms = np.nanmax(cmy, axis=0)
    cmy_norm = cmy / ms
    return {
        'midscale_neutral': ms.tolist(),
        'dye_density_head': cmy_norm[:10, :3].tolist(),
    }


def main():
    # Ensure C++ compact output exists
    os.system(os.path.abspath("../../build/independent_reconstruct_test") + " >/dev/null 2>&1")

    cpp_path = os.path.abspath("tmp_reconstruct_cmp.json")
    if not os.path.exists(cpp_path):
        alt = os.path.abspath("../../cpp/tests/reconstruct/tmp_reconstruct_cmp.json")
        if os.path.exists(alt):
            cpp_path = alt
    if not os.path.exists(cpp_path):
        raise SystemExit(f"Missing C++ compact output at {cpp_path}")

    py_out = run_python_no_opt("kodak_portra_400_au")
    with open(cpp_path) as f:
        cpp_out = json.load(f)

    py_mid = np.array(py_out['midscale_neutral'], dtype=float)
    cpp_mid = np.array(cpp_out['midscale_neutral'], dtype=float)

    py_head = np.array(py_out['dye_density_head'], dtype=float)
    cpp_head = np.array(cpp_out['dye_density_head'], dtype=float)

    mid_diff = np.abs(py_mid - cpp_mid)
    mask = ~(np.isnan(py_head) | np.isnan(cpp_head))
    head_diff = np.abs(py_head - cpp_head)
    head_diff_masked = np.where(mask, head_diff, np.nan)

    print('Midscale neutral PY:', py_mid)
    print('Midscale neutral C++:', cpp_mid)
    print('Abs diff mid:', mid_diff, 'max=', float(np.nanmax(mid_diff)))

    row_max = [float(np.nanmax(head_diff_masked[i])) for i in range(head_diff.shape[0])]
    print('\nDye density head max abs diff:', float(np.nanmax(head_diff_masked)))
    print('Row-wise max diffs:', row_max)


if __name__ == '__main__':
    main()


