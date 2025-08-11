import os, json, numpy as np

def parametric_density_curves_model_py(log_exposure, gamma, log_exposure_0, density_max, toe_size, shoulder_size):
    le = np.asarray(log_exposure, dtype=float)
    le = le.reshape(-1)
    out = np.zeros((le.size, 3), dtype=float)
    for i, (g, le0, dmax, ts, ss) in enumerate(zip(gamma, log_exposure_0, density_max, toe_size, shoulder_size)):
        a = g*ts * np.log10(1 + 10**((le - le0)/ts))
        b = g*ss * np.log10(1 + 10**((le - le0 - dmax/g)/ss))
        out[:, i] = a - b
    return out

def main():
    cpp_path = os.path.abspath('cpp/tests/parametric/tmp_parametric_cpp.json')
    if not os.path.exists(cpp_path):
        raise SystemExit(f'Missing C++ JSON: {cpp_path}')
    with open(cpp_path) as f:
        data = json.load(f)

    le = np.array(data['log_exposure'], dtype=float)
    g  = np.array(data['gamma'], dtype=float)
    le0 = np.array(data['log_exposure_0'], dtype=float)
    dmax = np.array(data['density_max'], dtype=float)
    ts = np.array(data['toe_size'], dtype=float)
    ss = np.array(data['shoulder_size'], dtype=float)
    cpp = np.array(data['density_curves'], dtype=float)

    py = parametric_density_curves_model_py(le, g, le0, dmax, ts, ss)
    diff = np.abs(py - cpp)
    print('Parametric PY vs C++: max abs diff =', float(diff.max()))
    print('Mean abs diff =', float(diff.mean()))

if __name__ == '__main__':
    main()


