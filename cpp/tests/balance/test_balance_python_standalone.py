import os
import sys
import numpy as np
import numpy.testing as npt

# Ensure project root is on path
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from agx_emulsion.profiles.io import load_profile
from agx_emulsion.profiles.balance import balance_sensitivity as py_balance_sensitivity
from agx_emulsion.profiles.balance import balance_density as py_balance_density
from agx_emulsion.profiles.balance import balance_metameric_neutral as py_balance_metameric
from agx_emulsion.model.illuminants import standard_illuminant

try:
    # The test modules are written to cpp/tests/<module> directories, so add them to sys.path
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'cpp', 'tests', 'config'))
    sys.path.insert(0, os.path.join(PROJECT_ROOT, 'cpp', 'tests', 'balance'))
    import config_cpp_tests as cpp_config
    import balance_cpp_tests as cpp_balance
except ImportError as e:
    print("Error: required C++ test modules not built or not on sys.path:", e)
    sys.exit(1)

# Initialize C++ global spectral config (LOG_EXPOSURE, CMFS)
# Ensure working directory is project root so relative data paths resolve
os.chdir(PROJECT_ROOT)
cpp_config.initialize_config_cpp()


def to_arrays(p):
    d = p.data
    return (
        np.array(d.log_sensitivity, dtype=np.float32),
        np.array(d.dye_density, dtype=np.float32),
        np.array(d.wavelengths, dtype=np.float32).reshape((-1, 1)),
        np.array(d.density_curves, dtype=np.float32),
        np.array(d.log_exposure, dtype=np.float32).reshape((-1, 1)),
    )


def test_balance_sensitivity_and_density():
    p = load_profile('kodak_portra_400_au')
    ls0, dd0, wl0, dc0, le0 = to_arrays(p)

    # Python path
    p_py = load_profile('kodak_portra_400_au')
    p_py = py_balance_sensitivity(p_py, correct_log_exposure=True)
    p_py = py_balance_density(p_py)

    # C++ path
    ls_cpp, dc_cpp = cpp_balance.balance_sensitivity(ls0, dd0, wl0, dc0, le0, p.info.reference_illuminant, True)
    ls_cpp2, dc_cpp2 = cpp_balance.balance_density(ls_cpp, dd0, wl0, dc_cpp, le0)

    npt.assert_allclose(ls_cpp2, np.array(p_py.data.log_sensitivity, dtype=np.float32), rtol=1e-4, atol=1e-4, equal_nan=True)
    npt.assert_allclose(dc_cpp2, np.array(p_py.data.density_curves, dtype=np.float32), rtol=1e-3, atol=1e-3)


def test_balance_metameric_neutral():
    p = load_profile('kodak_portra_400_au')
    dd = np.array(p.data.dye_density, dtype=np.float32)

    # Python
    p_py = load_profile('kodak_portra_400_au')
    p_py = py_balance_metameric(p_py, midgray_value=0.184)

    # Use Python illuminant for exact parity
    ill_py = standard_illuminant(p.info.viewing_illuminant).astype(np.float32).reshape((-1,1))
    # C++
    dd_out_cpp, d_met_cpp, d_scale_cpp = cpp_balance.balance_metameric_neutral_with_illuminant(dd, ill_py, 0.184)

    npt.assert_allclose(dd_out_cpp[:, :3], np.array(p_py.data.dye_density, dtype=np.float32)[:, :3], rtol=2e-5, atol=2e-5)
    npt.assert_allclose(dd_out_cpp[:, 4], np.array(p_py.data.dye_density, dtype=np.float32)[:, 4], rtol=2e-5, atol=2e-5)


if __name__ == '__main__':
    test_balance_sensitivity_and_density()
    test_balance_metameric_neutral()
    print('All balance tests passed')


