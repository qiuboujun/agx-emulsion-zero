import numpy as np
import numpy.testing as npt
import sys
import os

# --- Path Setup ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, project_root)

# 1. Import the original Python config module
from agx_emulsion import config as py_config

# 2. Import the C++ test module
try:
    import config_cpp_tests as cpp_config
except ImportError:
    print("Error: Could not import 'config_cpp_tests'. Build the C++ test module first.")
    sys.exit(1)

def run_test(test_name, py_value, cpp_value):
    """Helper function to run a single comparison and print results."""
    print(f"--- Testing: {test_name} ---")
    try:
        npt.assert_allclose(py_value, cpp_value, rtol=1e-6, atol=1e-6)
        print(f"[  OK  ] {test_name} passed.\n")
    except AssertionError as e:
        print(f"[ FAIL ] {test_name} failed!")
        print(e)

def main():
    print("="*50)
    print("Starting config.py vs config.hpp Comparison Test")
    print("="*50)

    # Initialize the C++ config module
    # This is a crucial step that populates the global variables on the C++ side.
    cpp_config.initialize_config_cpp()
    print("C++ config module initialized.\n")

    # --- Run Tests ---

    # Test 1: LOG_EXPOSURE
    py_log_exp = py_config.LOG_EXPOSURE
    cpp_log_exp = cpp_config.get_log_exposure_cpp()
    run_test("LOG_EXPOSURE", py_log_exp, cpp_log_exp)

    # Test 2: SPECTRAL_SHAPE wavelengths
    py_wl = py_config.SPECTRAL_SHAPE.wavelengths
    cpp_wl = cpp_config.get_spectral_shape_wavelengths_cpp()
    run_test("SPECTRAL_SHAPE.wavelengths", py_wl, cpp_wl)

    # Test 3: STANDARD_OBSERVER_CMFS
    py_cmfs = py_config.STANDARD_OBSERVER_CMFS
    cpp_cmfs = cpp_config.get_standard_observer_cmfs_cpp()
    
    # Debug output
    print(f"Python CMFS shape: {py_cmfs.shape}")
    print(f"C++ CMFS shape: {cpp_cmfs.shape}")
    print(f"Python CMFS first 3x3: {py_cmfs.values[:3, :3]}")
    print(f"C++ CMFS first 3x3: {cpp_cmfs[:3, :3]}")
    print(f"Python has NaN: {np.isnan(py_cmfs.values).any()}")
    print(f"C++ has NaN: {np.isnan(cpp_cmfs).any()}")
    
    # Calculate and show differences
    abs_diff = np.abs(py_cmfs.values - cpp_cmfs)
    
    # Calculate relative difference - C++ should handle zeros properly now
    rel_diff = np.abs(py_cmfs.values - cpp_cmfs) / np.abs(py_cmfs.values)
    # Handle division by zero by setting to 0
    rel_diff = np.where(np.abs(py_cmfs.values) < 1e-10, 0, rel_diff)
    
    # Debug zero values specifically
    zero_mask = np.abs(py_cmfs.values) < 1e-10
    if np.any(zero_mask):
        print(f"Zero values in Python: {np.sum(zero_mask)}")
        print(f"Corresponding C++ values for zeros: {cpp_cmfs[zero_mask]}")
        print(f"Max C++ value where Python is zero: {np.max(np.abs(cpp_cmfs[zero_mask]))}")
    
    print(f"Max absolute difference: {np.max(abs_diff):.10f}")
    print(f"Max relative difference: {np.max(rel_diff):.10f}")
    print(f"Mean absolute difference: {np.mean(abs_diff):.10f}")
    print(f"Mean relative difference: {np.mean(rel_diff):.10f}")
    print(f"Number of zero/near-zero values: {np.sum(zero_mask)}")
    
    run_test("STANDARD_OBSERVER_CMFS", py_cmfs.values, cpp_cmfs)

    print("="*50)
    print("Comparison Test Finished")
    print("="*50)

if __name__ == "__main__":
    main()
