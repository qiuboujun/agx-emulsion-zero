import numpy as np
import numpy.testing as npt
import sys
import os
import json

# --- Path Setup ---
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, project_root)

from agx_emulsion.utils import io as py_io
try:
    import io_cpp_tests as cpp_io
except ImportError:
    print("Error: Could not import 'io_cpp_tests'. Build the C++ test module first.")
    sys.exit(1)

def run_test(test_name, py_func, cpp_func, args=(), kwargs={}):
    """Helper function to run a single test and print results."""
    print(f"--- Testing: {test_name} ---")
    try:
        py_result = py_func(*args, **kwargs)
        cpp_result = cpp_func(*args, **kwargs)
        
        # Special handling for different return types
        if test_name == "load_agx_emulsion_data":
            py_keys = ["log_sensitivity", "dye_density", "wavelengths", "density_curves", "log_exposure"]
            py_result_dict = dict(zip(py_keys, py_result))
            for key in py_keys:
                print(f"  - Comparing key: {key}")
                npt.assert_allclose(py_result_dict[key], cpp_result[key], rtol=1e-5, atol=1e-5, err_msg=f"Mismatch for key '{key}'")
        elif test_name == "read_neutral_ymc_filter_values":
            # C++ returns JSON string, Python returns dict - parse C++ result
            cpp_dict = json.loads(cpp_result)
            assert py_result == cpp_dict, f"JSON data mismatch"
        else:
             npt.assert_allclose(py_result, cpp_result, rtol=1e-5, atol=1e-5)
        print(f"[  OK  ] {test_name} passed.\n")
    except AssertionError as e:
        print(f"[ FAIL ] {test_name} failed!")
        print(e)
    except Exception as e:
        print(f"[ ERROR ] An exception occurred in {test_name}: {e}")


def main():
    print("="*50)
    print("Starting io.py vs io.cpp Comparison Test")
    print("="*50)

    # --- Test Data ---
    test_data = np.array([[1, 2, 4, 5], [10, 20, 40, 50]], dtype=np.float32)
    new_x = np.linspace(0, 6, 10, dtype=np.float32)

    # --- Run Tests ---
    run_test("interpolate_to_common_axis",
             py_io.interpolate_to_common_axis,
             cpp_io.interpolate_to_common_axis_cpp,
             kwargs={'data': test_data, 'new_x': new_x, 'method': 'akima'})

    run_test("read_neutral_ymc_filter_values",
             py_io.read_neutral_ymc_filter_values,
             cpp_io.read_neutral_ymc_filter_values_cpp)

    run_test("load_densitometer_data",
             py_io.load_densitometer_data,
             cpp_io.load_densitometer_data_cpp)
    
    run_test("load_dichroic_filters",
             py_io.load_dichroic_filters,
             cpp_io.load_dichroic_filters_cpp,
             # Python version needs wavelengths explicitly passed for this test
             kwargs={'wavelengths': py_io.SPECTRAL_SHAPE.wavelengths, 'brand': 'thorlabs'})

    run_test("load_filter",
             py_io.load_filter,
             cpp_io.load_filter_cpp,
             kwargs={'wavelengths': py_io.SPECTRAL_SHAPE.wavelengths, 'name': 'KG5'})

    # This is the most comprehensive integration test
    run_test("load_agx_emulsion_data",
             py_io.load_agx_emulsion_data,
             cpp_io.load_agx_emulsion_data_cpp,
             kwargs={'stock': 'kodak_gold_200'})
    
    print("="*50)
    print("Comparison Test Finished")
    print("="*50)

if __name__ == "__main__":
    main()
