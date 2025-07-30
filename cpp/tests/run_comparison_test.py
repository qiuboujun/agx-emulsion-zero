# cpp/tests/run_comparison_test.py

import numpy as np
import numpy.testing as npt
import sys
import os

# Add the root of the project to the Python path to allow importing agx_emulsion
# This assumes the script is run from the 'cpp/tests' directory
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, project_root)

# --- Import the modules to compare ---

# 1. Import the original Python functions
from agx_emulsion.utils import spectral_upsampling as py_su
# Import config to get access to global data like SPECTRAL_SHAPE
from agx_emulsion import config as py_config


# 2. Import the C++ functions exposed via pybind11
# The name 'agx_cpp_tests' is what we defined in PYBIND11_MODULE
try:
    import agx_cpp_tests as cpp_su
except ImportError:
    print("Error: Could not import the C++ test module 'agx_cpp_tests'.")
    print("Please make sure you have built the project using CMake.")
    sys.exit(1)


def run_test(test_name, py_func, cpp_func, *args, **kwargs):
    """Helper function to run a single test and print results."""
    print(f"--- Testing: {test_name} ---")
    try:
        # Get results from both implementations
        py_result = py_func(*args, **kwargs)
        cpp_result = cpp_func(*args, **kwargs)

        # Handle functions that return tuples
        if isinstance(py_result, tuple):
            for i, (py_val, cpp_val) in enumerate(zip(py_result, cpp_result)):
                npt.assert_allclose(py_val, cpp_val, rtol=1e-4, atol=1e-4,
                                    err_msg=f"Mismatch in return value {i} for {test_name}")
        else:
            npt.assert_allclose(py_result, cpp_result, rtol=1e-4, atol=1e-4,
                                err_msg=f"Mismatch for {test_name}")

        print(f"[  OK  ] {test_name} passed.\n")
        return True
    except AssertionError as e:
        print(f"[ FAIL ] {test_name} failed!")
        print(e)
        # Uncomment for detailed debugging
        # if isinstance(py_result, tuple):
        #     for i, (py_val, cpp_val) in enumerate(zip(py_result, cpp_result)):
        #         print(f"PY result[{i}]:\n{py_val}")
        #         print(f"CPP result[{i}]:\n{cpp_val}")
        # else:
        #     print("Python result:\n", py_result)
        #     print("C++ result:\n", cpp_result)
        print("-" * 20)
        return False
    except Exception as e:
        print(f"[ ERROR ] {test_name} threw an exception!")
        print(e)
        return False

def main():
    """Main test runner."""
    print("="*50)
    print("Starting Comparison Test: Python vs. C++/CUDA")
    print("="*50)

    # --- Test Data ---
    # Create some common input data that can be used for multiple tests.
    # Using float32 is important as it's common in image processing and CUDA.
    np.random.seed(42) # for reproducible results
    test_coords_2d = np.array([[0.1, 0.2], [0.5, 0.5], [0.9, 0.8]], dtype=np.float32)
    test_rgb_image = np.random.rand(10, 5, 3).astype(np.float32) * 0.8 + 0.1
    test_rgb_pixel = np.array([[[0.2, 0.4, 0.8]]], dtype=np.float32)
    
    # A sample sensitivity matrix, shape must match spectral shape
    num_wavelengths = py_config.SPECTRAL_SHAPE.wavelengths.shape[0]
    test_sensitivity = np.random.rand(num_wavelengths, 3).astype(np.float32)
    
    # --- Run Tests One by One ---

    # Test 1: tri2quad
    run_test("tri2quad", py_su.tri2quad, cpp_su.tri2quad_cpp, tc=test_coords_2d)

    # Test 2: quad2tri
    run_test("quad2tri", py_su.quad2tri, cpp_su.quad2tri_cpp, xy=test_coords_2d)
    
    # Test 3: rgb_to_tc_b
    run_test(
        "rgb_to_tc_b",
        py_su.rgb_to_tc_b,
        cpp_su.rgb_to_tc_b_cpp,
        rgb=test_rgb_image,
        color_space='sRGB',
        apply_cctf_decoding=True,
        reference_illuminant='D65'
    )

    # Test 4: illuminant_to_xy
    run_test("illuminant_to_xy", py_su.illuminant_to_xy, cpp_su.illuminant_to_xy_cpp, illuminant_label='D65')

    # Test 5: compute_band_pass_filter
    run_test("compute_band_pass_filter", py_su.compute_band_pass_filter, cpp_su.compute_band_pass_filter_cpp)

    # Test 6: rgb_to_spectrum (single pixel)
    run_test(
        "rgb_to_spectrum",
        py_su.rgb_to_spectrum,
        cpp_su.rgb_to_spectrum_cpp,
        rgb=test_rgb_pixel,
        color_space='sRGB',
        apply_cctf_decoding=True,
        reference_illuminant='D50'
    )

    # Test 7: rgb_to_raw_mallett2019
    run_test(
        "rgb_to_raw_mallett2019",
        py_su.rgb_to_raw_mallett2019,
        cpp_su.rgb_to_raw_mallett2019_cpp,
        RGB=test_rgb_image,
        sensitivity=test_sensitivity,
        color_space='sRGB',
        apply_cctf_decoding=True,
        reference_illuminant='D65'
    )

    # Test 8: rgb_to_raw_hanatos2025 (image)
    # This requires the LUT to be loaded first
    py_su.HANATOS2025_SPECTRA_LUT = py_su.load_spectra_lut()
    run_test(
        "rgb_to_raw_hanatos2025 (image)",
        py_su.rgb_to_raw_hanatos2025,
        cpp_su.rgb_to_raw_hanatos2025_cpp,
        rgb=test_rgb_image,
        sensitivity=test_sensitivity,
        color_space='ITU-R BT.709',
        apply_cctf_decoding=False,
        reference_illuminant='D65'
    )
    
    # Test 9: rgb_to_raw_hanatos2025 (single pixel)
    run_test(
        "rgb_to_raw_hanatos2025 (pixel)",
        py_su.rgb_to_raw_hanatos2025,
        cpp_su.rgb_to_raw_hanatos2025_cpp,
        rgb=test_rgb_pixel,
        sensitivity=test_sensitivity,
        color_space='sRGB',
        apply_cctf_decoding=True,
        reference_illuminant='D55'
    )
    
    # Note: Testing load_coeffs_lut and other file I/O is implicitly done
    # when the functions that depend on them are tested. You could add
    # explicit tests if you want to verify file parsing in isolation.

    print("="*50)
    print("Comparison Test Finished")
    print("="*50)


if __name__ == "__main__":
    main()
