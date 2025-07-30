import numpy as np
import numpy.testing as npt
import sys
import os

# --- Path Setup ---
# Get the directory of the current script
script_dir = os.path.dirname(__file__)
# Go up three levels to reach the project root (e.g., from cpp/tests/fast_interp_lut/ to the root)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
# Add the project root to Python's path
sys.path.insert(0, project_root)

from agx_emulsion.utils import fast_interp_lut as py_interp

try:
    # This is the C++ module we build with CMake
    import fast_interp_cpp_tests as cpp_interp
except ImportError:
    print("Error: Could not import 'fast_interp_cpp_tests'.")
    print("Please build the C++ test module first using CMake.")
    sys.exit(1)


def run_test(test_name, py_func, cpp_func, *args, **kwargs):
    """Helper function to run a single test and print results."""
    print(f"--- Testing: {test_name} ---")
    try:
        py_result = np.array(py_func(*args, **kwargs))
        cpp_result = np.array(cpp_func(*args, **kwargs))
        
        npt.assert_allclose(py_result, cpp_result, rtol=1e-5, atol=1e-5)
        print(f"[  OK  ] {test_name} passed.\n")
    except AssertionError as e:
        print(f"[ FAIL ] {test_name} failed!")
        print(e)
        # print("Python result:\n", py_result)
        # print("C++ result:\n", cpp_result)
        print("-" * 20)

def main():
    print("="*50)
    print("Starting fast_interp_lut Comparison Test")
    print("="*50)

    # --- Test Data ---
    L_3D, L_2D = 32, 64
    height, width = 128, 256
    
    # 3D Data
    lut_3d = np.random.rand(L_3D, L_3D, L_3D, 3).astype(np.float32)
    image_3d = np.random.rand(height, width, 3).astype(np.float32)

    # 2D Data
    lut_2d = np.random.rand(L_2D, L_2D, 5).astype(np.float32) # 5 channels
    image_2d_coords = np.random.rand(height, width, 2).astype(np.float32)

    # --- Run Tests ---
    
    # Test single-point 2D interpolation
    run_test("cubic_interp_lut_at_2d",
             py_interp.cubic_interp_lut_at_2d,
             cpp_interp.cubic_interp_lut_at_2d_cpp,
             lut=lut_2d, x=15.3, y=40.8)

    # Test single-point 3D interpolation
    run_test("cubic_interp_lut_at_3d",
             py_interp.cubic_interp_lut_at_3d,
             cpp_interp.cubic_interp_lut_at_3d_cpp,
             lut=lut_3d, r=10.1, g=20.5, b=5.7)

    # Test full image 2D interpolation
    run_test("apply_lut_cubic_2d (GPU)",
             py_interp.apply_lut_cubic_2d,
             cpp_interp.apply_lut_cubic_2d_cpp,
             lut=lut_2d, image=image_2d_coords)

    # Test full image 3D interpolation
    run_test("apply_lut_cubic_3d (GPU)",
             py_interp.apply_lut_cubic_3d,
             cpp_interp.apply_lut_cubic_3d_cpp,
             lut=lut_3d, image=image_3d)

    print("="*50)
    print("Comparison Test Finished")
    print("="*50)

if __name__ == "__main__":
    main()
