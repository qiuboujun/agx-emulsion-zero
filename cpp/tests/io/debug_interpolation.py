import numpy as np
import sys
import os

# Add the project root to the path
script_dir = os.path.dirname(__file__)
project_root = os.path.abspath(os.path.join(script_dir, '..', '..', '..'))
sys.path.insert(0, project_root)

from agx_emulsion.utils import io as py_io

# Test data
test_data = np.array([
    [1, 3, 5, 7, 9],     # x values (5 distinct points)
    [10, 30, 50, 70, 90]  # y values
], dtype=np.float32)
new_x = np.linspace(0, 10, 10, dtype=np.float32)

print("Test data shape:", test_data.shape)
print("Test data:")
print(test_data)
print("New x shape:", new_x.shape)
print("New x:", new_x)

try:
    print("\nTesting Python interpolation...")
    result = py_io.interpolate_to_common_axis(test_data, new_x, method='akima')
    print("Python result shape:", result.shape)
    print("Python result:", result)
except Exception as e:
    print("Python error:", e) 