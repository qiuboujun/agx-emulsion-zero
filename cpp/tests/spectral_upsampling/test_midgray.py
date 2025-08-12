import numpy as np
from agx_emulsion.utils.spectral_upsampling import rgb_to_tc_b, load_spectra_lut
import scipy.interpolate

def test_midgray_normalization():
    # Test the exact midgray case that's used in normalization
    midgray_rgb = np.array([[[0.184]*3]])
    
    # Get the tc coordinates and brightness
    tc, b = rgb_to_tc_b(midgray_rgb, color_space='sRGB', apply_cctf_decoding=False, reference_illuminant='D65')
    print(f"Midgray tc shape: {tc.shape}")
    print(f"Midgray tc: {tc}, brightness: {b}")
    
    # Load the spectra LUT
    spectra_lut = load_spectra_lut()
    print(f"Spectra LUT shape: {spectra_lut.shape}")
    
    # Python's approach: RegularGridInterpolator with linear method
    v = np.linspace(0, 1, spectra_lut.shape[0])
    interpolator = scipy.interpolate.RegularGridInterpolator((v, v), spectra_lut, method='linear', bounds_error=False, fill_value=None)
    
    # Get the spectrum using Python's interpolation
    spectrum_py = interpolator(tc[0]) * b[0]
    print(f"Python spectrum shape: {spectrum_py.shape}")
    print(f"Python spectrum first 5 values: {spectrum_py[:5]}")
    print(f"Python spectrum sum: {spectrum_py.sum()}")
    
    # Flatten the spectrum to match expected shape
    spectrum_py = spectrum_py.flatten()
    print(f"Python spectrum flattened shape: {spectrum_py.shape}")
    
    # Create a trivial sensitivity for testing
    sensitivity = np.ones((spectra_lut.shape[-1], 3), dtype=float)
    
    # Project with sensitivity
    raw_mid_py = np.einsum('k,km->m', spectrum_py, sensitivity)
    print(f"Python raw_mid: {raw_mid_py}")
    print(f"Python normalization factor (green): {raw_mid_py[1]}")
    
    # Now let's test what happens with the same coordinates in our C++ approach
    # The C++ code uses bilinear_interp_lut_at_2d_channels
    # Handle the case where tc might have different dimensions
    if tc.ndim == 3:
        x = tc[0, 0, 0] * (spectra_lut.shape[0] - 1)
        y = tc[0, 0, 1] * (spectra_lut.shape[0] - 1)
    else:
        x = tc[0, 0] * (spectra_lut.shape[0] - 1)
        y = tc[0, 1] * (spectra_lut.shape[0] - 1)
    print(f"C++ coordinates: x={x}, y={y}")
    
    # Manual bilinear interpolation to match C++
    L = spectra_lut.shape[0]
    x0, y0 = int(np.floor(x)), int(np.floor(y))
    x1, y1 = min(x0 + 1, L - 1), min(y0 + 1, L - 1)
    fx, fy = x - x0, y - y0
    
    print(f"Bilinear coordinates: x0={x0}, y0={y0}, x1={x1}, y1={y1}")
    print(f"Bilinear fractions: fx={fx:.6f}, fy={fy:.6f}")
    
    # Get the four corner values
    v00 = spectra_lut[x0, y0, :]
    v10 = spectra_lut[x1, y0, :]
    v01 = spectra_lut[x0, y1, :]
    v11 = spectra_lut[x1, y1, :]
    
    # Bilinear weights
    w00 = (1 - fx) * (1 - fy)
    w10 = fx * (1 - fy)
    w01 = (1 - fx) * fy
    w11 = fx * fy
    
    print(f"Bilinear weights: w00={w00:.6f}, w10={w10:.6f}, w01={w01:.6f}, w11={w11:.6f}")
    
    # Interpolate
    spectrum_cpp = (w00 * v00 + w10 * v10 + w01 * v01 + w11 * v11) * b[0]
    print(f"C++ spectrum first 5 values: {spectrum_cpp[:5]}")
    print(f"C++ spectrum sum: {spectrum_cpp.sum()}")
    
    # Project with sensitivity
    raw_mid_cpp = np.einsum('k,km->m', spectrum_cpp, sensitivity)
    print(f"C++ raw_mid: {raw_mid_cpp}")
    print(f"C++ normalization factor (green): {raw_mid_cpp[1]}")
    
    # Compare
    print(f"Spectrum difference: max={np.abs(spectrum_py - spectrum_cpp).max():.8f}")
    print(f"Raw_mid difference: max={np.abs(raw_mid_py - raw_mid_cpp).max():.8f}")
    
    # Show the actual values being compared
    print(f"Python normalization factor: {raw_mid_py[1]:.8f}")
    print(f"C++ normalization factor: {raw_mid_cpp[1]:.8f}")
    print(f"Difference in normalization: {abs(raw_mid_py[1] - raw_mid_cpp[1]):.8f}")

if __name__ == '__main__':
    test_midgray_normalization()
