#!/usr/bin/env python3

import numpy as np
import scipy.ndimage
from scipy.ndimage import gaussian_filter
import math

def print_vector(vec, name, max_elements=20):
    """Print vector with limited elements"""
    print(f"{name}: [", end="")
    for i in range(min(len(vec), max_elements)):
        print(f"{vec[i]:.6f}", end="")
        if i < min(len(vec), max_elements) - 1:
            print(", ", end="")
    if len(vec) > max_elements:
        print(f", ... (showing first {max_elements} of {len(vec)} elements)", end="")
    print("]")

def print_image_stats(img, name):
    """Print image statistics"""
    if img.size == 0:
        print(f"{name}: empty")
        return
    
    min_val = np.min(img)
    max_val = np.max(img)
    mean_val = np.mean(img)
    print(f"{name} stats: min={min_val:.6f}, max={max_val:.6f}, mean={mean_val:.6f}, size={img.size}")

def apply_gaussian_blur_py(data, sigma):
    """Python implementation matching the C++ version"""
    if sigma > 0:
        return gaussian_filter(data, (sigma, sigma, 0))
    else:
        return data

def apply_gaussian_blur_um_py(data, sigma_um, pixel_size_um):
    """Python implementation matching the C++ version"""
    sigma = sigma_um / pixel_size_um
    if sigma > 0:
        return gaussian_filter(data, (sigma, sigma, 0))
    else:
        return data

def apply_unsharp_mask_py(image, sigma=0.0, amount=0.0):
    """Python implementation matching the C++ version"""
    image_blur = gaussian_filter(image, sigma=(sigma, sigma, 0))
    image_sharp = image + amount * (image - image_blur)
    return image_sharp

def apply_halation_um_py(raw, halation, pixel_size_um):
    """Python implementation matching the C++ version"""
    if not halation.active:
        return raw
    
    halation_size_pixel = np.array(halation.size_um) / pixel_size_um
    halation_strength = np.array(halation.strength)
    scattering_size_pixel = np.array(halation.scattering_size_um) / pixel_size_um
    scattering_strength = np.array(halation.scattering_strength)
    
    # Halation (truncate=7)
    for i in range(3):
        if halation_strength[i] > 0:
            raw[:,:,i] += halation_strength[i] * gaussian_filter(raw[:,:,i], halation_size_pixel[i], truncate=7)
            raw[:,:,i] /= (1 + halation_strength[i])
    
    # Scattering (truncate=7)
    for i in range(3):
        if scattering_strength[i] > 0:
            raw[:,:,i] += scattering_strength[i] * gaussian_filter(raw[:,:,i], scattering_size_pixel[i], truncate=7)
            raw[:,:,i] /= (1 + scattering_strength[i])
    
    return raw

class HalationParams:
    def __init__(self):
        self.active = False
        self.size_um = np.array([0.0, 0.0, 0.0])
        self.strength = np.array([0.0, 0.0, 0.0])
        self.scattering_size_um = np.array([0.0, 0.0, 0.0])
        self.scattering_strength = np.array([0.0, 0.0, 0.0])

def main():
    print("=== Python Diffusion Test Results ===")
    print()

    # Fixed test image (64x80x3 = 15360 elements)
    height = 64
    width = 80
    total_elements = height * width * 3
    
    # Create the exact same fixed, predictable input data as C++ (no random numbers)
    image_interleaved = np.zeros(total_elements, dtype=np.float32)
    for i in range(total_elements):
        # Create a predictable pattern: alternating values with some variation
        pixel = i // 3  # pixel index
        channel = i % 3  # RGB channel (0,1,2)
        
        # Base pattern: sine wave with different frequencies per channel
        base_val = 0.5 + 0.3 * math.sin(pixel * 0.1 + channel * 0.5)
        
        # Add some variation based on position
        variation = 0.1 * math.sin(pixel * 0.05) * math.cos(channel * 0.3)
        
        # Ensure values are in [0, 1] range
        image_interleaved[i] = max(0.0, min(1.0, base_val + variation))
    
    # Reshape to (height, width, 3) for processing
    image = image_interleaved.reshape(height, width, 3)
    
    print_image_stats(image_interleaved, "Input image")
    print_vector(image_interleaved, "Input image (first 20 elements)")
    
    print()

    # Test 1: Gaussian blur
    print("Test 1: apply_gaussian_blur")
    print("==========================")
    
    sigma = 2.0
    blurred = apply_gaussian_blur_py(image, sigma)
    blurred_interleaved = blurred.flatten()
    
    print_image_stats(blurred_interleaved, "Gaussian blurred image")
    print_vector(blurred_interleaved, "Gaussian blurred image (first 20 elements)")
    
    print()

    # Test 2: Gaussian blur with micrometres
    print("Test 2: apply_gaussian_blur_um")
    print("=============================")
    
    sigma_um = 3.25
    pixel_um = 2.5  # => sigma_px = 1.3
    blurred_um = apply_gaussian_blur_um_py(image, sigma_um, pixel_um)
    blurred_um_interleaved = blurred_um.flatten()
    
    print_image_stats(blurred_um_interleaved, "Gaussian blurred (um) image")
    print_vector(blurred_um_interleaved, "Gaussian blurred (um) image (first 20 elements)")
    
    print()

    # Test 3: Unsharp mask
    print("Test 3: apply_unsharp_mask")
    print("=========================")
    
    unsharp_sigma = 1.5
    unsharp_amount = 0.6
    unsharped = apply_unsharp_mask_py(image, unsharp_sigma, unsharp_amount)
    unsharped_interleaved = unsharped.flatten()
    
    print_image_stats(unsharped_interleaved, "Unsharp masked image")
    print_vector(unsharped_interleaved, "Unsharp masked image (first 20 elements)")
    
    print()

    # Test 4: Halation
    print("Test 4: apply_halation_um")
    print("=========================")
    
    # Create halation parameters
    halation = HalationParams()
    halation.active = True
    halation.size_um = np.array([5.0, 3.0, 2.0])
    halation.strength = np.array([0.10, 0.05, 0.00])
    halation.scattering_size_um = np.array([2.0, 0.0, 1.5])
    halation.scattering_strength = np.array([0.02, 0.00, 0.01])
    
    pixel_size_um = 2.5
    halated = image.copy()  # Copy input
    halated = apply_halation_um_py(halated, halation, pixel_size_um)
    halated_interleaved = halated.flatten()
    
    print_image_stats(halated_interleaved, "Halated image")
    print_vector(halated_interleaved, "Halated image (first 20 elements)")
    
    print()

    # Test 5: GPU vs CPU comparison
    print("Test 5: GPU vs CPU comparison")
    print("=============================")
    
    # For Python, we just use the CPU result as "GPU" result
    # In a real implementation, this would call the GPU version
    gpu_blurred = apply_gaussian_blur_py(image, sigma)
    gpu_blurred_interleaved = gpu_blurred.flatten()
    
    print_image_stats(gpu_blurred_interleaved, "GPU blurred image")
    print_vector(gpu_blurred_interleaved, "GPU blurred image (first 20 elements)")
    
    # Compare with CPU result (should be identical in Python)
    max_diff = np.max(np.abs(blurred_interleaved - gpu_blurred_interleaved))
    print(f"Max absolute difference (CPU vs GPU): {max_diff:.15f}")
    
    print()
    print("=== Test completed ===")

if __name__ == "__main__":
    main() 