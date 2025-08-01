#!/usr/bin/env python3
"""
Test Python measure_gamma function with the same data as C++.
"""

import numpy as np
from scipy.interpolate import interp1d

def measure_gamma(log_exposure, density_curves, density_0=0.25, density_1=1.0):
    gamma = np.zeros((3,))
    for i in range(3):
        loge0 = interp1d(density_curves[:, i], log_exposure, kind='cubic')(density_0)
        loge1 = interp1d(density_curves[:, i], log_exposure, kind='cubic')(density_1)
        gamma[i] = (density_1-density_0)/(loge1-loge0)
    return gamma

def main():
    # Use the same random seed as C++ for reproducible results
    np.random.seed(42)
    
    # Generate the same data as C++
    log_exposure = np.linspace(-2.0, 2.0, 20)
    density_curves = np.random.uniform(0.1, 1.3, (20, 3))
    
    density_0 = 0.25
    density_1 = 1.0
    
    print("=== Testing Python measure_gamma ===")
    print(f"Target densities: {density_0}, {density_1}")
    
    # Test Python measure_gamma function
    gamma_python = measure_gamma(log_exposure, density_curves, density_0, density_1)
    
    print(f"\nPython measure_gamma results:")
    print(f"  gamma[0] = {gamma_python[0]:.12f}")
    print(f"  gamma[1] = {gamma_python[1]:.12f}")
    print(f"  gamma[2] = {gamma_python[2]:.12f}")
    
    # Print the input data for C++ comparison
    print(f"\nInput data for C++ comparison:")
    print(f"log_exposure = {log_exposure}")
    print(f"density_curves = {density_curves}")

if __name__ == "__main__":
    main() 