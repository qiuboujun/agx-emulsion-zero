#!/usr/bin/env python3

import numpy as np
import sys
import os

# Add the agx_emulsion directory to the path
sys.path.insert(0, 'agx_emulsion')

from agx_emulsion.utils.io import load_agx_emulsion_data

def save_array_to_file(arr, filename, name):
    """Save a numpy array to a text file with a descriptive header"""
    with open(filename, 'w') as f:
        f.write(f"# {name}\n")
        f.write(f"# Shape: {arr.shape}\n")
        f.write(f"# dtype: {arr.dtype}\n")
        f.write("# Data:\n")
        
        if len(arr.shape) == 1:
            for val in arr:
                f.write(f"{val}\n")
        elif len(arr.shape) == 2:
            for row in arr:
                f.write(" ".join(f"{val}" for val in row) + "\n")

def main():
    print("Testing Python load_agx_emulsion_data...")
    
    # Test parameters - using kodak_vision3_500t which has all required files
    stock = "kodak_vision3_500t"
    log_sensitivity_donor = ""  # empty string like C++
    density_curves_donor = ""   # Note: Python has typo "denisty_curves_donor"
    dye_density_cmy_donor = ""
    dye_density_min_mid_donor = ""
    type_val = "negative"
    color = True
    
    try:
        # Call the function with exact same parameters as C++ version
        # Note: Python version has the typo "denisty_curves_donor" in its signature
        result = load_agx_emulsion_data(
            stock=stock,
            log_sensitivity_donor=None if log_sensitivity_donor == "" else log_sensitivity_donor,
            denisty_curves_donor=None if density_curves_donor == "" else density_curves_donor,  # Using the typo as it exists in Python
            dye_density_cmy_donor=None if dye_density_cmy_donor == "" else dye_density_cmy_donor,
            dye_density_min_mid_donor=None if dye_density_min_mid_donor == "" else dye_density_min_mid_donor,
            type=type_val,
            color=color
            # Note: Python version has additional spectral_shape and log_exposure parameters
            # that C++ version doesn't have - these use defaults
        )
        
        # Unpack the returned tuple
        log_sensitivity, dye_density, wavelengths, density_curves, log_exposure = result
        
        print(f"Log sensitivity shape: {log_sensitivity.shape}")
        print(f"Dye density shape: {dye_density.shape}")
        print(f"Wavelengths shape: {wavelengths.shape}")
        print(f"Density curves shape: {density_curves.shape}")
        print(f"Log exposure shape: {log_exposure.shape}")
        
        # Save results to files
        save_array_to_file(log_sensitivity, "python_log_sensitivity.txt", "Log Sensitivity")
        save_array_to_file(dye_density, "python_dye_density.txt", "Dye Density")
        save_array_to_file(wavelengths, "python_wavelengths.txt", "Wavelengths")
        save_array_to_file(density_curves, "python_density_curves.txt", "Density Curves")
        save_array_to_file(log_exposure, "python_log_exposure.txt", "Log Exposure")
        
        print("Python results saved to files with 'python_' prefix")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 