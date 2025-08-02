#!/usr/bin/env python3
"""
Final test script to compare Python and C++ color_filters implementations.
"""

import numpy as np
import scipy.special
import sys
import os

# Add the agx_emulsion package to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'agx_emulsion'))

from agx_emulsion.config import SPECTRAL_SHAPE
from agx_emulsion.model.color_filters import (
    sigmoid_erf, create_combined_dichroic_filter, filterset, 
    compute_band_pass_filter, DichroicFilters, GenericFilter, color_enlarger
)

def test_sigmoid_erf():
    """Test sigmoid_erf function"""
    print("=== Testing sigmoid_erf ===")
    
    # Test data
    x = np.array([400, 450, 500, 550, 600, 650, 700], dtype=np.float32)
    center = 550.0
    width = 10.0
    
    # Python implementation
    result_py = sigmoid_erf(x, center, width)
    
    print(f"Input x: {x}")
    print(f"Center: {center}, Width: {width}")
    print(f"Python result: {result_py}")
    print()

def test_create_combined_dichroic_filter():
    """Test create_combined_dichroic_filter function"""
    print("=== Testing create_combined_dichroic_filter ===")
    
    # Test data
    wavelength = np.array([400, 450, 500, 550, 600, 650, 700], dtype=np.float32)
    filtering_amount_percent = [50.0, 30.0, 20.0]
    transitions = [10.0, 10.0, 10.0, 10.0]
    edges = [510.0, 495.0, 605.0, 590.0]
    nd_filter = 5.0
    
    # Python implementation
    result_py = create_combined_dichroic_filter(
        wavelength, filtering_amount_percent, transitions, edges, nd_filter
    )
    
    print(f"Wavelength: {wavelength}")
    print(f"Filtering amount: {filtering_amount_percent}")
    print(f"Transitions: {transitions}")
    print(f"Edges: {edges}")
    print(f"ND filter: {nd_filter}")
    print(f"Python result shape: {result_py.shape}")
    print(f"Python result: {result_py}")
    print()

def test_filterset():
    """Test filterset function"""
    print("=== Testing filterset ===")
    
    # Test data - create a SpectralDistribution with the full spectral shape
    import colour
    illuminant_values = np.ones(81, dtype=np.float32)  # Full spectral shape
    illuminant = colour.SpectralDistribution(illuminant_values, domain=SPECTRAL_SHAPE)
    
    values = [25.0, 15.0, 10.0]
    edges = [510.0, 495.0, 605.0, 590.0]
    transitions = [10.0, 10.0, 10.0, 10.0]
    
    # Python implementation
    result_py = filterset(illuminant, values, edges, transitions)
    
    print(f"Illuminant shape: {illuminant_values.shape}")
    print(f"Values: {values}")
    print(f"Python result shape: {result_py.shape}")
    print(f"Python result first 5: {result_py[:5]}")
    print(f"Python result last 5: {result_py[-5:]}")
    print()

def test_compute_band_pass_filter():
    """Test compute_band_pass_filter function"""
    print("=== Testing compute_band_pass_filter ===")
    
    # Test data
    filter_uv = [0.8, 410.0, 8.0]
    filter_ir = [0.6, 675.0, 15.0]
    
    # Python implementation
    result_py = compute_band_pass_filter(filter_uv, filter_ir)
    
    print(f"UV filter params: {filter_uv}")
    print(f"IR filter params: {filter_ir}")
    print(f"Python result shape: {result_py.shape}")
    print(f"Python result first 5: {result_py[:5]}")
    print(f"Python result last 5: {result_py[-5:]}")
    print()

def test_dichroic_filters():
    """Test DichroicFilters class"""
    print("=== Testing DichroicFilters ===")
    
    # Create filter instance
    filters = DichroicFilters('thorlabs')
    
    # Test data
    illuminant = np.ones(81, dtype=np.float32)  # Full spectral shape
    values = [0.5, 0.3, 0.2]
    
    # Python implementation
    result_py = filters.apply(illuminant, values)
    
    print(f"Brand: thorlabs")
    print(f"Illuminant shape: {illuminant.shape}")
    print(f"Values: {values}")
    print(f"Python result shape: {result_py.shape}")
    print(f"Python result first 5: {result_py[:5]}")
    print(f"Python result last 5: {result_py[-5:]}")
    print()

def test_generic_filter():
    """Test GenericFilter class"""
    print("=== Testing GenericFilter ===")
    
    # Create filter instance
    filter_obj = GenericFilter('KG3', 'heat_absorbing', 'schott')
    
    # Test data
    illuminant = np.ones(81, dtype=np.float32)
    value = 0.7
    
    # Python implementation
    result_py = filter_obj.apply(illuminant, value)
    
    print(f"Name: KG3, Type: heat_absorbing, Brand: schott")
    print(f"Illuminant shape: {illuminant.shape}")
    print(f"Value: {value}")
    print(f"Python result shape: {result_py.shape}")
    print(f"Python result first 5: {result_py[:5]}")
    print(f"Python result last 5: {result_py[-5:]}")
    print()

def test_color_enlarger():
    """Test color_enlarger function"""
    print("=== Testing color_enlarger ===")
    
    # Test data
    light_source = np.ones(81, dtype=np.float32)
    y_filter_value = 85.0
    m_filter_value = 45.0
    c_filter_value = 25.0
    
    # Python implementation
    result_py = color_enlarger(light_source, y_filter_value, m_filter_value, c_filter_value)
    
    print(f"Light source shape: {light_source.shape}")
    print(f"Y filter: {y_filter_value}, M filter: {m_filter_value}, C filter: {c_filter_value}")
    print(f"Python result shape: {result_py.shape}")
    print(f"Python result first 5: {result_py[:5]}")
    print(f"Python result last 5: {result_py[-5:]}")
    print()

if __name__ == "__main__":
    print("Color Filters Python Reference Test")
    print("=" * 50)
    
    test_sigmoid_erf()
    test_create_combined_dichroic_filter()
    test_filterset()
    test_compute_band_pass_filter()
    test_dichroic_filters()
    test_generic_filter()
    test_color_enlarger()
    
    print("All Python tests completed!") 