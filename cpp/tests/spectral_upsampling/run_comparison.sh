#!/bin/bash
cd /home/jimmyqiu/cursor/agx-emulsion-zero
export PYTHONPATH=.
python3 cpp/tests/spectral_upsampling/compare_rgb_to_raw_py.py
