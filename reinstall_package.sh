#!/bin/bash
pkg=denoise_lung_masks
python3 -m build && 
    pip uninstall -y ${pkg} &&
    pip install ${pkg} -f dist
