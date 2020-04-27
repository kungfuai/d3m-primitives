#!/usr/bin/env python

"""
    data.py
"""

import os
import numpy as np
from PIL import Image
from tifffile import imread as tiffread

BANDS = ['B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12']

# --
# Helpers

def bilinear_upsample(x, n=120):
    dtype = x.dtype
    assert len(x.shape) == 2
    if (x.shape[0] == n) and (x.shape[1] == n):
        return x
    else:
        x = x.astype(np.float)
        x = Image.fromarray(x)
        x = x.resize((n, n), Image.BILINEAR)
        x = np.array(x)
        x = x.astype(dtype)
        return x

def load_patch(patch_dir):
    patch_name = os.path.basename(patch_dir)
    patch      = [tiffread(os.path.join(patch_dir, f'{patch_name}_{band}.tif')) for band in BANDS]
    patch      = np.stack([bilinear_upsample(xx) for xx in patch])
    return patch