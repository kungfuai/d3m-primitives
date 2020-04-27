#!/usr/bin/env python

"""
    moco_r50/data.py
"""

import numpy as np

from albumentations import Compose as ACompose
from albumentations.pytorch.transforms import ToTensor as AToTensor
from albumentations.augmentations import transforms as atransforms

# --
# NAIP

NAIP_BAND_STATS = {
    'mean' : np.array([0.38194386, 0.38695849, 0.35312921, 0.45349037])[None,None],
    'std'  : np.array([0.21740159, 0.18325207, 0.15651401, 0.20699527])[None,None],
}

def _naip_normalize(x, **kwargs):
    return (x - NAIP_BAND_STATS['mean']) / NAIP_BAND_STATS['std']

def naip_augmentation_valid():
    return ACompose([
        atransforms.Lambda(name='normalize', image=_naip_normalize),
        AToTensor(),
    ])

def load_patch_naip(X):
    #X = np.load(inpath)
    X = X[:4].transpose(1, 2, 0).astype(np.float32) / 255
    
    transform = naip_augmentation_valid()
    X = transform(image=X)['image']
    return X

# --
# SENTINEL-2

SENTINEL_BAND_STATS = {
    'mean' : np.array([1.08158484e-01, 1.21479766e-01, 1.45487537e-01, 1.58012632e-01, 1.94156398e-01, 2.59219257e-01,
                       2.83195732e-01, 2.96798923e-01, 3.01822935e-01, 3.08726458e-01, 2.37724304e-01, 1.72824851e-01])[None,None],
    'std'  : np.array([2.00349529e-01, 2.06218237e-01, 1.99808794e-01, 2.05981393e-01, 2.00533060e-01, 1.82050607e-01,
                       1.76569472e-01, 1.80955308e-01, 1.68494856e-01, 1.80597534e-01, 1.15451671e-01, 1.06993609e-01])[None,None],
}


def _sentinel_normalize(x, **kwargs):
    return (x - SENTINEL_BAND_STATS['mean']) / SENTINEL_BAND_STATS['std']

def sentinel_augmentation_valid():
    return ACompose([
        atransforms.Lambda(name='normalize', image=_sentinel_normalize),
        AToTensor(),
    ])

def load_patch_sentinel(X):
    #X = np.load(inpath)
    X = X[:12].transpose(1, 2, 0).astype(np.float32) / 10_000
    
    transform = sentinel_augmentation_valid()
    X = transform(image=X)['image']
    return X

