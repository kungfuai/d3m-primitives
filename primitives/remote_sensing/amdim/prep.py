#!/usr/bin/env python

"""
    models/amdim/prep.py
"""

import numpy as np

import torch
from torchvision import transforms
from albumentations import Compose as ACompose
from albumentations.augmentations import transforms as atransforms

BANDS = ('B01', 'B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'B08', 'B8A', 'B09', 'B11', 'B12')

BAND_STATS = {
    'mean': np.array([
        340.76769064,
        429.9430203,
        614.21682446,
        590.23569706,
        950.68368468,
        1792.46290469,
        2075.46795189,
        2218.94553375,
        2266.46036911,
        2246.0605464,
        1594.42694882,
        1009.32729131
    ]),
    'std': np.array([
        554.81258967,
        572.41639287,
        582.87945694,
        675.88746967,
        729.89827633,
        1096.01480586,
        1273.45393088,
        1365.45589904,
        1356.13789355,
        1302.3292881,
        1079.19066363,
        818.86747235,
    ])
}


class BENTransformValid:
    def __init__(self):
        self.transform = ACompose([
            atransforms.Resize(128, 128, interpolation=3)
        ])
        
        self.post_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=BAND_STATS['mean'], std=BAND_STATS['std'])
        ])
    
    def __call__(self, inp):
        a = self.transform(image=inp)['image']
        a = self.post_transform(a)
        return a