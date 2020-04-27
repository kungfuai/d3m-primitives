#!/usr/bin/env python

"""
    inference/amdim.py
"""

import warnings
import numpy as np
from functools import partial
import pickle

import torch
from torch import nn
from torch.nn import functional as F

from .model import Encoder
from .prep import BENTransformValid

class AMDIM(nn.Module):
    def __init__(self, state_dict):
        super().__init__()
        
        config = {
            "ndf"          : 128,
            "num_channels" : 12,
            "n_rkhs"       : 1024,
            "n_depth"      : 3,
            "encoder_size" : 128,
            "use_bn"       : 0,
        }
        
        dummy_batch = torch.zeros((2, config['num_channels'], config['encoder_size'], config['encoder_size']))
        
        self.encoder = Encoder(dummy_batch, **config)
        
        state_dict = {k:v for k,v in state_dict.items() if 'encoder.' in k}
        state_dict = {k.replace('encoder.', '').replace('module.', ''):v for k,v in state_dict.items()}
        self.encoder.load_state_dict(state_dict)
        
        self.transform = BENTransformValid()
    
    def _forward(self, x):
        assert len(x.shape) == 4, "Input must be (batch_size, 12, 128, 128)"
        assert x.shape[1] == 12, "Input must be (batch_size, 12, 128, 128)"
        
        if (x.shape[2] != 128) or (x.shape[3] != 128):
            warnings.warn("input will be resized to (_, 12, 128, 128)")
        
        # --
        # Preprocessing
        device = x.device
        x      = x.cpu()
        
        tmp = [xx.numpy().transpose(1, 2, 0) for xx in x]
        tmp = [self.transform(xx) for xx in tmp]
        x   = torch.stack(tmp)
        
        x = x.to(device)
        
        # --
        # Forward
        acts = self.encoder._forward_acts(x)
        out  = self.encoder.rkhs_block_1(acts[self.encoder.dim2layer[1]])
        out  = out[:,:,0,0]
        
        return out
    
    def forward(self, x, with_grad=False):
        if with_grad:
            return self._forward(x)
        else:
            with torch.no_grad():
                return self._forward(x)


def amdim(state_dict_path, map_location = torch.device('cpu')):
    
    state_dict = torch.load(state_dict_path, map_location=map_location)
    return AMDIM(state_dict)

