#!/usr/bin/env python

"""
    moco_r50/inference.py
"""

import torch
from torch import nn

from .resnet import resnet50

def moco_r50(
    state_dict_path, 
    drop_fc=True, 
    map_location=torch.device('cpu')
):
    state_dict = torch.load(state_dict_path, map_location=map_location)['state_dict']
    
    in_channels = state_dict['module.encoder_q.conv1.weight'].shape[1]
    model = resnet50(in_channels=in_channels, num_classes=128)
    
    dim_mlp  = model.fc.weight.shape[1]
    model.fc = nn.Sequential(nn.Linear(dim_mlp, dim_mlp), nn.ReLU(), model.fc)
    
    for k in list(state_dict.keys()):
        if 'encoder_q' not in k:
            del state_dict[k]
            
    state_dict = {k.replace('module.', '').replace('encoder_q.', ''):v for k,v in state_dict.items()}
    model.load_state_dict(state_dict)
    
    model = model.eval()
    
    if drop_fc:
        model.fc = nn.Sequential()
    
    return model