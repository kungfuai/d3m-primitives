import os 
import pandas as pd
import numpy as np

from rsp.data import load_patch
from rsp.moco_r50.resnet import ResNet
from rsp.amdim.inference import AMDIM

from primitives.remote_sensing.featurizer.remote_sensing_pretrained import (
    RemoteSensingPretrainedPrimitive, 
    Hyperparams as rs_hp
)

dataset_path = 'test_data/BigEarthNet-trimmed'
amdim_path = 'static_volumes/8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46'
moco_path = 'static_volumes/fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f'

def load_frame():
    img_paths = [
        os.path.join(dataset_path, filename) 
        for filename in os.listdir(dataset_path)
    ]
    imgs = [
        load_patch(img_path).astype(np.float32) 
        for img_path in img_paths
    ]
    return pd.DataFrame({'image_col': imgs})

test_frame = load_frame()

# def test_init_when_inference_model_amdim():
#     rsp = RemoteSensingPretrainedPrimitive(
#         hyperparams=rs_hp(
#             rs_hp.defaults(),
#             inference_model = 'amdim',
#             use_columns = [0],
#         ),
#         volumes = {'amdim_weights': amdim_path, 'moco_weights': moco_path},
#     )
#     assert isinstance(rsp.model, AMDIM)

# def test_init_when_inference_model_moco():
#     rsp = RemoteSensingPretrainedPrimitive(
#         hyperparams=rs_hp(
#             rs_hp.defaults(),
#             inference_model = 'moco',
#             use_columns = [0],
#         ),
#         volumes = {'amdim_weights': amdim_path, 'moco_weights': moco_path},
#     )
#     assert isinstance(rsp.model, ResNet)

# def test_produce_when_inference_model_amdim():
#     rsp = RemoteSensingPretrainedPrimitive(
#         hyperparams=rs_hp(
#             rs_hp.defaults(),
#             inference_model = 'amdim',
#             use_columns = [0],
#         ),
#         volumes = {'amdim_weights': amdim_path, 'moco_weights': moco_path},
#     )
#     feature_df = rsp.produce(inputs=test_frame).value
#     assert feature_df.shape[0] == test_frame.shape[0]
#     assert feature_df.shape[1] == 1024

def test_produce_when_inference_model_moco():
    rsp = RemoteSensingPretrainedPrimitive(
        hyperparams=rs_hp(
            rs_hp.defaults(),
            inference_model = 'moco',
            use_columns = [0],
        ),
        volumes = {'amdim_weights': amdim_path, 'moco_weights': moco_path},
    )
    feature_df = rsp.produce(inputs=test_frame).value
    assert feature_df.shape[0] == test_frame.shape[0]
    assert feature_df.shape[1] == 2048

def test_produce_when_inference_model_moco_no_pooling():
    rsp = RemoteSensingPretrainedPrimitive(
        hyperparams=rs_hp(
            rs_hp.defaults(),
            inference_model = 'moco',
            use_columns = [0],
            pool_features = False,
        ),
        volumes = {'amdim_weights': amdim_path, 'moco_weights': moco_path},
    )
    feature_df = rsp.produce(inputs=test_frame).value
    assert feature_df.shape[0] == test_frame.shape[0]
    assert feature_df.shape[1] == 2048*4*4
