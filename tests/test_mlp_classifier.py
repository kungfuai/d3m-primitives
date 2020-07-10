import os 
import json
from glob import glob

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch
from rsp.data import bilinear_upsample, BANDS
from tifffile import imread as tiffread

from primitives.remote_sensing.featurizer.remote_sensing_pretrained import (
    RemoteSensingPretrainedPrimitive, 
    Hyperparams as rs_hp
)
from primitives.remote_sensing.classifier.mlp_classifier import ( 
    MlpClassifierPrimitive, 
    Hyperparams as mlp_hp
)

dataset_path = 'test_data/BigEarthNet-trimmed'
amdim_path = 'static_volumes/8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46'
moco_path = 'static_volumes/fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f'

def load_patch(imname):
    patch = [
        tiffread(f'{imname}_{band}.tif')
        for band in BANDS
    ]
    patch = np.stack([bilinear_upsample(xx) for xx in patch]) 
    return patch

def load_inputs():
    fnames  = sorted(glob('bigearth-100-single/*/*.tif'))
    imnames = sorted(list(set(['_'.join(f.split('_')[:-1]) for f in fnames])))

    imgs = [
        load_patch(img_path).astype(np.float32) 
        for img_path in imnames
    ]

    y = [i.split('/')[1] for i in imnames]
    tgts = LabelEncoder().fit_transform(y)

    return pd.DataFrame({'image_col': imgs}), pd.DataFrame({'target': tgts})

train_inputs, labels = load_inputs()

featurizer = RemoteSensingPretrainedPrimitive(
    hyperparams=rs_hp(
        rs_hp.defaults(),
        inference_model = 'moco',
        use_columns = [0],
        pool_features = False
    ),
    volumes = {'amdim_weights': amdim_path, 'moco_weights': moco_path}
)
features = featurizer.produce(inputs = train_inputs).value

def test_fit():

    mlp = MlpClassifierPrimitive(
        hyperparams=mlp_hp(
            mlp_hp.defaults(),
            epochs=0,
            weights_filepath = '/scratch_dir/model_weights.pth'
        )
    )

    mlp.set_training_data(inputs = features, outputs = labels)
    assert mlp._clf_model[-1].weight.shape[0] == mlp._nclasses
    mlp.fit()
    global mlp_params
    mlp_params = mlp.get_params()

def test_produce():

    mlp = MlpClassifierPrimitive(
        hyperparams=mlp_hp(
            mlp_hp.defaults(),
            weights_filepath = '/scratch_dir/model_weights.pth'
        )
    )
    mlp.set_params(params = mlp_params)

    preds = mlp.produce(inputs=features).value
    assert preds.shape == (train_inputs.shape[0],1)

def test_produce_explanations():

    mlp = MlpClassifierPrimitive(
        hyperparams=mlp_hp(
            mlp_hp.defaults(),
            weights_filepath = '/scratch_dir/model_weights.pth',
        )
    )
    mlp.set_params(params = mlp_params)

    explanations = mlp.produce_explanations(inputs=features).value
    assert explanations.shape == (train_inputs.shape[0], 1)
    assert explanations.iloc[0,0].shape == (120,120)

def test_produce_explanations_all_classes():

    mlp = MlpClassifierPrimitive(
        hyperparams=mlp_hp(
            mlp_hp.defaults(),
            weights_filepath = '/scratch_dir/model_weights.pth',
            explain_all_classes = True
        )
    )
    mlp.set_params(params = mlp_params)

    explanations = mlp.produce_explanations(inputs=features).value
    assert explanations.shape == (train_inputs.shape[0], mlp._nclasses)
    assert explanations.iloc[0,0].shape == (120,120)