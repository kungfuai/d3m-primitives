import os 
import json
from glob import glob

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
import torch
from rsp.data import bilinear_upsample, BANDS
from tifffile import imread as tiffread
from d3m.container import DataFrame as d3m_DataFrame

from kf_d3m_primitives.remote_sensing.featurizer.remote_sensing_pretrained import (
    RemoteSensingPretrainedPrimitive, 
    Hyperparams as rs_hp
)
from kf_d3m_primitives.remote_sensing.classifier.mlp_classifier import ( 
    MlpClassifierPrimitive, 
    Hyperparams as mlp_hp
)

amdim_path = '/static_volumes/8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46'
moco_path = '/static_volumes/fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f'

def load_patch(imname):
    patch = [
        tiffread(f'{imname}_{band}.tif')
        for band in BANDS
    ]
    patch = np.stack([bilinear_upsample(xx) for xx in patch]) 
    return patch

def load_inputs():
    fnames  = sorted(glob('/test_data/bigearth-100-single/*/*.tif'))
    imnames = sorted(list(set(['_'.join(f.split('_')[:-1]) for f in fnames])))
    imgs = [
        load_patch(img_path).astype(np.float32) 
        for img_path in imnames
    ]
    imgs_df = pd.DataFrame({'image_col': imgs, 'dummy_idx': range(len(imgs))})

    y = [i.split('/')[3] for i in imnames]
    tgts = LabelEncoder().fit_transform(y)
    tgts_df = pd.DataFrame({'target': tgts})

    return (
        d3m_DataFrame(imgs_df), 
        d3m_DataFrame(tgts_df)
    )

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
features = features.drop(columns = 'dummy_idx')
# features.to_pickle("dummy.pkl")
# labels.to_pickle("labels.pkl")
# features = pd.read_pickle("dummy.pkl")
# labels = pd.read_pickle("labels.pkl")

def test_fit():

    mlp = MlpClassifierPrimitive(
        hyperparams=mlp_hp(
            mlp_hp.defaults(),
            epochs=1,
            weights_filepath = '/scratch_dir/model_weights.pth'
        )
    )

    mlp.set_training_data(inputs = features, outputs = labels)
    mlp.fit()
    assert mlp._clf_model[-1].weight.shape[0] == mlp._nclasses
    global mlp_params
    mlp_params = mlp.get_params()

def test_produce():

    mlp = MlpClassifierPrimitive(
        hyperparams=mlp_hp(
            mlp_hp.defaults(),
            weights_filepath = '/scratch_dir/model_weights.pth',
            all_confidences = False
        )
    )
    mlp.set_params(params = mlp_params)

    preds = mlp.produce(inputs=features).value
    assert preds.shape == (features.shape[0],2)
    assert (preds.columns == ['target', 'confidence']).all()

def test_produce_all_confidences():

    mlp = MlpClassifierPrimitive(
        hyperparams=mlp_hp(
            mlp_hp.defaults(),
            weights_filepath = '/scratch_dir/model_weights.pth',
        )
    )
    mlp.set_params(params = mlp_params)

    preds = mlp.produce(inputs=features).value
    nc = mlp._nclasses
    assert preds.shape == (features.shape[0]*nc,2)
    if nc > 2:
        for i in range(0,features.shape[0]):
            assert round(preds['confidence'][i*nc:(i+1)*nc].sum()) == 1
    assert (preds.columns == ['target', 'confidence']).all()

def test_produce_explanations():

    mlp = MlpClassifierPrimitive(
        hyperparams=mlp_hp(
            mlp_hp.defaults(),
            weights_filepath = '/scratch_dir/model_weights.pth',
        )
    )
    mlp.set_params(params = mlp_params)

    explanations = mlp.produce_explanations(inputs=features).value
    assert explanations.shape == (features.shape[0], 1)
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
    assert explanations.shape == (features.shape[0], mlp._nclasses)
    assert explanations.iloc[0,0].shape == (120,120)