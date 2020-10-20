import os
from glob import glob
import time
import json

from PIL import Image
import pandas as pd
import numpy as np
import torchvision as tv
from rsp.data import bilinear_upsample, BANDS
from tifffile import imread as tiffread
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import base as metadata_base

from kf_d3m_primitives.remote_sensing.featurizer.remote_sensing_pretrained import (
    RemoteSensingPretrainedPrimitive, 
    Hyperparams as rs_hp
)
from kf_d3m_primitives.remote_sensing.image_retrieval.image_retrieval import (
    ImageRetrievalPrimitive, 
    Hyperparams as ir_hp
)
from kf_d3m_primitives.remote_sensing.image_retrieval.image_retrieval_pipeline import ImageRetrievalPipeline


amdim_path = '/static_volumes/8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46'
moco_path = '/static_volumes/fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f'

def load_nwpu(data_dir: str = '/NWPU-RESISC45', n_imgs = 200):
    paths = sorted(glob(os.path.join(data_dir, '*/*')))
    paths = [os.path.abspath(p) for p in paths]
    imgs = [Image.open(p) for p in paths[:n_imgs]]
    labels  = [os.path.basename(os.path.dirname(p)) for p in paths[:n_imgs]]

    transform = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(
            mean = (0.3680, 0.3810, 0.3436),
            std  = (0.2034, 0.1854, 0.1848),
        )
    ])
    imgs = [transform(img) for img in imgs]

    imgs = d3m_DataFrame(pd.DataFrame({'imgs': imgs}))
    labels = np.array(labels)
    return imgs, labels

def load_patch(imname):
    patch = [
        tiffread(f'{imname}_{band}.tif')
        for band in BANDS
    ]
    patch = np.stack([bilinear_upsample(xx) for xx in patch]) 
    return patch

def load_big_earthnet():
    fnames  = sorted(glob('/test_data/bigearth-100-single/*/*.tif'))
    imnames = sorted(list(set(['_'.join(f.split('_')[:-1]) for f in fnames])))
    imgs = [
        load_patch(img_path).astype(np.float32) 
        for img_path in imnames
    ]
    imgs_df = pd.DataFrame({'image_col': imgs, 'index': range(len(imgs))})
    imgs_df = d3m_DataFrame(imgs_df)
    imgs_df.metadata = imgs_df.metadata.add_semantic_type(
        (metadata_base.ALL_ELEMENTS, 1),
        'https://metadata.datadrivendiscovery.org/types/PrimaryKey'
    )

    y = [i.split('/')[3] for i in imnames]

    return imgs_df, np.array(y)

def iterative_labeling(features, labels, seed_idx = 2, n_rounds = 5):

    # initial query image
    y = (labels == labels[seed_idx]).astype(np.int)
    annotations = np.zeros(features.shape[0]) - 1
    annotations[seed_idx] = 1

    sampler = ImageRetrievalPrimitive(
        hyperparams=ir_hp(
            ir_hp.defaults(),
            reduce_dimension=32
        )
    )

    n_pos, n_neg = 1, 0
    for i in range(n_rounds):
        
        # generate ranking by similarity
        sampler.set_training_data(
            inputs = features, 
            outputs = d3m_DataFrame(pd.DataFrame({'annotations': annotations}))
        )
        sampler.fit()
        ranking_df = sampler.produce(inputs = features).value
        assert ranking_df.shape[0] == features.shape[0] - i - 1

        exc_labeled = ranking_df['index'].values
        inc_labeled = np.concatenate((sampler.pos_idxs, exc_labeled))

        # simulate human labeling
        next_idx = exc_labeled[0]
        next_label = y[next_idx]
        annotations[next_idx] = next_label

        if next_label == 1:
            n_pos += 1
        else:
            n_neg += 1

        # evaluate ranking against ground truth
        results = {
            'round': i + 1,
            'next_idx': int(next_idx),
            'next_label': next_label,
            'n_pos': n_pos,
            'n_neg': n_neg,
            'a_p': [
                float(y[inc_labeled[:k]].mean()) 
                for k in 2 ** np.arange(11)
            ], # precision, including labeled
            'u_p': [
                float(y[exc_labeled[:k]].mean()) 
                for k in 2 ** np.arange(11)
            ], # precision, excluding labeled
            'r_p': [
                float(y[inc_labeled[:k]].sum()/y.sum()) 
                for k in 2**np.arange(11)
            ], # recall, including labeled
        }
        print()
        print(results)

# def test_nwpu():
#     train_inputs, labels = load_nwpu()

#     featurizer = RemoteSensingPretrainedPrimitive(
#         hyperparams=rs_hp(
#             rs_hp.defaults(),
#             inference_model = 'moco',
#             use_columns = [0],
#         ),
#         volumes = {'amdim_weights': amdim_path, 'moco_weights': moco_path}
#     )
#     features = featurizer.produce(inputs = train_inputs).value
#     #features.to_pickle("dummy.pkl")
#     #features = pd.read_pickle("dummy.pkl")

#     iterative_labeling(features, labels)

def test_big_earthnet():

    train_inputs, labels = load_big_earthnet()

    featurizer = RemoteSensingPretrainedPrimitive(
        hyperparams=rs_hp(
            rs_hp.defaults(),
            inference_model = 'moco',
            use_columns = [0],
        ),
        volumes = {'amdim_weights': amdim_path, 'moco_weights': moco_path}
    )
    features = featurizer.produce(inputs = train_inputs).value
    features.to_pickle("dummy.pkl")
    # features = pd.read_pickle("dummy.pkl")

    iterative_labeling(features, labels)

def test_iterative_pipeline(
    dataset = 'LL1_bigearth_landuse_detection', 
    n_rows = 2188,
    n_rounds = 2,
):
    pipeline = ImageRetrievalPipeline(dataset = dataset)
    pipeline.write_pipeline()
    for i in range(n_rounds):
        print(f'Running round {i} pipeline...')
        pipeline.make_annotations_dataset(n_rows, round_num = i)
        pipeline.fit_produce()
    pipeline.delete_pipeline()
    pipeline.delete_annotations_dataset()


