from glob import glob

import numpy as np
import pandas as pd
from rsp.data import bilinear_upsample, BANDS
from tifffile import imread as tiffread
from d3m.container import DataFrame as d3m_DataFrame

from kf_d3m_primitives.remote_sensing.segmentation.image_segmentation import (
    ImageSegmentationPrimitive,
    Hyperparams as segmentation_hp
)
from kf_d3m_primitives.remote_sensing.segmentation.image_segmentation_pipeline import (
    ImageSegmentationPipeline
)

moco_path = '/static_volumes/fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f'

def load_patch(imname):
    patch = [
        tiffread(f'{imname}_{band}.tif')
        for band in BANDS
    ]
    patch = np.stack([bilinear_upsample(xx) for xx in patch]) 
    return patch

def load_inputs():
    fnames  = sorted(glob('/test_data/bigearth-100-single-2c/*/*.tif'))
    imnames = sorted(list(set(['_'.join(f.split('_')[:-1]) for f in fnames])))
    imgs = [
        load_patch(img_path).astype(np.float32) 
        for img_path in imnames
    ]
    imgs_df = pd.DataFrame({'image_col': imgs, 'dummy_idx': range(len(imgs))})

    y = [i.split('/')[3] for i in imnames]
    tgts_df = pd.DataFrame({'target': y})

    return (
        d3m_DataFrame(imgs_df), 
        d3m_DataFrame(tgts_df)
    )

imgs, labels = load_inputs()

def test_fit():
    segmentation = ImageSegmentationPrimitive(
        hyperparams=segmentation_hp(
            segmentation_hp.defaults(),
            use_columns=[0],
            epochs_frozen=0,
            epochs_unfrozen=1,
        ),
        volumes = {'moco_weights': moco_path},
    )
    segmentation.set_training_data(inputs=imgs, outputs=labels)
    segmentation.fit()
    global seg_params
    seg_params = segmentation.get_params()

def test_produce():

    segmentation = ImageSegmentationPrimitive(
        hyperparams=segmentation_hp(
            segmentation_hp.defaults(),
            use_columns=[0]
        ),
        volumes = {'moco_weights': moco_path},
    )
    segmentation.set_params(params = seg_params)
    preds = segmentation.produce(inputs=imgs).value
    assert preds.shape == (imgs.shape[0],1)
    assert len(preds.iloc[0,0]) == 120
    assert len(preds.iloc[0,0][0]) == 120

def test_serialize(
    dataset="LL1_bigearth_landuse_detection",
    target_class="Pastures"
):

    data = pd.read_csv(
        f"/datasets/seed_datasets_current/{dataset}/SCORE/dataset_SCORE/tables/learningData.csv"
    )
    multi_class_labels = data[data['band'] == '1']['label']

    def replace(label):
        if label != target_class:
            return f'Not {target_class}'
        else:
            return target_class

    binary_labels = multi_class_labels.apply(replace)
    binary_labels = d3m_DataFrame(binary_labels.to_frame())

    pipeline = ImageSegmentationPipeline(
        binary_labels,
        epochs_frozen=1,
        epochs_unfrozen=1
    )
    pipeline.write_pipeline()
    pipeline.fit_produce(dataset)
    pipeline.delete_pipeline()


