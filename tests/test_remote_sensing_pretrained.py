import os
import struct

import pandas as pd
import numpy as np
import lz4
from rsp.data import load_patch
from rsp.moco_r50.resnet import ResNet
from rsp.amdim.inference import AMDIM
from d3m.container import DataFrame as d3m_DataFrame

from kf_d3m_primitives.remote_sensing.featurizer.remote_sensing_pretrained import (
    RemoteSensingPretrainedPrimitive,
    Hyperparams as rs_hp,
)

dataset_path = "test_data/BigEarthNet-trimmed"
amdim_path = (
    "static_volumes/8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46"
)
moco_path = (
    "static_volumes/fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f"
)


def load_frame(compress_data=False):
    img_paths = [
        os.path.join(dataset_path, filename) for filename in os.listdir(dataset_path)
    ]
    imgs = [load_patch(img_path).astype(np.float32) for img_path in img_paths]
    if compress_data:
        compressed_imgs = []
        for img in imgs:
            output_bytes = bytearray(
                struct.pack(
                    "cIII",
                    bytes(img.dtype.char.encode()),
                    len(img),
                    img.shape[1],
                    img.shape[1],
                )
            )
            output_bytes.extend(img.tobytes())
            compressed_bytes = lz4.frame.compress(bytes(output_bytes))
            compressed_img = np.frombuffer(
                compressed_bytes, dtype="uint8", count=len(compressed_bytes)
            )
            compressed_imgs.append(compressed_img)
        imgs = compressed_imgs

    df = pd.DataFrame({"dummy_idx": range(len(imgs)), "image_col": imgs})
    return d3m_DataFrame(df)


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


def _test_output_df(input_df, output_df, num_features):
    assert output_df.shape[0] == 5
    assert output_df.columns[0] == "dummy_idx"
    assert output_df.shape[1] == input_df.shape[1] - 1 + num_features
    for i in range(1, num_features + 1):
        assert (
            "http://schema.org/Float"
            in output_df.metadata.query_column(i)["semantic_types"]
        )


def test_produce_when_inference_model_moco():
    test_frame = load_frame()
    rsp = RemoteSensingPretrainedPrimitive(
        hyperparams=rs_hp(
            rs_hp.defaults(),
            inference_model="moco",
            use_columns=[1],
        ),
        volumes={"amdim_weights": amdim_path, "moco_weights": moco_path},
    )
    global feature_df
    feature_df = rsp.produce(inputs=test_frame).value
    _test_output_df(test_frame, feature_df, 2048)


# def test_produce_when_inference_model_moco_no_pooling():
#     test_frame = load_frame()
#     rsp = RemoteSensingPretrainedPrimitive(
#         hyperparams=rs_hp(
#             rs_hp.defaults(),
#             inference_model="moco",
#             use_columns=[1],
#             pool_features=False,
#         ),
#         volumes={"amdim_weights": amdim_path, "moco_weights": moco_path},
#     )
#     feature_df = rsp.produce(inputs=test_frame).value
#     _test_output_df(test_frame, feature_df, 2048 * 4 * 4)


def test_produce_when_inference_model_moco_decompress():
    test_frame = load_frame(compress_data=True)
    rsp = RemoteSensingPretrainedPrimitive(
        hyperparams=rs_hp(
            rs_hp.defaults(),
            inference_model="moco",
            use_columns=[1],
            decompress_data=True,
        ),
        volumes={"amdim_weights": amdim_path, "moco_weights": moco_path},
    )
    decompress_feature_df = rsp.produce(inputs=test_frame).value
    _test_output_df(test_frame, decompress_feature_df, 2048)
    assert decompress_feature_df.equals(feature_df)
