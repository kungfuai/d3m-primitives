import os.path
import typing
import types

import numpy as np
import pandas as pd
from tqdm import tqdm
from d3m.primitive_interfaces.base import (
    CallResult,
    NeuralNetworkModuleMixin,
    PrimitiveBase,
)
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
import torch
from torch.utils.data import DataLoader
from rsp.data import load_patch
from rsp.moco_r50.resnet import ResNet
from rsp.moco_r50.inference import moco_r50
from rsp.amdim.inference import amdim, AMDIM

from .streaming_dataset import StreamingDataset

__author__ = "Distil"
__version__ = "1.0.0"
__contact__ = "mailto:cbethune@uncharted.software"

Inputs = container.DataFrame
Outputs = container.DataFrame
# Module = typing.Union[ResNet, AMDIM]


class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. \
            If any specified column cannot be parsed, it is skipped.",
    )
    inference_model = hyperparams.Enumeration(
        default="moco",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["amdim", "moco"],
        description="type pretrained inference model to use",
    )
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=512,
        default=256,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="inference batch size",
    )
    pool_features = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to pool features across spatial dimensions in returned frame",
    )
    decompress_data = hyperparams.Hyperparameter[bool](
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="If True, applies LZ4 decompression algorithm to the data. \
                    Compressed data stores a header consisting of the dtype character and the \
                    data shape as unsigned integers. Given c struct alignment, will occupy \
                    16 bytes (1 + 4 + 4 + 4 + 3 ) padding",
    )


class RemoteSensingPretrainedPrimitive(
    TransformerPrimitiveBase[Inputs, Outputs, Hyperparams],
    # NeuralNetworkModuleMixin[Inputs, Outputs, Params, Hyperparams, Module]
):
    """
    This primitive featurizes remote sensing imagery using a pre-trained model.
    The pre-trained model was learned on a sample of Sentinel-2 imagery and
    optimized using a self-supervised objective. There are two inference models that
    correspond to two pretext tasks: Augmented Multiscale Deep InfoMax (amdim),
    https://arxiv.org/abs/1906.00910 and Momentum Contrast (moco), https://arxiv.org/abs/1911.05722

    Training inputs: D3M dataset
    Outputs: D3M dataset with featurized RS images (one feature/column)
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "544bb61f-f354-48f5-b055-5c03de71c4fb",
            "version": __version__,
            "name": "RSPretrained",
            "keywords": [
                "remote sensing",
                "self-supervised",
                "pretrained",
                "featurizer",
                "moco",
                "momentum contrast",
            ],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": [
                    "https://github.com/kungfuai/d3m-primitives",
                ],
            },
            "installation": [
                {"type": "PIP", "package": "cython", "version": "0.29.16"},
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/kungfuai/d3m-primitives.git@{git_commit}#egg=kf-d3m-primitives".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
                {
                    "type": "FILE",
                    "key": "amdim_weights",
                    "file_uri": "http://public.datadrivendiscovery.org/amdim_weights.pth",
                    "file_digest": "8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46",
                },
                {
                    "type": "FILE",
                    "key": "moco_weights",
                    "file_uri": "http://public.datadrivendiscovery.org/moco_sentinel_v0.pth.tar",
                    "file_digest": "fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f",
                },
            ],
            "python_path": "d3m.primitives.remote_sensing.remote_sensing_pretrained.RemoteSensingPretrained",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.MUTUAL_INFORMATION,
                metadata_base.PrimitiveAlgorithmType.MOMENTUM_CONTRAST,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.REMOTE_SENSING,
            "can_use_gpus": True,
        }
    )

    def __init__(
        self,
        *,
        hyperparams: Hyperparams,
        random_seed: int = 0,
        volumes: typing.Dict[str, str] = None,
    ) -> None:
        super().__init__(
            hyperparams=hyperparams, random_seed=random_seed, volumes=volumes
        )

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if self.device == "cuda:0":
            torch.cuda.manual_seed(random_seed)

        self.model = self._load_inference_model(volumes)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : D3M dataframe

        Returns
        ----------
        Outputs
           D3M dataframe with featurized RS images (one feature/column)
        """

        if len(self.hyperparams["use_columns"]) == 0:
            image_cols = inputs.metadata.get_columns_with_semantic_type(
                "http://schema.org/ImageObject"
            )
        else:
            image_cols = self.hyperparams["use_columns"]

        if len(image_cols) > 1:
            raise ValueError("Primitive only supports featurizing one image column")
        image_col = image_cols[0]

        image_dataset = StreamingDataset(
            inputs,
            image_col,
            self.hyperparams["inference_model"],
            decompress_data=self.hyperparams["decompress_data"],
        )
        image_loader = DataLoader(
            image_dataset,
            batch_size=self.hyperparams["batch_size"],
        )

        all_img_features = []
        with torch.no_grad():
            for image_batch in tqdm(image_loader):
                image_batch = image_batch.to(self.device)
                features = self.model(image_batch)
                if self.hyperparams["pool_features"]:
                    features = self._aggregate_features(features)
                features = features.detach().cpu().numpy()
                all_img_features.append(features)
        all_img_features = np.vstack(all_img_features)
        all_img_features = all_img_features.reshape(all_img_features.shape[0], -1)
        col_names = [f"feat_{i}" for i in range(0, all_img_features.shape[1])]
        feature_df = pd.DataFrame(all_img_features, columns=col_names)
        feature_df = d3m_DataFrame(feature_df, generate_metadata=True)

        for idx in range(feature_df.shape[1]):
            feature_df.metadata = feature_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, idx), "http://schema.org/Float"
            )

        if inputs.shape[1] > 1:
            input_df = inputs.remove_columns(image_cols)
            feature_df = input_df.append_columns(feature_df)

        return CallResult(feature_df)

    def _load_inference_model(
        self,
        volumes: typing.Dict[str, str] = None,
    ):
        """load either amdim or moco inference model"""
        if self.hyperparams["inference_model"] == "amdim":
            model = amdim(volumes["amdim_weights"], map_location=self.device)
        elif self.hyperparams["inference_model"] == "moco":
            model = moco_r50(volumes["moco_weights"], map_location=self.device)

            def forward(self, x):
                """ Patch forward to eliminate pooling, flattening + fc """
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)

                return x

            model.forward = types.MethodType(forward, model)

        model = model.to(self.device)
        model = model.eval()

        return model

    def _aggregate_features(self, features, spatial_a=2.0, spatial_b=2.0):
        """aggregate features with
        Cross-dimensional Weighting for Aggregated Deep Convolutional Features.
        https://arxiv.org/pdf/1512.04065.pdf
        """

        spatial_weight = features.sum(dim=1, keepdims=True)
        z = (spatial_weight ** spatial_a).sum(dim=(2, 3), keepdims=True)
        z = z ** (1.0 / spatial_a)
        spatial_weight = (spatial_weight / z) ** (1.0 / spatial_b)

        bs, c, w, h = features.shape
        nonzeros = (features != 0).float().sum(dim=(2, 3)) / 1.0 / (w * h) + 1e-6
        channel_weight = torch.log(nonzeros.sum(dim=1, keepdims=True) / nonzeros)

        features = features * spatial_weight
        features = features.sum(dim=(2, 3))
        features = features * channel_weight
        return features
