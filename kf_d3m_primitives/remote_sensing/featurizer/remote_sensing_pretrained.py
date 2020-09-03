import os.path
import typing

import numpy as np
import pandas as pd
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

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:jeffrey.gleason@kungfu.ai'

Inputs = container.DataFrame
Outputs = container.DataFrame
#Module = typing.Union[ResNet, AMDIM] 

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be parsed, it is skipped.",
    )
    inference_model = hyperparams.Enumeration(
        default = 'moco', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['amdim', 'moco'],
        description = 'type pretrained inference model to use'
    )
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=512,
        default=256,
        upper_inclusive=True,
        semantic_types=["https://metadata.datadrivendiscovery.org/types/TuningParameter"],
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
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="If True, applies LZO decompression algorithm to the data.\
                    Compressed data stores a header consisting of the dtype character and the \
                    data shape as unsigned integers. Given c struct alignment, will occupy \
                    16 bytes (1 + 4 + 4 + 4 + 3 ) padding"
    )

class RemoteSensingPretrainedPrimitive(
    TransformerPrimitiveBase[Inputs, Outputs, Hyperparams],
    #NeuralNetworkModuleMixin[Inputs, Outputs, Params, Hyperparams, Module]
):
    '''
        Primitive that featurizes remote sensing imagery using a pre-trained model that was optimized
        with a self-supervised objective. There are two inference models that correspond to two pretext tasks:
        Augmented Multiscale Deep InfoMax (amdim), https://arxiv.org/abs/1906.00910 and 
        Momentum Contrast (moco), https://arxiv.org/abs/1911.05722

        Training inputs: D3M dataset
        Outputs: D3M dataset with featurized RS images (one feature/column)
    '''

    metadata = metadata_base.PrimitiveMetadata({
        'id': "544bb61f-f354-48f5-b055-5c03de71c4fb",
        'version': __version__,
        'name': "RSPretrained",
        'keywords': ['remote sensing', 'self-supervised', 'pretrained', 'featurizer', 'moco', 'momentum contrast'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            "uris": [
                "https://github.com/kungfuai/d3m-primitives",
            ],
        },
        "installation": [
            {"type": "PIP", "package": "cython", "version": "0.29.16"}, 
            {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'zlib1g-dev',
                'version': '1:1.2.11.dfsg-0ubuntu2',
            }, 
            {
                'type': metadata_base.PrimitiveInstallationType.UBUNTU,
                'package': 'liblzo2-dev',
                'version': '2.08-1.2',
            },
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
            "file_digest":"8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46"
            },
            {
            "type": "FILE",
            "key": "moco_weights",
            "file_uri": "http://public.datadrivendiscovery.org/moco_sentinel_v0.pth.tar",
            "file_digest":"fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f"
            },
        ],
        'python_path': 'd3m.primitives.remote_sensing.remote_sensing_pretrained.RemoteSensingPretrained',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.MUTUAL_INFORMATION,
            metadata_base.PrimitiveAlgorithmType.MOMENTUM_CONTRAST,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.REMOTE_SENSING,
        'can_use_gpus': True
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str, str] = None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_inference_model(volumes).to(self.device)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : D3M dataframe

        Returns
        ----------
        Outputs
           D3M dataframe with featurized RS images (one feature/column)
        """ 

        if len(self.hyperparams['use_columns']) == 0:
            image_cols = inputs.metadata.get_columns_with_semantic_type('http://schema.org/ImageObject')
        else:
            image_cols = self.hyperparams['use_columns']
        
        if len(image_cols) > 1:
            raise ValueError('Primitive only supports featurizing one image column')
        image_col = image_cols[0]

        image_dataset = StreamingDataset(
            inputs, 
            image_col,
            self.hyperparams['inference_model'],
            decompress_data = self.hyperparams['decompress_data']
        )
        image_loader = DataLoader(
            image_dataset, 
            batch_size=self.hyperparams['batch_size'],
        )

        all_img_features = []
        with torch.no_grad():
            for image_batch in image_loader:
                image_batch = image_batch.to(self.device)
                features = self.model(image_batch).cpu().data.numpy()
                all_img_features.append(features)
        all_img_features = np.vstack(all_img_features)
        col_names = [f'feat_{i}' for i in range(0, all_img_features.shape[1])]
        feature_df = pd.DataFrame(all_img_features, columns = col_names)
        feature_df = d3m_DataFrame(feature_df, generate_metadata = True)

        for idx in range(feature_df.shape[1]):
            feature_df.metadata = feature_df.metadata.add_semantic_type(
                (metadata_base.ALL_ELEMENTS, idx),
                "http://schema.org/Float"
            )

        return CallResult(feature_df)

    # def get_neural_network_module(self) -> Module:
    #     return self.model

    # def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
    #     return None

    # def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
    #     return CallResult(None)

    # def get_params(self) -> Params:
    #     pass

    # def set_params(self, *, params: Params) -> None:
    #     pass

    def _load_inference_model(
        self, 
        volumes: typing.Dict[str, str] = None,
    ):
        """ load either amdim or moco inference model
        """
        if self.hyperparams['inference_model'] == 'amdim':
            model = amdim(volumes['amdim_weights'], map_location=self.device)
        elif self.hyperparams['inference_model'] == 'moco':
            model = moco_r50(volumes['moco_weights'], map_location=self.device) 

        if not self.hyperparams['pool_features']:
            model.avgpool = torch.nn.Sequential()

        return model

    # def _load_patch_sentinel(
    #     self,
    #     img: np.ndarray
    # ):
    #     """ load and transform sentinel image patch to prep for model """
    #     img = img[:12].transpose(1, 2, 0) / 10_000
    #     return sentinel_augmentation_valid()(image=img)['image']

    # def _load_dataset(
    #     self,
    #     inputs: d3m_DataFrame,
    #     img_col: int
    # ) -> TensorDataset:
    #     """ load image dataset from 1 or more columns of np arrays representing images """
    #     imgs = inputs.iloc[:, img_col]
    #     if self.hyperparams['inference_model'] == 'moco':
    #         imgs = [self._load_patch_sentinel(img) for img in imgs]
    #     else:
    #         imgs = [torch.Tensor(img) for img in imgs]
    #     return torch.stack(imgs)
