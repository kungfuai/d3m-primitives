import os.path
import typing
from typing import List

import numpy as np
import pandas as pd
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base
import torch
from torch.utils.data import DataLoader, TensorDataset
from rsp.data import load_patch
from rsp.moco_r50.inference import moco_r50
from rsp.moco_r50.data import sentinel_augmentation_valid
from rsp.amdim.inference import amdim


__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:jeffrey.gleason@kungfu.ai'

Inputs = container.DataFrame
Outputs = container.DataFrame

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
    num_workers = hyperparams.UniformInt(
        lower=1,
        upper=16,
        default=8,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of workers to do if using multiprocessing threading",
    )

class RemoteSensingPretrainedPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
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

    def _load_inference_model(
        self, 
        volumes: typing.Dict[str, str] = None,
    ):
        """ load either amdim or moco inference model
        """
        if self.hyperparams['inference_model'] == 'amdim':
            return amdim(volumes['amdim_weights'], map_location=self.device)
        elif self.hyperparams['inference_model'] == 'moco':
            return moco_r50(volumes['moco_weights'], map_location=self.device)

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

        image_dataset = self._load_dataset(inputs, image_cols)
        image_loader = DataLoader(
            image_dataset, 
            batch_size=self.hyperparams['batch_size'],
            num_workers=self.hyperparams['num_workers'],
        )

        all_img_features = []
        for image_batch in image_loader:
            image_batch = image_batch[0].to(self.device)
            features = self.model(image_batch).cpu().data.numpy()
            all_img_features.append(features)
        all_img_features = np.vstack(all_img_features)

        col_names = [
            'img_{}_feat_{}'.format(i // all_img_features.shape[1], i % all_img_features.shape[1]) 
            for i in range(0, all_img_features.shape[1])
        ]
        all_img_features = self._sort_multiple_img_cols(all_img_features, inputs.shape[0])

        feature_df = d3m_DataFrame(
            pd.DataFrame(all_img_features, columns = col_names),
            generate_metadata = True
        )

        return CallResult(feature_df)

    def _load_patch_sentinel(
        self,
        img: np.ndarray
    ):
        """ load and transform sentinel image patch to prep for model """
        img = img[:12].transpose(1, 2, 0) / 10_000
        return sentinel_augmentation_valid()(image=img)['image']

    def _load_dataset(
        self,
        inputs: d3m_DataFrame,
        image_cols: List[int]
    ) -> TensorDataset:
        """ load image dataset from 1 or more columns of np arrays representing images """
        imgs = [img for img_col in image_cols for img in inputs.iloc[:, img_col]]
        if self.hyperparams['inference_model'] == 'moco':
            imgs = [self._load_patch_sentinel(img) for img in imgs]
        return TensorDataset(torch.FloatTensor(np.stack(imgs)))

    def _sort_multiple_img_cols(
        self,
        img_features: np.ndarray,
        imgs_per_col: int
    ) -> np.ndarray:
        """ if multiple original columns of images were processed, sort generated feature vecs correctly
        """

        return np.concatenate(
            [
                img_features[i:i+imgs_per_col] 
                for i in range(0, img_features.shape[0], imgs_per_col)
            ],
            axis=1
        )