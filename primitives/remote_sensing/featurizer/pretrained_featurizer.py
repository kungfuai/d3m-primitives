import sys
import os.path
import numpy as np
import pandas as pd
import typing
from typing import List

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataframe_utils

import torch
from torch.utils.data import DataLoader, TensorDataset
from ..utils.data import load_patch
from ..moco_r50.inference import moco_r50
from ..moco_r50.data import load_patch_sentinel
from ..amdim.inference import amdim


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
    # ------ CAN REMOVE THIS HP WHEN BASE PATH IS IN METADATA ---------
    base_path = hyperparams.Hyperparameter[str](
        default='/test_data/BigEarthNet-trimmed',
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="weights of trained model will be saved to this filepath",
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

class RemoteSensingTransferPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    '''
        Primitive that applies a self-supervised, pretrained remote sensing featurizer. There are two 
        options for the inference model pretraining task: Augmented Multiscale Deep InfoMax (amdim),
        https://arxiv.org/abs/1906.00910 and Momentum Contrast (moco), https://arxiv.org/abs/1911.05722

        Training inputs: D3M dataset
        Outputs: D3M dataset with featurized RS images (one feature/column)
    '''

    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "544bb61f-f354-48f5-b055-5c03de71c4fb",
        'version': __version__,
        'name': "RSPretrained",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['remote sensing', 'self-supervised', 'pretrained', 'featurizer', 'moco', 'momentum contrast'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            "uris": [
                # Unstructured URIs.
                "https://github.com/kungfuai/d3m-primitives",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
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
            "file_uri": "http://public.datadrivendiscovery.org/",
            "file_digest":"8946fea864c29ed785e00a9cbaa9a50295eb5a334b014f27ba20927104b07f46"
            },
            {
            "type": "FILE",
            "key": "moco_weights",
            "file_uri": "http://public.datadrivendiscovery.org/",
            "file_digest":"fcc8a5a05fa7dbad8fc55584a77fc5d2c407e03a88610267860b45208e152f1f"
            },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        # TODO PR adding remote_sensing_pretrained to names, and two alg types
        'python_path': 'd3m.primitives.remote_sensing.remote_sensing_pretrained.RemoteSensingTransfer',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.MUTUAL_INFORMATION,
            #metadata_base.PrimitiveAlgorithmType.MOMENTUM_CONTRAST,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.REMOTE_SENSING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str, str] = None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self._load_inference_model(volumes)

    def _load_inference_model(
        self, 
        volumes: typing.Dict[str, str] = None,
    ):

        if self.hyperparams['inference_model'] == 'amdim':
            return amdim(volumes['amdim_weights'], map_location=self.device)
        elif self.hyperparams['inference_model'] == 'moco':
            return moco_r50(volumes['moco_weights'], map_location=self.device)

    def _get_image_paths(
        self,
        inputs: d3m_DataFrame,
        col: int
    ) -> List[str]:
        """ get image paths (potentially in multiple columns) that we want to featurize
        """
        #base_path = inputs.metadata.query((metadata_base.ALL_ELEMENTS, col))['location_base_uris'][0].replace('file:///', '/') 
        base_path = self.hyperparams['base_path']
        return [os.path.join(base_path, filename) for filename in inputs.iloc[:,col]]

    def _load_images(
        self,
        image_paths: np.ndarray
    ) -> List:
        """ load images from array of image filepaths
        """
        imgs = [load_patch(img_path).astype(np.float32) for img_path in image_paths]
        if self.hyperparams['inference_model'] == 'moco':
            imgs = [load_patch_sentinel(img) for img in imgs]
        return TensorDataset(torch.FloatTensor(np.stack(imgs)))

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

        image_paths = [self._get_image_paths(inputs, col) for col in image_cols] # bands in different rows?
        image_datasets = [self._load_images(path) for path in image_paths]
        image_loaders = [
            DataLoader(
                image_dataset, 
                batch_size=self.hyperparams['batch_size'],
                num_workers=self.hyperparams['num_workers'],
            )
            for image_dataset in image_datasets
        ]

        for image_loader, col in zip(image_loaders, image_cols):
            all_img_features = []
            for image_batch in image_loader:
                image_batch = image_batch[0].to(self.device)
                features = self.model(image_batch).data.numpy()
                all_img_features.append(features)

        feature_df = d3m_DataFrame(
            pd.DataFrame(
                np.stack(features), 
                columns = ['v{}'.format(i) for i in range(0, features.shape[1])]
            ),
            generate_metadata = True
        )

        return CallResult(feature_df)

