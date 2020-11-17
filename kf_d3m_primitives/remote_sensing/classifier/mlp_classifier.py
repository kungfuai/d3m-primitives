import os.path
import typing
import sys
from time import time
import logging
import math

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult, NeuralNetworkModuleMixin
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.exceptions import PrimitiveNotFittedError
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:jeffrey.gleason@kungfu.ai'

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)

class Params(params.Params):
    is_fit: bool
    output_column: str
    label_encoder: LabelEncoder
    nclasses: int

class Hyperparams(hyperparams.Hyperparams):
    weights_filepath = hyperparams.Hyperparameter[str](
        default='model_weights.pth',
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="weights of trained model will be saved to this filepath",
    )
    image_dim = hyperparams.UniformInt(
        lower=1,
        upper=512,
        default=120,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="input dimension of image (height and width)",
    )
    feature_dim = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=2048,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="feature dimension after reshaping flattened feature vector",
    )
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=512,
        default=256,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="training and inference batch size",
    )
    epochs = hyperparams.UniformInt(
        lower = 0,
        upper = sys.maxsize,
        default = 25,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = 'how many epochs for which to finetune classification head (happens first)'
    )
    learning_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="learning rate",
    )
    # explanation_method = hyperparams.Enumeration(
    #     default = 'gradcam',
    #     semantic_types = [
    #         'https://metadata.datadrivendiscovery.org/types/ControlParameter'
    #     ],
    #     values = [
    #         'gradcam',
    #         'gradcam-gbprop'
    #     ],
    #     description = 'Determines whether the output is a dataframe with just predictions,\
    #         or an additional feature added to the input dataframe.'
    # )
    explain_all_classes = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to return explanations for all classes or only the predicted class"
    )
    all_confidences = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to return explanations all classes and all confidences from produce method"
    )

class MlpClassifierPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that adds a relatively simple MLP classification head to the remote sensing
        featurizer. This primitive also surfaces explanation methods based on GradCam:
        https://arxiv.org/pdf/1610.02391v1.pdf.

        Training inputs: 1) Feature dataframe, 2) Label dataframe
        Outputs: D3M dataset with predictions
    '''

    metadata = metadata_base.PrimitiveMetadata({
        'id': "dce5255d-b63c-4601-8ace-d63b42d6d03e",
        'version': __version__,
        'name': "MlpClassifier",
        'keywords': [
            'remote sensing',
            'neural network',
            'classification',
            'explainability',
            'GradCam',
            'GuidedBackProp',
            'GradCam-GuidedBackProp'
        ],
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
                'type': metadata_base.PrimitiveInstallationType.PIP,
                'package': 'python-lzo',
                'version': '1.12',
            },
        ],
        'python_path': 'd3m.primitives.remote_sensing.mlp.MlpClassifier',
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.MULTILAYER_PERCEPTRON,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.REMOTE_SENSING,
        'can_use_gpus': True
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self._is_fit = False
        self._all_outputs = None

        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        if self._device == 'cuda:0':
            torch.cuda.manual_seed(random_seed)

    def get_params(self) -> Params:
        return Params(
            is_fit = self._is_fit,
            output_column = self._output_column,
            label_encoder = self._label_encoder,
            nclasses = self._nclasses
        )

    def set_params(self, *, params: Params) -> None:
        self._is_fit = params['is_fit']
        self._output_column = params['output_column']
        self._label_encoder = params['label_encoder']
        self._nclasses = params['nclasses']

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ Sets primitive's training data

            Arguments:
                inputs {Inputs} -- D3M dataframe containing features
                outputs {Outputs} -- D3M dataframe containing targets

        """

        self._output_column = outputs.columns[0]
        self._label_encoder = LabelEncoder()
        labels = self._label_encoder.fit_transform(outputs.values.ravel())
        self._value_counts = outputs[self._output_column].value_counts()
        self._nclasses = np.unique(labels).shape[0]
        if self._nclasses == 2:
            self._nclasses = 1

        self._train_loader, self._val_loader = self._get_train_loaders(
            inputs,
            labels
        )

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ Fits mlp classification head using training data from set_training_data and hyperparameters

            Keyword Arguments:
                timeout {float} -- timeout, considered (default: {None})
                iterations {int} -- iterations, considered (default: {None})

            Returns:
                CallResult[None]
        """

        if iterations is None:
            iterations = self.hyperparams["epochs"]
            has_finished = True
        else:
            has_finished = False

        if self._nclasses > 2:
            class_weights = torch.Tensor([
                1 / val_ct for val_ct in self._value_counts
            ]).to(self._device)
            criterion = nn.CrossEntropyLoss(weight = class_weights)
        else:
            class_weight = torch.Tensor([
                self._value_counts.iloc[0] / self._value_counts.iloc[1]
            ]).to(self._device)
            criterion = nn.BCEWithLogitsLoss(pos_weight = class_weight)

        self._clf_model = self._build_clf_model(
            self.hyperparams['feature_dim'],
            self._nclasses
        ).to(self._device)
        if os.path.isfile(self.hyperparams['weights_filepath']):
            self._clf_model.load_state_dict(
                torch.load(self.hyperparams['weights_filepath'])
            )

        optimizer = Adam(
            self._clf_model.parameters(),
            lr=self.hyperparams['learning_rate']
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode='min',
            verbose=True
        )

        st = time()
        for epoch in range(iterations):

            self._clf_model = self._clf_model.train()
            for train_inputs, train_labels in tqdm(self._train_loader):
                optimizer.zero_grad()
                loss = self._get_loss(
                    train_inputs,
                    train_labels,
                    criterion
                )
                loss.backward()
                optimizer.step()

            self._clf_model = self._clf_model.eval()
            val_losses = []
            with torch.no_grad():
                for val_inputs, val_labels in tqdm(self._val_loader):
                    val_loss = self._get_loss(
                        val_inputs,
                        val_labels,
                        criterion
                    )
                    val_losses.append(val_loss.item())
            scheduler.step(np.sum(val_losses))

            if epoch % 10 == 0:
                logger.info(
                    f'Epoch: {epoch+1}/{iterations}, ' +
                    f'Val Loss: {round(np.sum(val_losses),2)}, '
                )

        logger.info(f'Finished training, took {time() - st}s')
        self._is_fit = True
        torch.save(self._clf_model.state_dict(), self.hyperparams['weights_filepath'])

        return CallResult(None, has_finished=has_finished)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """ Produce primitive's predictions

            Arguments:
                inputs {Inputs} -- D3M dataframe containing attributes

            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})
        """

        clf_model, test_loader = self._prepare_test_inputs(inputs)

        if self._all_outputs is None:
            all_outputs = []
            with torch.no_grad():
                for test_inputs in tqdm(test_loader):
                    test_inputs = test_inputs[0].to(self._device)
                    test_outputs = clf_model(test_inputs)
                    all_outputs.append(test_outputs)
            all_outputs = torch.cat(all_outputs)
        else:
            all_outputs = self._all_outputs

        if self._nclasses > 1:
            all_probs = nn.functional.softmax(all_outputs, dim = 1)
            all_classes = [i for i in range(self._nclasses)]
        else:
            all_probs = nn.functional.sigmoid(all_outputs)
            all_classes = [1]

        if self.hyperparams['all_confidences']:
            index = np.repeat(
                range(all_probs.shape[0]),
                self._nclasses
            )
            output_labels = self._label_encoder.inverse_transform(all_classes)
            all_preds = np.tile(output_labels, all_probs.shape[0])
            all_probs = all_probs.cpu().data.numpy().flatten()
        else:
            index = None
            all_probs, all_preds = torch.max(all_probs, 1)
            all_preds = all_preds.cpu().data.numpy()
            all_preds = self._label_encoder.inverse_transform(all_preds)
            all_probs = all_probs.cpu().data.numpy()

        preds_df = d3m_DataFrame(
            pd.DataFrame(
                np.vstack((all_preds, all_probs)).T,
                columns = [self._output_column, 'confidence'],
                index = index
            ),
            generate_metadata = True
        )

        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget"
        )
        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/Score",
        )
        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        )
        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "http://schema.org/Float"
        )

        return CallResult(preds_df)

    def produce_explanations(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """ Produce explanation masks for primitive's predictions

            Arguments:
                inputs {Inputs} -- D3M dataframe containing attributes

            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})
        """

        clf_model, test_loader = self._prepare_test_inputs(inputs)

        if self.hyperparams['explain_all_classes']:
            all_class_masks = [[] for _ in range(self._nclasses)]
        else:
            all_class_masks = [[]]

        all_outputs = []
        for test_inputs in tqdm(test_loader):
            test_inputs = test_inputs[0].to(self._device)
            test_inputs.requires_grad = True
            test_outputs = clf_model(test_inputs)
            all_outputs.append(test_outputs)
            
            one_hots = self._get_one_hots(test_outputs)
            for i, one_hot in enumerate(one_hots):
                masks = self._get_masks(clf_model, test_inputs, test_outputs, one_hot)
                masks = self._resize_masks(masks, self.hyperparams['image_dim'])
                all_class_masks[i].append(masks)

        all_class_masks = [list(np.concatenate(masks).tolist()) for masks in all_class_masks]
        self._all_outputs = torch.cat(all_outputs)

        explain_df = pd.DataFrame()
        for i, masks in enumerate(all_class_masks):
            explain_df[f'class_{i}'] = masks

        if not self.hyperparams['explain_all_classes']:
            explain_df.columns = ['class_argmax']

        explain_df = d3m_DataFrame(explain_df, generate_metadata=False)
        return CallResult(explain_df)

    def _build_clf_model(self, dim_mlp, clf_classes):
        """ build classification head starting from unpooled features """
        return nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(dim_mlp, dim_mlp),
            nn.BatchNorm1d(dim_mlp),
            nn.ReLU(),
            nn.Linear(dim_mlp, clf_classes)
        )

    def _get_loss(self, inputs, labels, criterion):
        """ get loss from batch of inputs and labels"""
        inputs = inputs.to(self._device)
        labels = labels.to(self._device)
        outputs = self._clf_model(inputs)
        return criterion(outputs, labels)

    def _prepare_test_inputs(self, inputs):
        """ prepare test inputs and model to produce either predictions or explanations"""
        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        spatial_dim = int(math.sqrt(
            inputs.values.shape[1] / self.hyperparams['feature_dim']
        ))
        features = inputs.values.reshape(
            -1,
            self.hyperparams['feature_dim'],
            spatial_dim,
            spatial_dim
        )
        features = torch.Tensor(features)
        test_dataset = TensorDataset(features)

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=False
        )

        model = self._build_clf_model(
            self.hyperparams['feature_dim'],
            self._nclasses
        ).to(self._device)
        model.load_state_dict(
            torch.load(self.hyperparams['weights_filepath'])
        )
        model = model.eval()

        return model, test_loader

    def _get_train_loaders(self, inputs, outputs):
        """ build training and validation datasets and data loaders from inputs and ouputs """
        spatial_dim = int(math.sqrt(
            inputs.values.shape[1] // self.hyperparams['feature_dim']
        ))
        features = inputs.values.reshape(
            -1,
            self.hyperparams['feature_dim'],
            spatial_dim,
            spatial_dim
        )

        if self._value_counts.min() == 1:
            stratify = None
        else:
            stratify = outputs

        f_train, f_test, tgt_train, tgt_test = train_test_split(
            features,
            outputs,
            test_size = 0.1,
            random_state = self.random_seed,
            stratify=stratify
        )

        if self._nclasses > 2:
            train_labels = torch.LongTensor(tgt_train)
            val_labels = torch.LongTensor(tgt_test)
        else:
            train_labels = torch.FloatTensor(tgt_train).unsqueeze(-1)
            val_labels = torch.FloatTensor(tgt_test).unsqueeze(-1)

        train_dataset = TensorDataset(
            torch.Tensor(f_train),
            train_labels
        )

        val_dataset = TensorDataset(
            torch.Tensor(f_test),
            val_labels
        )

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.hyperparams['batch_size'],
            shuffle=False
        )

        return train_loader, val_loader

    def _get_one_hots(self, test_outputs):
        """ get list of one hot outputs for each class we are explaining """

        if not self.hyperparams['explain_all_classes']:
            max_idxs = np.argmax(test_outputs.cpu().data.numpy(), axis = 1)
            one_hot = np.zeros(test_outputs.shape, dtype=np.float32)
            for i, idx in enumerate(max_idxs):
                one_hot[i][idx] = 1
            one_hots = [one_hot]
        else:
            one_hots = []
            for i in range(self._nclasses):
                one_hot = np.zeros(test_outputs.shape, dtype=np.float32)
                one_hot[:,i] = 1
                one_hots.append(one_hot)
        one_hots = [torch.from_numpy(o_h).to(self._device) for o_h in one_hots]
        one_hots = [o_h.requires_grad_(True) for o_h in one_hots]
        return one_hots

    def _get_masks(self, clf_model, test_inputs, test_outputs, one_hot):
        """ get GradCam mask given batch of inputs, outputs, and classes to explain """

        one_hot = torch.sum(one_hot * test_outputs)
        clf_model.zero_grad()
        one_hot.backward(retain_graph=True)
        grads_val = test_inputs.grad.cpu().data.numpy()
        features = test_inputs.cpu().data.numpy()
        weights = np.mean(grads_val, axis=(2,3))
        cam_mask = np.sum(
            weights[:,:,np.newaxis,np.newaxis] * features,
            axis = 1
        )
        return cam_mask

    def _resize_masks(self, masks, input_dim):
        """ resize masks to dimension of input image"""

        all_masks = []
        masks = np.maximum(masks, 0)
        for mask in masks:
            mask = cv2.resize(mask, (input_dim, input_dim))
            mask = mask - np.min(mask)
            mask = mask / np.max(mask)
            all_masks.append(mask)
        all_masks = np.stack(all_masks)
        return all_masks
