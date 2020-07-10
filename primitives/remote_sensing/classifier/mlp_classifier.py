import os.path
import typing
import sys
from time import time
import logging

import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
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
    spatial_dim = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=4,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="spatial dimension (height and with) after reshaping flattened feature vector",
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
        default = 500, 
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
            }
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
    
    def get_params(self) -> Params:
        return Params(
            is_fit = self._is_fit,
            output_column = self._output_column,
            nclasses = self._nclasses,
        )

    def set_params(self, *, params: Params) -> None:
        self._is_fit = params['is_fit']
        self._output_column = params['output_column']
        self._nclasses = params['nclasses']

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ Sets primitive's training data
        
            Arguments:
                inputs {Inputs} -- D3M dataframe containing features
                outputs {Outputs} -- D3M dataframe containing targets
            
        """
        
        self._output_column = outputs.columns[0]

        features = inputs.values.reshape(
            -1, 
            self.hyperparams['feature_dim'],
            self.hyperparams['spatial_dim'],
            self.hyperparams['spatial_dim']
        )

        if outputs.iloc[0].value_counts().min() == 1:
            stratify = None
        else:
            stratify = outputs.values
        f_train, f_test, tgt_train, tgt_test = train_test_split(
            features, 
            outputs.values,
            test_size = 0.1,
            random_state = self.random_seed,
            stratify=stratify
        )

        train_dataset = TensorDataset(
            torch.Tensor(f_train),
            torch.LongTensor(tgt_train).squeeze() #need?
        )       

        val_dataset = TensorDataset(
            torch.Tensor(f_test),
            torch.LongTensor(tgt_test).squeeze() #need?
        )  
        
        self._val_size = tgt_test.shape[0]
        self._train_size = tgt_train.shape[0]

        self._train_loader = DataLoader(
            train_dataset, 
            batch_size=self.hyperparams['batch_size'],
            shuffle=True
        )
        self._val_loader = DataLoader(
            val_dataset, 
            batch_size=self.hyperparams['batch_size'],
            shuffle=False
        )
        
        self._nclasses = np.unique(outputs.values).shape[0]
        self._clf_model = self._build_clf_model(
            features.shape[1],
            self._nclasses
        ).to(self._device)

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

        criterion = nn.CrossEntropyLoss()
        optimizer = Adam(
            self._clf_model.parameters(), 
            lr=self.hyperparams['learning_rate']
        )
        scheduler = ReduceLROnPlateau(
            optimizer, 
            mode='max',
            verbose=True
        )

        st = time()
        for epoch in range(iterations):
            
            train_tp = 0
            self._clf_model = self._clf_model.train()
            for train_inputs, train_labels in tqdm(self._train_loader):
                optimizer.zero_grad()
                train_inputs = train_inputs.to(self._device)
                train_outputs = self._clf_model(train_inputs)
                _, train_pred = torch.max(train_outputs.data, 1)
                train_tp += (train_pred == train_labels).sum().item()
                loss = criterion(train_outputs, train_labels)
                loss.backward()
                optimizer.step()
            
            val_tp = 0
            self._clf_model = self._clf_model.eval()
            with torch.no_grad():
                for val_inputs, val_labels in tqdm(self._val_loader):
                    val_inputs = val_inputs.to(self._device)
                    val_outputs = self._clf_model(val_inputs)
                    _, val_pred = torch.max(val_outputs.data, 1)
                    val_tp += (val_pred == val_labels).sum().item()
            
            train_acc = 100 * train_tp / self._train_size
            val_acc = 100 * val_tp / self._val_size
            scheduler.step(val_acc)

            logger.info(
                f'Epoch: {epoch+1}/{iterations}, ' +
                f'Train Loss: {round(loss.item(),2)}, ' +
                f'Train Acc: {round(train_acc,2)}, Val Acc: {round(val_acc,2)}'
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
        
        _, all_preds = torch.max(all_outputs, 1)
        all_preds = all_preds.cpu().data.numpy()

        preds_df = d3m_DataFrame(
            pd.DataFrame(all_preds, columns = [self._output_column]),
            generate_metadata = True
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
        
        all_class_masks = [list(np.concatenate(masks)) for masks in all_class_masks]
        self._all_outputs = torch.cat(all_outputs)

        explain_df = pd.DataFrame()
        for i, masks in enumerate(all_class_masks):
            explain_df[f'class_{i}'] = masks
        
        if not self.hyperparams['explain_all_classes']:
            explain_df.columns = ['class_argmax']
        
        explain_df = d3m_DataFrame(explain_df, generate_metadata=False)
        #explain_df.metadata = explain_df.metadata.generate(explain_df, compact=True)
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

    def _prepare_test_inputs(self, inputs):
        """ prepare test inputs and model to produce either predictions or explanations"""

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")
    
        features = inputs.values.reshape(
            -1, 
            self.hyperparams['feature_dim'],
            self.hyperparams['spatial_dim'],
            self.hyperparams['spatial_dim']
        )
        features = torch.Tensor(features)
        test_dataset = TensorDataset(features)

        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.hyperparams['batch_size'],
            shuffle=False
        )
        
        model = self._build_clf_model(
            features.shape[1],
            self._nclasses
        ).to(self._device)
        model.load_state_dict(
            torch.load(self.hyperparams['weights_filepath'])
        )
        model = model.eval()

        return model, test_loader

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