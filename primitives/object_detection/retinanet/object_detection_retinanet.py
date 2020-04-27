import os
import sys
import warnings
import typing
import time

import keras
import keras.preprocessing.image
import tensorflow as tf
import pandas as pd
import numpy as np

from object_detection_retinanet import layers
from object_detection_retinanet import losses
from object_detection_retinanet import models

from collections import OrderedDict

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import base as metadata_base, hyperparams, params
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from object_detection_retinanet.callbacks import RedirectModel
from object_detection_retinanet.callbacks.eval import Evaluate
from object_detection_retinanet.utils.eval import evaluate
from object_detection_retinanet.models.retinanet import retinanet_bbox
from object_detection_retinanet.preprocessing.csv_generator import CSVGenerator
from object_detection_retinanet.utils.anchors import make_shapes_callback
from object_detection_retinanet.utils.model import freeze as freeze_model
from object_detection_retinanet.utils.gpu import setup_gpu
from object_detection_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

__author__ = 'Distil'
__version__ = '0.1.0'
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    backbone = hyperparams.Union(
        OrderedDict({
            'resnet50': hyperparams.Constant[str](
                default = 'resnet50',
                semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
                description = "Backbone architecture from resnet50 architecture (https://arxiv.org/abs/1512.03385)"
            )
            # 'resnet101': hyperparams.Constant[str](
            #     default = 'resnet101',
            #     semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            #     description = "Backbone architecture from resnet101 architecture (https://arxiv.org/abs/1512.03385)"
            # ),
            # 'resnet152': hyperparams.Constant[str](
            #     default = 'resnet152',
            #     semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
            #     description = "Backbone architecture from resnet152 architecture (https://arxiv.org/abs/1512.03385)"
            # )
        }),
        default = 'resnet50',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Backbone architecture from which RetinaNet is built. All backbones " +
                      "require a weights file downloaded for use during runtime."
    )
    batch_size = hyperparams.Hyperparameter[int](
        default = 1,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Size of the batches as input to the model."
    )
    n_epochs = hyperparams.Hyperparameter[int](
        default = 1,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Number of epochs to train."
    )
    freeze_backbone = hyperparams.Hyperparameter[bool](
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Freeze training of backbone layers."
    )
    weights = hyperparams.Hyperparameter[bool](
        default = True,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Load the model with pretrained weights specific to selected backbone."
    )
    learning_rate = hyperparams.Hyperparameter[float](
        default = 1e-5,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Learning rate."
    )
    n_steps = hyperparams.Hyperparameter[int](
        default = 20,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = "Number of steps/epoch."
    )
    output = hyperparams.Hyperparameter[bool](
        default = False,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "Output images and predicted bounding boxes after evaluation."
    )
    weights_path = hyperparams.Hyperparameter[str](
        default = '/root/',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description = "An output path for where model weights should be saved/loaded from during runtime"
    )

class Params(params.Params):
    base_dir: str
    image_paths: pd.Series
    annotations: pd.DataFrame
    classes: pd.DataFrame

class ObjectDetectionRNPrimitive(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
    Primitive that utilizes RetinaNet, a convolutional neural network (CNN), for object
    detection. The methodology comes from "Focal Loss for Dense Object Detection" by
    Lin et al. 2017 (https://arxiv.org/abs/1708.02002). The code implementation is based
    off of the base library found at: https://github.com/fizyr/keras-retinanet.

    The primitive accepts a Dataset consisting of images, labels as input and returns
    a dataframe as output which include the bounding boxes for each object in each image.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
        'id': 'd921be1e-b158-4ab7-abb3-cb1b17f42639',
        'version': __version__,
        'name': 'retina_net',
        'python_path': 'd3m.primitives.object_detection.retinanet',
        'keywords': ['object detection', 'convolutional neural network', 'digital image processing', 'RetinaNet'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                'https://github.com/kungfuai/d3m-primitives',
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
            'type': "FILE",
            'key': "resnet50",
            'file_uri': "http://public.datadrivendiscovery.org/ResNet-50-model.keras.h5",
            'file_digest': "0128cdfa3963288110422e4c1a57afe76aa0d760eb706cda4353ef1432c31b9c"
            }
        ],
        'algorithm_types': [metadata_base.PrimitiveAlgorithmType.RETINANET],
        'primitive_family': metadata_base.PrimitiveFamily.OBJECT_DETECTION,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, volumes: typing.Dict[str,str] = None) -> None:
        super().__init__(hyperparams = hyperparams, volumes = volumes)
        self.image_paths = None
        self.annotations = None
        self.base_dir = None
        self.classes = None
        self.backbone = None
        self.y_true = None
        self.workers = 1
        self.multiprocessing = 1
        self.max_queue_size = 10

    def get_params(self) -> Params:
        return Params(
            base_dir = self.base_dir,
            image_paths = self.image_paths,
            annotations = self.annotations,
            classes = self.classes
        )

    def set_params(self, *, params: Params) -> None:
        self.base_dir = params['base_dir']
        self.image_paths = params['image_paths']
        self.annotations = params['annotations']
        self.classes = params['classes']

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Sets the primitive's training data and preprocesses the files for RetinaNet format.

        Parameters
        ----------
            inputs: numpy ndarray of size (n_images, dimension) containing the d3m Index, image name,
                    and bounding box for each image.

        Returns
        -------
            No returns. Function is called by pipeline at runtime.
        """

        # Prepare annotation file
        ## Generate image paths
        image_cols = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        self.base_dir = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in image_cols]
        self.image_paths = np.array([[os.path.join(self.base_dir, filename) for filename in inputs.iloc[:,col]] for self.base_dir, col in zip(self.base_dir, image_cols)]).flatten()
        self.image_paths = pd.Series(self.image_paths)

        ## Arrange proper bounding coordinates
        bounding_coords = inputs.bounding_box.str.split(',', expand = True)
        bounding_coords = bounding_coords.drop(bounding_coords.columns[[2, 5, 6, 7]], axis = 1)
        bounding_coords.columns = ['x1', 'y1', 'y2', 'x2']
        bounding_coords = bounding_coords[['x1', 'y1', 'x2', 'y2']]

        ## Generate class names
        class_name = pd.Series(['class'] * inputs.shape[0])

        ## Assemble annotation file
        self.annotations = pd.concat([self.image_paths, bounding_coords, class_name], axis = 1)
        self.annotations.columns = ['img_file', 'x1', 'y1', 'x2', 'y2', 'class_name']

        # Prepare ID file
        self.classes = pd.DataFrame({'class_name': ['class'],
                                     'class_id': [0]})

    def _create_callbacks(self, model, training_model, prediction_model):
        """
        Creates the callbacks to use during training.

        Parameters
        ----------
            model                : The base model.
            training_model       : The model that is used for training.
            prediction_model     : The model that should be used for validation.
            validation_generator : The generator for creating validation data.

        Returns
        -------
            callbacks            : A list of callbacks used for training.
        """
        callbacks = []

        callbacks.append(keras.callbacks.ReduceLROnPlateau(
            monitor   = 'loss',
            factor    = 0.1,
            patience  = 2,
            verbose   = 1,
            mode      = 'auto',
            min_delta = 0.0001,
            cooldown  = 0,
            min_lr    = 0
        ))

        return callbacks

    def _create_models(self, backbone_retinanet, num_classes, weights, freeze_backbone = False, lr = 1e-5):

        """
        Creates three models (model, training_model, prediction_model).

        Parameters
        ----------
            backbone_retinanet : A function to call to create a retinanet model with a given backbone.
            num_classes        : The number of classes to train.
            weights            : The weights to load into the model.
            multi_gpu          : The number of GPUs to use for training.
            freeze_backbone    : If True, disables learning for the backbone.
            config             : Config parameters, None indicates the default configuration.

        Returns
        -------
            model              : The base model.
            training_model     : The training model. If multi_gpu=0, this is identical to model.
            prediction_model   : The model wrapped with utility functions to perform object detection (applies regression values and performs NMS).
        """

        modifier = freeze_model if freeze_backbone else None
        anchor_params = None
        num_anchors   = None

        model = self._model_with_weights(backbone_retinanet(num_classes, num_anchors = num_anchors, modifier = modifier), weights = weights, skip_mismatch = True)
        training_model = model
        prediction_model = retinanet_bbox(model = model, anchor_params = anchor_params)
        training_model.compile(
            loss = {
                'regression'    :  losses.smooth_l1(),
                'classification':  losses.focal()
            },
            optimizer = keras.optimizers.adam(lr = lr, clipnorm = 0.001)
        )

        return model, training_model, prediction_model

    def _num_classes(self):
        """
        Number of classes in the dataset.
        """
        return max(self.classes.values()) + 1

    def _model_with_weights(self, model, weights, skip_mismatch):
        """
        Load weights for model.

        Parameters
        ----------
            model         : The model to load weights for.
            weights       : The weights to load.
            skip_mismatch : If True, skips layers whose shape of weights doesn't match with the model.

        Returns
        -------
            model         : Model with loaded weights.
        """

        if weights is not None:
            model.load_weights(weights, by_name = True, skip_mismatch = skip_mismatch)
        return model

    def _create_generator(self, annotations, classes, shuffle_groups):
        """
        Create generator for evaluation.
        """

        validation_generator = CSVGenerator(self.annotations, self.classes, self.base_dir, self.hyperparams['batch_size'], self.backbone.preprocess_image, shuffle_groups = False)
        return validation_generator

    def _fill_empty_predictions(self, empty_predictions_image_names, d3mIdx_image_mapping):
        """
        D3M metrics evaluator needs at least one prediction per image. If RetinaNet does not return
        predictions for an image, this method creates a dummy empty prediction row to add to results_df for that
        missing image.

        TODO: DUMMY CONFIDENCE SCORES LOWER AVERAGE PRECISION. FIND A FIX.
        """

        # Prepare D3M index
        empty_predictions_d3mIdx = [d3mIdx_image_mapping.get(key) for key in empty_predictions_image_names]
        empty_predictions_d3mIdx = [item for sublist in empty_predictions_d3mIdx for item in sublist]

        # Prepare dummy columns
        d3mIdx = empty_predictions_d3mIdx
        bounding_box = ["0,0,0,0,0,0,0,0"] * len(empty_predictions_d3mIdx)
        confidence = [float(0)] * len(empty_predictions_d3mIdx)

        empty_predictions_df = pd.DataFrame({
            'd3mIndex': d3mIdx,
            'bounding_box': bounding_box,
            'confidence': confidence
        })

        return empty_predictions_df

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Creates the image generators and then trains RetinaNet model on the image paths in the input
        dataframe column.

        Can choose to use validation generator.

        If no weight file is provided, the default is to use the ImageNet weights.
        """

        # Create object that stores backbone information
        self.backbone = models.backbone(self.hyperparams['backbone'])

        # Create the generators
        train_generator = CSVGenerator(self.annotations, self.classes, self.base_dir, self.hyperparams['batch_size'], self.backbone.preprocess_image)

        # Running the model
        ## Assign weights
        if self.hyperparams['weights'] is False:
            weights = None
        else:
            weights = self.volumes[self.hyperparams['backbone']]

        ## Create model
        print('Creating model...', file = sys.__stdout__)

        model, training_model, prediction_model = self._create_models(
            backbone_retinanet = self.backbone.retinanet,
            num_classes = train_generator.num_classes(),
            weights = weights,
            freeze_backbone = self.hyperparams['freeze_backbone'],
            lr = self.hyperparams['learning_rate']
        )

        model.summary()

        ### !!! vgg AND densenet BACKBONES CURRENTLY NOT IMPLEMENTED !!!
        ## Let the generator compute the backbone layer shapes using the actual backbone model
        # if 'vgg' in self.hyperparams['backbone'] or 'densenet' in self.hyperparams['backbone']:
        #     train_generator.compute_shapes = make_shapes_callback(model)
        #     if validation_generator:
        #         validation_generator.compute_shapes = train_generator.compute_shapes

        ## Set up callbacks
        callbacks = self._create_callbacks(
            model,
            training_model,
            prediction_model,
        )

        start_time = time.time()
        print('Starting training...', file = sys.__stdout__)

        training_model.fit_generator(
            generator = train_generator,
            steps_per_epoch = self.hyperparams['n_steps'],
            epochs = self.hyperparams['n_epochs'],
            verbose = 1,
            callbacks = callbacks,
            workers = self.workers,
            use_multiprocessing = self.multiprocessing,
            max_queue_size = self.max_queue_size
        )

        training_model.save_weights(self.hyperparams['weights_path'] + 'model_weights.h5')

        print(f'Training complete. Training took {time.time()-start_time} seconds.', file = sys.__stdout__)
        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce image detection predictions.

        Parameters
        ----------
            inputs  : numpy ndarray of size (n_images, dimension) containing the d3m Index, image name,
                      and bounding box for each image.

        Returns
        -------
            outputs : A d3m dataframe container with the d3m index, image name, bounding boxes as
                      a string (8 coordinate format), and confidence scores.
        """
        iou_threshold = 0.5     # Bounding box overlap threshold for false positive or true positive
        score_threshold = 0.05  # The score confidence threshold to use for detections
        max_detections = 100    # Maxmimum number of detections to use per image

        # Create object that stores backbone information
        backbone = models.backbone(self.hyperparams['backbone'])

        # Create the generators
        train_generator = CSVGenerator(self.annotations, self.classes, self.base_dir, self.hyperparams['batch_size'], backbone.preprocess_image)

        # Assign weights
        if self.hyperparams['weights'] is False:
            weights = None
        else:
            weights = self.volumes[self.hyperparams['backbone']]

        # Instantiate model
        model, training_model, prediction_model = self._create_models(
            backbone_retinanet = backbone.retinanet,
            num_classes = train_generator.num_classes(),
            weights = weights,
            freeze_backbone = self.hyperparams['freeze_backbone'],
            lr = self.hyperparams['learning_rate']
        )

        # Load model weights saved in fit
        training_model.load_weights(self.hyperparams['weights_path'] + 'model_weights.h5')

        # Convert training model to inference model
        inference_model = models.convert_model(training_model)

        # Generate image paths
        image_cols = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        self.base_dir = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in image_cols]
        self.image_paths = np.array([[os.path.join(self.base_dir, filename) for filename in inputs.iloc[:,col]] for self.base_dir, col in zip(self.base_dir, image_cols)]).flatten()
        self.image_paths = pd.Series(self.image_paths)

        # Initialize output objects
        box_list = []
        score_list = []
        image_name_list = []

        # Predict bounding boxes and confidence scores for each image
        image_list = [x for i, x in enumerate(self.image_paths.tolist()) if self.image_paths.tolist().index(x) == i]

        start_time = time.time()
        print('Starting testing...', file = sys.__stdout__)

        for i in image_list:
            image = read_image_bgr(i)

            # preprocess image for network
            image = preprocess_image(image)
            image, scale = resize_image(image)

            boxes, scores, labels = inference_model.predict_on_batch(tf.constant(np.expand_dims(image, axis = 0), dtype = tf.float32))

            # correct for image scale
            boxes /= scale

            for box, score in zip(boxes[0], scores[0]):
                if score < 0.5:
                    break

                b = box.astype(int)
                box_list.append(b)
                score_list.append(score)
                image_name_list.append(i * len(b))

        print(f'Testing complete. Testing took {time.time()-start_time} seconds.', file = sys.__stdout__)

        ## Convert predicted boxes from a list of arrays to a list of strings
        boxes = np.array(box_list).tolist()
        boxes = list(map(lambda x : [x[0], x[1], x[0], x[3], x[2], x[3], x[2], x[1]], boxes))  # Convert to 8 coordinate format for D3M
        boxes = list(map(lambda x : ",".join(map(str, x)), boxes))

        # Create mapping between image names and D3M index
        input_df = pd.DataFrame({
            'd3mIndex': inputs.d3mIndex,
            'image': [os.path.basename(list) for list in self.image_paths]
        })

        d3mIdx_image_mapping = input_df.set_index('image').T.to_dict('list')

        # Extract values for image name keys and get missing image predictions (if they exist)
        image_name_list = [os.path.basename(list) for list in image_name_list]
        d3mIdx = [d3mIdx_image_mapping.get(key) for key in image_name_list]
        empty_predictions_image_names = [k for k,v in d3mIdx_image_mapping.items() if v not in d3mIdx]
        d3mIdx = [item for sublist in d3mIdx for item in sublist]   # Flatten list of lists

        ## Assemble in a Pandas DataFrame
        results = pd.DataFrame({
            'd3mIndex': d3mIdx,
            'bounding_box': boxes,
            'confidence': score_list
        })

        # D3M metrics evaluator needs at least one prediction per image. If RetinaNet does not return
        # predictions for an image, create a dummy empty prediction row to add to results_df for that
        # missing image.
        if len(empty_predictions_image_names) != 0:
            # Create data frame of empty predictions for missing each image and concat with results.
            # Sort results_df.
            empty_predictions_df = self._fill_empty_predictions(empty_predictions_image_names, d3mIdx_image_mapping)
            results_df = pd.concat([results, empty_predictions_df]).sort_values('d3mIndex')
        else:
            results_df = results

        # Convert to DataFrame container
        results_df = d3m_DataFrame(results_df)

        ## Assemble first output column ('d3mIndex)
        col_dict = dict(results_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'd3mIndex'
        col_dict['semantic_types'] = ('http://schema.org/Integer',
                                      'https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        results_df.metadata = results_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

        ## Assemble second output column ('bounding_box')
        col_dict = dict(results_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'bounding_box'
        col_dict['semantic_types'] = ('http://schema.org/Text',
                                      'https://metadata.datadrivendiscovery.org/types/PredictedTarget',
                                      'https://metadata.datadrivendiscovery.org/types/BoundingPolygon')
        results_df.metadata = results_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        ## Assemble third output column ('confidence')
        col_dict = dict(results_df.metadata.query((metadata_base.ALL_ELEMENTS, 2)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'confidence'
        col_dict['semantic_types'] = ('http://schema.org/Integer',
                                      'https://metadata.datadrivendiscovery.org/types/Score')
        results_df.metadata = results_df.metadata.update((metadata_base.ALL_ELEMENTS, 2), col_dict)

        return CallResult(results_df)