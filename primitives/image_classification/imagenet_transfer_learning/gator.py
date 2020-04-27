import os
import sys
import typing
import numpy as np
import pandas as pd
import time
from d3m.primitive_interfaces.base import CallResult, PrimitiveBase
from ..utils.imagenet import ImagenetModel, ImageNetGen
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array, load_img

from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp
from sklearn.preprocessing import LabelEncoder 
import logging
from d3m.exceptions import PrimitiveNotFittedError

__author__ = 'Distil'
__version__ = '1.0.2'
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

class Params(params.Params):
    label_encoder: LabelEncoder
    output_columns: pd.Index
    weights_path: str

class Hyperparams(hyperparams.Hyperparams):
    weights_filepath = hyperparams.Hyperparameter[str](
        default='model_weights.h5',
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="weights of trained model will be saved to this filepath",
    )
    pooling = hyperparams.Enumeration(
        default = 'avg', 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        values = ['avg', 'max'],
        description = 'whether to use average or max pooling to transform 4D ImageNet features to 2D output'
    )
    dense_dim = hyperparams.UniformInt(
        lower = 128, 
        upper = 4096,
        upper_inclusive=True, 
        default = 1024, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'dimension of classification head (1 single dense layer)'
    )
    batch_size = hyperparams.UniformInt(
        lower = 1, 
        upper = 256,
        upper_inclusive=True, 
        default = 32, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'batch size'
    )
    top_layer_epochs = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize,
        default = 100, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'how many epochs for which to finetune classification head (happens first)'
    )
    all_layer_epochs = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize,
        default = 100, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'how many epochs for which to finetune entire model (happens second)'
    )
    unfreeze_proportions = hyperparams.Set(
        elements=hyperparams.Hyperparameter[float](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="""list of proportions representing how much of the base ImageNet model one wants to
                    unfreeze (later layers unfrozen) for another round of finetuning"""
    )
    early_stopping_patience = hyperparams.UniformInt(
        lower = 0, 
        upper = sys.maxsize, 
        default = 5, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = """number of epochs to wait before invoking early stopping criterion. applied to all 
            iterations of finetuning""")
    val_split = hyperparams.Uniform(
        lower = 0.0, 
        upper = 1.0, 
        default = 0.2, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/TuningParameter'], 
        description = 'proportion of training records to set aside for validation. Ignored \
            if iterations flag in `fit` method is not None')
    include_class_weights = hyperparams.UniformBool(
        default = True, 
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description="whether to include class weights in finetuning of ImageNet model"
    )

class GatorPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Produce image classification predictions by iteratively finetuning an Inception V3 model
        trained on ImageNet (can have multiple columns of images, but assumption is that there is a
        single column of target labels, these labels are broadcast to all images by row)

        Training inputs: 1) Feature dataframe, 2) Label dataframe
        Outputs: Dataframe with predictions

    """

    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "475c26dc-eb2e-43d3-acdb-159b80d9f099",
        'version': __version__,
        'name': "gator",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Image Recognition', 'transfer learning', 'classification', 'ImageNet', 'Convolutional Neural Network'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
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
            "key": "gator_weights",
            "file_uri": "http://public.datadrivendiscovery.org/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5",
            "file_digest":"9617109a16463f250180008f9818336b767bdf5164315e8cd5761a8c34caa62a"
            },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.classification.inceptionV3_image_feature.Gator',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        "algorithm_types": [
            metadata_base.PrimitiveAlgorithmType.IMAGENET
        ],
        "primitive_family": metadata_base.PrimitiveFamily.CLASSIFICATION
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:

        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)

        # set seed for reproducibility
        tf.random.set_seed(random_seed)

        self.class_weights = None
        self.targets = None
        self.ImageNet = ImagenetModel( 
            weights = self.volumes["gator_weights"], 
            pooling = self.hyperparams['pooling'],
            clf_head_dense_dim=self.hyperparams['dense_dim']
        )

    def get_params(self) -> Params:
        if not self._is_fit:
            return Params(
                label_encoder=None,
                output_columns=None,
            )
        
        return Params(
            label_encoder=self.encoder,
            output_columns=self.output_columns,
        )

    def set_params(self, *, params: Params) -> None:
        self.encoder = params['label_encoder']
        self.output_columns = params['output_columns']
        self._is_fit = all(param is not None for param in params.values())

    def _image_array_from_path(self, fpath, target_size=(299, 299)):
        img = load_img(fpath, target_size=target_size)
        return img_to_array(img)

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
            Sets primitive's training data
            
            Parameters
            ----------
            inputs: feature dataframe
            outputs: labels from dataframe's target column
        '''

        self.output_columns = outputs.columns

        # create single list of image paths from all target image columns
        image_cols = inputs.metadata.get_columns_with_semantic_type('http://schema.org/ImageObject')
        base_paths = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') 
            for t in image_cols]
        image_paths = np.array([[os.path.join(base_path, filename) 
            for filename in inputs.iloc[:,col]] 
            for base_path, col in zip(base_paths, image_cols)]).flatten()

        # preprocess images for ImageNet model 
        images_array = np.array(
            [self._image_array_from_path(fpath, target_size=self.ImageNet.target_size) for fpath in image_paths]
        )
        logger.debug(f'preprocessing {images_array.shape[0]} images')
        if images_array.ndim != 4:
            raise Exception('invalid input shape for images_array, expects a 4d array')
        self._X_train = self.ImageNet.preprocess(images_array)

        # broadcast image labels for each column of images
        if outputs.shape[1] > 1:
            raise ValueError('There are multiple columns labeled as target, but this primitive expects only one')

        # train label encoder
        self.encoder = LabelEncoder().fit(outputs.values.ravel())
        image_labels = \
            self.encoder.transform(np.repeat(outputs.values.ravel(), len(image_cols)))
        self._y_train = to_categorical(image_labels)

        # calculate class weights for target labels if desired
        if self.hyperparams['include_class_weights']:
           self.class_weights = dict(pd.Series(image_labels).value_counts())


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
            Trains a single Inception model on all columns of image paths using dataframe's target column
        '''

        # break out validation set if iterations arg not set
        if iterations is None:
            iterations_set = False
            train_split = (1 - self.hyperparams['val_split']) * self._X_train.shape[0]
            x_train = self._X_train[:int(train_split)]
            y_train = self._y_train[:int(train_split)]
            x_val = self._X_train[int(train_split):]
            y_val = self._y_train[int(train_split):]
            val_dataset = ImageNetGen(x_val, 
                y = y_val, 
                batch_size = self.hyperparams['batch_size'])
            top_layer_iterations = self.hyperparams['top_layer_epochs']
            all_layer_iterations = self.hyperparams['all_layer_epochs']
            callbacks = [EarlyStopping(monitor='val_loss', 
                patience=self.hyperparams['early_stopping_patience'], 
                restore_best_weights=False)]   
        else:
            iterations_set = True
            top_layer_iterations = iterations
            all_layer_iterations = iterations
            x_train = self._X_train
            y_train = self._y_train
            val_dataset = None
            callbacks = None
        train_dataset = ImageNetGen(x_train, 
            y = y_train, 
            batch_size = self.hyperparams['batch_size'])

        # time training for 1 epoch so we can consider timeout argument thoughtfully
        if timeout:
            logger.info('Timing the fitting procedure for one epoch so we \
                can consider timeout thoughtfully')
            start_time = time.time()
            fitting_histories = self.ImageNet.finetune(train_dataset, 
                val_dataset = val_dataset, 
                nclasses = len(self.encoder.classes_),
                top_layer_epochs = 1,
                unfreeze_proportions = self.hyperparams['unfreeze_proportions'],
                all_layer_epochs = 1, 
                class_weight = self.class_weights,
                optimizer_top = 'rmsprop',
                optimizer_full = SGD(lr=0.0001, momentum=0.9),
                callbacks = callbacks,
            )
            epoch_time_estimate = time.time() - start_time

            # (1 + len(self.hyperparams['unfreeze_proportions'])) how many times we will call 'fit' on model
            timeout_epochs = \
                timeout // ((1 + len(self.hyperparams['unfreeze_proportions'])) * epoch_time_estimate) - 1 # subract 1 more to be safe
            top_layer_iters = min(timeout_epochs, top_layer_iterations)
            all_layer_iters = min(timeout_epochs, all_layer_iterations)
        else:
            top_layer_iters = top_layer_iterations
            all_layer_iters = all_layer_iterations

        # normal fitting 
        start_time = time.time()
        logger.info('finetuning begins!')
        fitting_histories = self.ImageNet.finetune(train_dataset, 
            val_dataset = val_dataset, 
            nclasses = len(self.encoder.classes_),
            top_layer_epochs = top_layer_iters,
            unfreeze_proportions = self.hyperparams['unfreeze_proportions'],
            all_layer_epochs = all_layer_iters, 
            class_weight = self.class_weights,
            optimizer_top = 'rmsprop',
            optimizer_full = SGD(lr=0.0001, momentum=0.9),
            callbacks = callbacks,
            save_weights_path = self.hyperparams['weights_filepath'],
        )
        iterations_completed = sum([len(h.history['loss']) for h in fitting_histories])
        logger.info(f'finetuning ends!. it took {time.time()-start_time} seconds')

        # maintain primitive state (mark that training data has been used)
        self._is_fit = True

        # use fitting history to set CallResult return values
        if iterations_set:
            has_finished = False
        elif top_layer_iters < top_layer_iterations or all_layer_iters < all_layer_iterations:
            has_finished = False
        else:
            has_finished = self._is_fit

        return CallResult(None, has_finished = has_finished, iterations_done = iterations_completed)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
            Produce image object classification predictions

            Parameters
            ----------
            inputs : feature dataframe

            Returns
            -------
            output : A dataframe with image labels/classifications/cluster assignments
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")
        
        result_df = pd.DataFrame({})

        image_cols = inputs.metadata.get_columns_with_semantic_type('http://schema.org/ImageObject')
        for idx, col in enumerate(image_cols):
            base_path = inputs.metadata.query((metadata_base.ALL_ELEMENTS, col))['location_base_uris'][0].replace('file:///', '/')
            image_paths = np.array([os.path.join(base_path, filename) for filename in inputs.iloc[:,col]])

            # preprocess images for ImageNet model 
            images_array = np.array(
                [self._image_array_from_path(fpath, target_size=self.ImageNet.target_size) for fpath in image_paths]
            )
            logger.debug(f'preprocessing {images_array.shape[0]} images')
            if images_array.ndim != 4:
                raise Exception('invalid input shape for images_array, expects a 4d array')
            X_test = self.ImageNet.preprocess(images_array)
            test_dataset = ImageNetGen(
                X_test, 
                batch_size = self.hyperparams['batch_size']
            )

            # make predictions on finetuned model and decode
            preds = self.ImageNet.finetune_classify(
                test_dataset, 
                nclasses = len(self.encoder.classes_),
                load_weights_path = self.hyperparams['weights_filepath']
            )
            preds = self.encoder.inverse_transform(np.argmax(preds, axis=1))
            result_df[self.output_columns[idx]] = preds

        # create output frame with metadata
        result_df = container.DataFrame(result_df, generate_metadata=True)
        for i in range(result_df.shape[1]):
            result_df.metadata = result_df.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, i), 
                ('https://metadata.datadrivendiscovery.org/types/PredictedTarget'))
        
        # ok to set to True because we have checked that primitive has been fit
        return CallResult(result_df, has_finished=True)

    
