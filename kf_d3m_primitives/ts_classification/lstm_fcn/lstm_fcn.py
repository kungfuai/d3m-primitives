import sys
import os
import time
import logging

import numpy as np
import pandas as pd
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.exceptions import PrimitiveNotFittedError
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import LabelEncoder

from ..utils.lstm_model_utils import (
    generate_lstmfcn,
    LSTMSequence,
    LSTMSequenceTest,
)

__author__ = "Distil"
__version__ = "1.2.0"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)

Inputs = container.DataFrame
Outputs = container.DataFrame


class Params(params.Params):
    label_encoder: LabelEncoder
    output_columns: pd.Index
    ts_sz: int
    n_classes: int


class Hyperparams(hyperparams.Hyperparams):
    weights_filepath = hyperparams.Hyperparameter[str](
        default='model_weights.h5',
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="weights of trained model will be saved to this filepath",
    )
    attention_lstm = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="whether to use attention in the lstm component of the model",
    )
    lstm_dim = hyperparams.UniformInt(
        lower=8,
        upper=256,
        default=128,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of cells to use in the lstm component of the model",
    )
    epochs = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=5000,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of training epochs",
    )
    learning_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=1e-3,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="learning rate",
    )
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=256,
        default=32,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="batch size",
    )
    dropout_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.2,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="dropout rate (before lstm layer in model)",
    )
    val_split = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.2,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="proportion of training records to set aside for validation. Ignored " +
            "if iterations flag in `fit` method is not None",
    )
    early_stopping_patience = hyperparams.UniformInt(
        lower=0,
        upper=sys.maxsize,
        default=100,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of epochs to wait before invoking early stopping criterion",
    )


class LstmFcnPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Primitive that applies a LSTM FCN (LSTM fully convolutional network) for time
        series classification. The implementation is based off this paper: 
        https://ieeexplore.ieee.org/document/8141873 and this base library: 
        https://github.com/NewKnowledge/LSTM-FCN.
    
        Arguments:
            hyperparams {Hyperparams} -- D3M Hyperparameter object
        
        Keyword Arguments:
            random_seed {int} -- random seed (default: {0})
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "a55cef3a-a7a9-411e-9dde-5c935ff3504b",
            "version": __version__,
            "name": "lstm_fcn",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": [
                "Time Series",
                "convolutional neural network",
                "cnn",
                "lstm",
                "time series classification",
            ],
            "source": {
                "name": __author__,
                "contact": __contact__,
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
            ],
            # The same path the primitive is registered with entry points in setup.py.
            "python_path": "d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_CLASSIFICATION,
            "can_use_gpus": True
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # set seed for reproducibility
        # tf.random.set_seed(random_seed)

        self._is_fit = False

    def get_params(self) -> Params:
        if not self._is_fit:
            return Params(
                label_encoder=None,
                output_columns=None,
                ts_sz=None,
                n_classes=None
            )
        
        return Params(
            label_encoder=self._label_encoder,
            output_columns=self._output_columns,
            ts_sz=self._ts_sz,
            n_classes=self._n_classes
        )

    def set_params(self, *, params: Params) -> None:
        self._label_encoder = params['label_encoder']
        self._output_columns = params['output_columns']
        self._ts_sz = params['ts_sz']
        self._n_classes = params['n_classes']
        self._is_fit = all(param is not None for param in params.values())

    def _get_cols(self, input_metadata):
        """ private util function that finds grouping column from input metadata
        
        Arguments:
            input_metadata {D3M Metadata object} -- D3M Metadata object for input frame
        
        Returns:
            list[int] -- list of column indices annotated with GroupingKey metadata
        """

        # find column with ts value through metadata
        grouping_column = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/GroupingKey",)
        )
        return grouping_column

    def _get_value_col(self, input_metadata):
        """
        private util function that finds the value column from input metadata

        Arguments:
        input_metadata {D3M Metadata object} -- D3M Metadata object for input frame

        Returns:
        int -- index of column that contains time series value after Time Series Formatter primitive
        """

        # find attribute column but not file column
        attributes = input_metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Attribute',))
        # this is assuming alot, but timeseries formaters typicaly place value column at the end
        attribute_col = attributes[-1]
        return attribute_col

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ Sets primitive's training data

            Arguments:
                inputs {Inputs} -- D3M dataframe containing attributes
                outputs {Outputs} -- D3M dataframe containing targets
        """

        # load and reshape training data
        self._output_columns = outputs.columns
        outputs = np.array(outputs)
        n_ts = outputs.shape[0]
        self._ts_sz = inputs.shape[0] // n_ts

        attribute_col = self._get_value_col(inputs.metadata)
        self._X_train = inputs.iloc[:, attribute_col].values.reshape(n_ts, 1, self._ts_sz)
        y_train = np.array(outputs)

        # encode labels and convert to categorical
        self._label_encoder = LabelEncoder()
        y_ind = self._label_encoder.fit_transform(y_train.ravel())

        # calculate inverse class weights
        counts = np.bincount(y_ind).astype(np.float32)
        weights = [count / sum(counts) for count in counts]
        self._class_weights = {i: 1 / w for i, w in enumerate(weights)}

        # convert labels to categorical
        self._n_classes = len(np.unique(y_ind))
        self._y_train = to_categorical(y_ind, self._n_classes)

        # instantiate classifier
        clf = generate_lstmfcn(
            self._ts_sz,
            self._n_classes,
            lstm_dim=self.hyperparams["lstm_dim"],
            attention=self.hyperparams["attention_lstm"],
            dropout=self.hyperparams["dropout_rate"],
        )

        # mark that new training data has been set
        self._new_train_data = True

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ Fits LSTM_FCN classifier using training data from set_training_data and hyperparameters
            
            Keyword Arguments:
                timeout {float} -- timeout, considered (default: {None})
                iterations {int} -- iterations, considered (default: {None})
            
            Returns:
                CallResult[None]
        """

        # instantiate classifier and load saved weights
        clf = generate_lstmfcn(
            self._ts_sz,
            self._n_classes,
            lstm_dim=self.hyperparams["lstm_dim"],
            attention=self.hyperparams["attention_lstm"],
            dropout=self.hyperparams["dropout_rate"],
        )
        clf.compile(
            optimizer=Adam(lr=self.hyperparams["learning_rate"]),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )

        # break out validation set if iterations arg not set
        if iterations is None:
            iterations_set = False
            train_split = (1 - self.hyperparams["val_split"]) * self._X_train.shape[0]
            x_train = self._X_train[: int(train_split)].astype("float32")
            y_train = self._y_train[: int(train_split)].astype("float32")
            x_val = self._X_train[int(train_split) :]
            y_val = self._y_train[int(train_split) :]
            val_dataset = LSTMSequence(x_val, y_val, self.hyperparams["batch_size"])
            iterations = self.hyperparams["epochs"]
            callbacks = [
                EarlyStopping(
                    monitor="val_loss",
                    patience=self.hyperparams["early_stopping_patience"],
                    mode="min",
                    restore_best_weights=False,
                )
            ]
        else:
            iterations_set = True
            x_train = self._X_train
            y_train = self._y_train
            val_dataset = None
            callbacks = None
        train_dataset = LSTMSequence(x_train, y_train, self.hyperparams["batch_size"])

        # time training for 1 epoch so we can consider timeout argument thoughtfully
        if timeout:
            logger.info(
                "Timing the fitting procedure for one epoch so we \
                can consider timeout thoughtfully"
            )
            start_time = time.time()
            fitting_history = clf.fit(
                train_dataset,
                epochs=iterations,
                validation_data=val_dataset,
                class_weight=self._class_weights,
                shuffle=False,
            )
            epoch_time_estimate = time.time() - start_time
            timeout_epochs = (
                timeout // epoch_time_estimate - 1
            )  # subract 1 more to be safe
            iters = min(timeout_epochs, iterations)
            start_epoch = 1  # account for one training epoch that already happened for timing purposes
        else:
            iters = iterations
            start_epoch = 0

        # normal fitting
        logger.info(f"Fitting for {iters-start_epoch} iterations")
        start_time = time.time()
        fitting_history = clf.fit(
            train_dataset,
            epochs=iters,
            validation_data=val_dataset,
            class_weight=self._class_weights,
            shuffle=False,
            callbacks=callbacks,
            initial_epoch=start_epoch,
        )
        iterations_completed = len(fitting_history.history["loss"])
        logger.info(
            f"Fit for {iterations_completed} epochs, took {time.time() - start_time}s"
        )

        # maintain primitive state 
        self._is_fit = True
        clf.save_weights(self.hyperparams['weights_filepath'])

        # use fitting history to set CallResult return values
        if iterations_set:
            has_finished = False
        elif iters < iterations:
            has_finished = False
        else:
            has_finished = self._is_fit

        return CallResult(
            None, has_finished=has_finished, iterations_done=iterations_completed
        )

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ Produce primitive's classifications for new time series data

            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
            
            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Raises:
                PrimitiveNotFittedError: if primitive not fit

            Returns:
                CallResult[Outputs] -- dataframe with a column containing a predicted class 
                    for each input time series
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")


        # instantiate classifier and load saved weights
        clf = generate_lstmfcn(
            self._ts_sz,
            self._n_classes,
            lstm_dim=self.hyperparams["lstm_dim"],
            attention=self.hyperparams["attention_lstm"],
            dropout=self.hyperparams["dropout_rate"],
        )
        clf.load_weights(self.hyperparams['weights_filepath'])

        # find column with ts value through metadata
        grouping_column = self._get_cols(inputs.metadata)

        n_ts = inputs.iloc[:, grouping_column[0]].nunique()
        ts_sz = inputs.shape[0] // n_ts
        attribute_col = self._get_value_col(inputs.metadata)
        x_vals = inputs.iloc[:, attribute_col].values.reshape(n_ts, 1, ts_sz)
        x_vals = tf.cast(x_vals, tf.float32)
        test_dataset = LSTMSequenceTest(x_vals, self.hyperparams['batch_size'])

        # make predictions
        preds = clf.predict(test_dataset)
        preds = self._label_encoder.inverse_transform(np.argmax(preds, axis=1))

        # create output frame
        result_df = container.DataFrame(
            {self._output_columns[0]: preds}, generate_metadata=True
        )
        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            ("https://metadata.datadrivendiscovery.org/types/PredictedTarget"),
        )

        # ok to set to True because we have checked that primitive has been fit
        return CallResult(result_df, has_finished=True)

