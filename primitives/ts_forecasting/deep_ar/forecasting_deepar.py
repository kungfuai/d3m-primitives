import sys
import os
from pathlib import Path
import logging
import collections
import time
from datetime import timedelta
import typing

import numpy as np
import pandas as pd
import mxnet as mx
from sklearn.preprocessing import OrdinalEncoder

from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from gluonts.distribution import NegativeBinomialOutput, StudentTOutput

from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m import container, utils
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.exceptions import PrimitiveNotFittedError

from ..utils.var_model_utils import (
    calculate_time_frequency,
    discretize_time_difference,
)


__author__ = "Distil"
__version__ = "1.2.0"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

class Params(params.Params):
    drop_cols_no_tgt: typing.List[int]
    cols_after_drop: int
    train_data: pd.DataFrame
    ts_frame: pd.DataFrame
    target_column: int
    timestamp_column: int
    # ts_object: TimeSeriesTrain
    grouping_column: typing.Union[int, None]
    output_columns: pd.Index
    min_train: float
    freq: str
    integer_timestamps: bool
    is_fit: bool
  

class Hyperparams(hyperparams.Hyperparams):
    weights_filepath = hyperparams.Hyperparameter[str](
        default='model_weights.h5',
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="weights of trained model will be saved to this filepath",
    )
    prediction_length = hyperparams.UniformInt(
        lower=1,
        upper=60,
        default=30,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of future timesteps to predict",
    )
    context_length = hyperparams.UniformInt(
        lower=1,
        upper=60,
        default=30,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of context timesteps to consider before prediction, for both training and test",
    )
    num_layers = hyperparams.UniformInt(
        lower=1,
        upper=16,
        default=2,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of cells to use in the lstm component of the model",
    )
    lstm_dim = hyperparams.UniformInt(
        lower=10,
        upper=400,
        default=40,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of cells to use in the lstm component of the model",
    )
    epochs = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=10,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of training epochs",
    )
    steps_per_epoch = hyperparams.UniformInt(
        lower=1,
        upper=200,
        default=100,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of steps to do per epoch",
    )
    learning_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=1e-4,
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
        default=0.1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="dropout to use in lstm model (input and recurrent transform)",
    )
    count_data = hyperparams.Union[typing.Union[bool, None]](
        configuration=collections.OrderedDict(
            user_selected=hyperparams.UniformBool(default=True),
            auto_selected=hyperparams.Hyperparameter[None](default=None),
        ),
        default="auto_selected",
        description="Whether we should label the target column as real or count (positive) "
        + "based on user input or automatic selection. For example, user might want to specify "
        + "positive only count data if target column is real-valued, but domain is > 0",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
    )
    output_mean = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to output mean (or median) forecasts from probability distributions",
    )
    confidence_interval_horizon = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=2,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="horizon for confidence interval forecasts. Exposed through auxiliary "
        + "'produce_confidence_intervals' method",
    )
    confidence_interval_alpha = hyperparams.Uniform(
        lower=0.01,
        upper=1,
        default=0.1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="significance level for confidence interval, i.e. alpha = 0.05 "
        + "returns a 95%% confdience interval from alpha / 2 to 1 - (alpha / 2) "
        + "Exposed through auxiliary 'produce_confidence_intervals' method ",
    )
    confidence_interval_samples = hyperparams.UniformInt(
        lower=1,
        upper=1000,
        default=100,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="number of samples to draw at each timestep, which will be used to calculate " +
            "confidence intervals",
    )


class DeepArPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Primitive that applies a deep autoregressive forecasting algorithm for time series
        prediction. The implementation is based off of this paper: https://arxiv.org/pdf/1704.04110.pdf
        and this implementation: https://gluon-ts.mxnet.io/index.html

        Training inputs: 1) Feature dataframe, 2) Target dataframe
        Outputs: Dataframe with predictions for specific time series at specific future time instances 
    
        Arguments:
            hyperparams {Hyperparams} -- D3M Hyperparameter object
        
        Keyword Arguments:
            random_seed {int} -- random seed (default: {0})
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "3410d709-0a13-4187-a1cb-159dd24b584b",
            "version": __version__,
            "name": "DeepAR",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": [
                "time series",
                "forecasting",
                "recurrent neural network",
                "autoregressive",
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
            "python_path": "d3m.primitives.time_series_forecasting.lstm.DeepAR",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.RECURRENT_NEURAL_NETWORK,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # set random seeds for reproducibility
        mx.random.seed(random_seed)
        np.random.seed(random_seed)

        self._freq = None
        self._is_fit = False
        self._enc = OrdinalEncoder()

    def get_params(self) -> Params:
        ## TODO serialize/deserialize through this interface
        if not self._is_fit:
            return Params(
                drop_cols_no_tgt = None,
                cols_after_drop = None,
                train_data = None,
                ts_frame = None,
                target_column = None,
                timestamp_column = None,
                ts_object = None,
                grouping_column = None,
                output_columns = None,
                min_train = None,
                freq = None,
                integer_timestamps = None,
                is_fit = None
            )
        return Params(
            drop_cols_no_tgt = self._drop_cols_no_tgt,
            cols_after_drop = self._cols_after_drop,
            train_data = self._train_data,
            ts_frame = self._ts_frame,
            target_column = self._target_column,
            timestamp_column = self._timestamp_column,
            ts_object = self._ts_object,
            grouping_column = self._grouping_column,
            output_columns = self._output_columns,
            min_train = self._min_train,
            freq = self._freq,
            integer_timestamps = self._integer_timestamps,
            is_fit = self._is_fit
        )

    def set_params(self, *, params: Params) -> None:
        self._drop_cols_no_tgt = params['drop_cols_no_tgt']
        self._cols_after_drop = params['cols_after_drop']
        self._train_data = params['train_data']
        self._ts_frame = params['ts_frame']
        self._target_column = params['target_column']
        self._timestamp_column = params['timestamp_column']
        self._ts_object = params['ts_object']
        self._grouping_column = params['grouping_column']
        self._output_columns = params['output_columns']
        self._min_train = params['min_train']
        self._freq = params['freq']
        self._integer_timestamps = params['integer_timestamps']
        self._is_fit = params['is_fit']

    def _process_special_cols(self, col_list, col_type):
        """
            private util function that warns if multiple special columns 

            Arguments:
                col_list {List[int]} -- list of column indices 
                col_type {str} -- D3M semantic type

            Returns:
                int or None -- first column idx in col_list if any column idxs are marked (else None)
        """

        if len(col_list) == 0:
            return None
        elif len(col_list) > 1:
            logger.warn(
                f"""There are more than one {col_type} marked. This primitive will use the first"""
            )
        return col_list[0]

    def _sort_by_timestamp(self, frame):
        """ private util function: sort by raw timestamp and convert to pd datetime
        """
        time_name = frame.columns[self._timestamp_column]

        if "http://schema.org/Integer" in frame.metadata.query_column_field(
            self._timestamp_column, "semantic_types"
        ):
            frame[time_name] = pd.to_datetime(frame[time_name] - 1, unit = 'D')
            self._freq = 'D'
        else:
            frame[time_name] = pd.to_datetime(frame[time_name], unit = 's')

        return frame.sort_values(by = time_name)

    def _get_cols(self, frame):
        """ private util function: get indices of important columns from metadata 
        """

        input_metadata = frame.metadata

        # get target idx (first column by default)
        target_columns = input_metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/Target",
            )
        )
        if len(target_columns) == 0:
            raise ValueError("At least one column must be marked as a target")
        self._target_column = self._process_special_cols(
            target_columns, "target column"
        )

        # get timestamp idx (first column by default)
        timestamp_columns = input_metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
                "http://schema.org/DateTime",
            )
        )
        self._timestamp_column = self._process_special_cols(
            timestamp_columns, "timestamp column"
        )

        # get grouping idx 
        self._grouping_columns = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/GroupingKey",)
        )
        if len(self._grouping_columns) == 0:
            self._grouping_columns = input_metadata.list_columns_with_semantic_types(
                ("https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey",)
            )

        def diff(li1, li2): 
            return list(set(li1) - set(li2))

        # categorical columns
        self._cat_columns = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/CategoricalData",)
        )
        self._cat_columns = diff(self._cat_columns, self._grouping_columns)

        # real valued columns
        self._real_columns = input_metadata.list_columns_with_semantic_types(
            ("http://schema.org/Integer", "http://schema.org/Float")
        )

        self._real_columns = diff(
            self._real_columns, 
            [self._timestamp_column] + [self._target_column] + self._grouping_columns
        )

        # determine whether targets are count data
        target_semantic_types = input_metadata.query_column_field(
            self._target_column, "semantic_types"
        )
        if self.hyperparams["count_data"]:
            self._distr_output = NegativeBinomialOutput()
        elif self.hyperparams["count_data"] == False:
            self._distr_output = StudentTOutput()
        elif "http://schema.org/Integer" in target_semantic_types:
            if np.min(frame.iloc[:, self._target_column]) > 0:
                self._distr_output = NegativeBinomialOutput()
            else:
                self._distr_output = StudentTOutput()
        elif "http://schema.org/Float" in target_semantic_types:
            self._distr_output = StudentTOutput()
        else:
            raise ValueError("Target column is not of type 'Integer' or 'Float'")

    def _get_features(self, df, stop_idx = None):
        """ private util function: returns features for one individual time series in dataset"""
        
        features = {
            FieldName.START: df.iloc[0, self._timestamp_column],
            FieldName.TARGET: df.iloc[:stop_idx, self._target_column].values
        }
        if len(self._grouping_columns) != 0:
            features[FieldName.FEAT_STATIC_CAT] = df.iloc[0, self._grouping_columns].values
        if len(self._cat_columns) != 0:
            features[FieldName.FEAT_DYNAMIC_CAT] = df.iloc[:stop_idx, self._cat_columns].values
        if len(self._real_columns) != 0:
            features[FieldName.FEAT_DYNAMIC_REAL] = df.iloc[:stop_idx, self._real_columns].values 
        return features

    def _create_train_dataset(self, frame, stop_idx = None):
        """ private util function: creates train ds object """

        # label encode groupings /cats
        frame.iloc[:, self._grouping_columns + self._cat_columns] = self._enc.transform(
            frame.iloc[:, self._grouping_columns + self._cat_columns]
        )

        self._cardinality = [frame.iloc[:,col] for col in self._grouping_columns]

        # create dataset object
        if len(self._grouping_columns) == 0:
            return [self._get_features(frame, stop_idx)]
        else:
            data = []
            for _, df in frame.groupby(self._grouping_columns):
                data.append(self._get_features(df, stop_idx)) 
            return data

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ Sets primitive's training data
        
            Arguments:
                inputs {Inputs} -- D3M dataframe containing attributes
                outputs {Outputs} -- D3M dataframe containing targets
            
            Raises:
                ValueError: If multiple columns are annotated with 'Time' or 'DateTime' metadata
        """

        self._output_columns = outputs.columns
        self._train_data = inputs.copy()
        frame = inputs.append_columns(outputs)

        # Parse cols needed for ts object
        self._get_cols(frame)

        # calculate frequency of time series
        frame = self._sort_by_timestamp(frame)
        if len(self._grouping_columns) == 0:
            self._min_train = frame.iloc[0, self._timestamp_column]
            if self._freq is None:
                self._freq = calculate_time_frequency(
                    frame.iloc[1, self._timestamp_column] - self._min_train
                )
        else:
            # assuming frequency is the same across all grouped time series
            g_cols = [frame.columns[col] for col in self._grouping_columns]
            if sel.freq is None:
                self._freq = calculate_time_frequency(
                    int(
                        frame.groupby(g_cols)[t_col]
                        .apply(lambda x: np.diff(np.sort(x)))
                        .iloc[0][0]
                    )
                )
            self._min_train = frame.groupby(g_col)[t_col].agg("min").min()

        # Create dataset
        self._enc.fit(frame.iloc[:, self._grouping_columns + self._cat_columns])
        self._train_frame = frame
        self._train_data = self._create_train_dataset(frame)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ Fits DeepAR model using training data from set_training_data and hyperparameters
            
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

        # Create learner
        estimator = DeepAREstimator(
            freq=self._freq,
            prediction_length=self.hyperparams['prediction_length'],
            context_length=self.hyperparams['context_length'],
            use_feat_static_cat=len(self._grouping_columns) != 0,
            use_feat_dynamic_cat=len(self._cat_columns) != 0,
            use_feat_dynamic_real=len(self._real_columns) != 0,
            cardinality=self._cardinality,
            distr_output=self._distr_output,
            dropout_rate=self.hyperparams['dropout_rate'],
            trainer=Trainer(
                epochs=iterations,
                learning_rate=self.hyperparams['learning_rate'], 
                batch_size=self.hyperparams['batch_size'],
                num_batches_per_epoch=self.hyperparams['steps_per_epoch']
            )
        )

        # Fit + serialize
        logger.info(f"Fitting for {iterations} iterations")
        start_time = time.time()
        predictor = estimator.train(
            ListDataset(self._train_data, freq=self._freq)
        )
        predictor.serialize(Path(self.hyperparams['weights_filepath']))
        self.is_fit = True
        logger.info(f"Fit for {iterations} epochs, took {time.time() - start_time}s")

        return CallResult(None, has_finished=has_finished)

    def _get_pred_intervals(self, df, keep_all=False):
        """ private util function that retrieves unevenly spaced prediction intervals from data frame 

            Arguments:
                df {pandas df} -- df of predictions from which to extract prediction intervals

            Keyword Arguments:
                keep_all {bool} -- if True, take every interval slice, otherwise only take
                    those given by the df

            Returns:
                pd Series -- series of intervals, indexed by group, granularity of 1 interval 

        """

        # no grouping column
        if self._grouping_column is None:
            interval = discretize_time_difference(
                df.iloc[:, self._timestamp_column],
                self._min_train,
                self._freq,
                self._integer_timestamps,
            )
            if keep_all:
                interval = np.arange(min(interval), max(interval) + 1)
            return pd.Series([interval])

        # grouping column
        else:
            g_col, t_col = (
                df.columns[self._grouping_column],
                df.columns[self._timestamp_column],
            )
            all_intervals, groups = [], []
            for (group, vals) in df.groupby(g_col)[t_col]:
                interval = discretize_time_difference(
                    vals, self._min_train, self._freq, self._integer_timestamps
                )
                if keep_all:
                    interval = np.arange(min(interval), max(interval) + 1)
                all_intervals.append(interval)
                groups.append(group)
            return pd.Series(all_intervals, index=groups)

    def _predict(test_frame):
        """ private util function 
        """

        predictor = GluonPredictor.deserialize(Path(self.hyperparams['weights_filepath']))
        
        def _forecast(train_data):
            ## forecast (just pred_length into future)
            output_forecasts = [] 
            with tqdm(
                predictor.predict(ListDataset(train_data, freq=self.freq)),
                total=len(train_data),
                desc="Making Predictions"
            ) as it, np.errstate(invalid='ignore'):
                for forecast in it:
                    if self.hyperparams['output_mean']:
                        output_forecasts.append(forecast.mean)
                    else:
                        output_forecasts.append(forecast.quantile(0.5))
            return np.array(output_forecasts)

        # cycle through training set to get in-sample predictions
        if test_frame.equals(self._train_frame): 
            preds = np.array([])
            for _, df in test_frame.groupby(self._grouping_columns):
                data = []
                for stop_idx in range(
                    self.hyperparams['context_length'], 
                    test_frame.shape[0], 
                    self.hyperparams['prediction_length']
                ):
                    data.append(self._get_features(df, stop_idx))
                series_forecast = np.concatenate((
                    np.zeros(self.hyperparams['context_length']),  
                    np.flatten(_forecast(data))[:-self.hyperparams['context_length']]
                )) # 1, 120
                preds = np.concatenate((preds, series_forecast)) # 10, 120
        else:
            preds = _forecast(self._train_data) # 10, 30

        return preds

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ Produce primitive's predictions for specific time series at specific future time instances
            * these specific timesteps / series are specified implicitly by input dataset

            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
            
            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Raises:
                PrimitiveNotFittedError: if primitive not fit
            
            Returns:
                CallResult[Outputs] -- (N, 2) dataframe with d3m_index and value for each prediction slice requested.
                    prediction slice = specific horizon idx for specific series in specific regression 
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        # Create TimeSeriesTest object
        test_frame = inputs.copy()

        # predict
        test_frame = self._sort_by_timestamp(test_frame)
        preds = self._predict(test_frame)

        # slice predictions with learned intervals
        pred_intervals = self._get_pred_intervals(test_frame)
        # condense this guy
        all_preds = []
        for p, idxs in zip(preds, pred_intervals.values):
            all_preds.extend([p[i] for i in idxs]) 
        flat_list = np.array([p for pred_list in all_preds for p in pred_list])

        # create output frame
        result_df = container.DataFrame(
            {self._output_columns[self._target_column]: flat_list},
            generate_metadata=True,
        )
        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            ("https://metadata.datadrivendiscovery.org/types/PredictedTarget"),
        )

        return CallResult(result_df, has_finished=self._is_fit)

    def produce_confidence_intervals(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ produce confidence intervals for each series 'confidence_interval_horizon' periods into
                the future
        
        Arguments:
            inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
        
        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, considered (default: {None})
        
        Raises:
            PrimitiveNotFittedError: 
        
        Returns:
            CallResult[Outputs] -- 

            Ex. 
                series | timestep | mean | 0.05 | 0.95
                --------------------------------------
                a      |    0     |  5   |   3  |   7
                a      |    1     |  6   |   4  |   8
                b      |    0     |  5   |   3  |   7
                b      |    1     |  6   |   4  |   8
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        alpha = self.hyperparams["confidence_interval_alpha"]

        if len(self._drop_cols_no_tgt) > 0 and inputs.shape[1] != self._cols_after_drop:
            test_frame = inputs.remove_columns(self._drop_cols_no_tgt)
        else:
            test_frame = inputs.copy()

        # Create TimeSeriesTest object
        if self._train_data.equals(inputs):
            ts_test_object = TimeSeriesTest(self._ts_object)
            include_all_training = True
            horizon = 0
        # test
        else:
            ts_test_object = TimeSeriesTest(self._ts_object, test_frame)
            include_all_training = self.hyperparams['seed_predictions_with_all_data']
            horizon = self.hyperparams["confidence_interval_horizon"]

        # make predictions with learner
        start_time = time.time()
        logger.info(f"Making predictions...")
        preds = self._learner.predict(
            ts_test_object,
            horizon=horizon,
            samples=self.hyperparams["confidence_interval_samples"],
            include_all_training=include_all_training,
            point_estimate = False
        )
        logger.info(
            f"Prediction took {time.time() - start_time}s. Predictions array shape: {preds.shape}"
        )

        # convert samples to percentiles
        means = np.percentile(preds, 50, axis=2).reshape(-1, 1)
        lowers = np.percentile(preds, alpha / 2 * 100, axis=2).reshape(-1, 1)
        uppers = np.percentile(preds, (1 - alpha / 2) * 100, axis=2).reshape(-1, 1)

        assert (lowers < means).all()
        assert (means < uppers).all()

        # convert to df
        if self._grouping_column is None:
            indices = np.repeat(self._output_columns[0], preds.shape[1])
        else:
            indices = np.repeat(
                test_frame[test_frame.columns[self._grouping_column]].unique(), preds.shape[1]
            )
        interval_df = pd.DataFrame(
            np.concatenate((means, lowers, uppers), axis=1),
            columns=["mean", str(alpha / 2), str(1 - alpha / 2)],
            index=indices,
        )        

        # add index column
        interval_df["horizon_index"] = np.tile(
            np.arange(preds.shape[1]), len(interval_df.index.unique())
        )
        
        logger.debug(interval_df.head())

        # structure return df
        return CallResult(
            container.DataFrame(interval_df, generate_metadata=True),
            has_finished=self._is_fit,
        )
