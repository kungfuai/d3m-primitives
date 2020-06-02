import sys
import os
from pathlib import Path
import logging
import time
from typing import List, Union
import collections

import numpy as np
import pandas as pd
import mxnet as mx
from gluonts.model.deepar import DeepAREstimator
from gluonts.trainer import Trainer
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m import container, utils
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.exceptions import PrimitiveNotFittedError

from ..utils.time_utils import (
    calculate_time_frequency,
    discretize_time_difference,
)
from .deepar_dataset import DeepARDataset
from .deepar_forecast import DeepARForecast


__author__ = "Distil"
__version__ = "1.2.0"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)

class Params(params.Params):
    deepar_dataset: DeepARDataset
    is_fit: bool
    input_frame: pd.DataFrame
    timestamp_column: int
    freq: str
    reind_freq: str
    real_cols: List[int]
    cat_cols: List[int]
    group_cols: List[int]
    output_column: str
    max_train: Union[pd._libs.tslibs.timestamps.Timestamp, List[pd._libs.tslibs.timestamps.Timestamp]]

class Hyperparams(hyperparams.Hyperparams):
    weights_dir= hyperparams.Hyperparameter[str](
        default='deepar_weights',
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="weights of trained model will be saved to this filepath",
    )
    prediction_length = hyperparams.UniformInt(
        lower=1,
        upper=120,
        default=30,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of future timesteps to predict",
    )
    context_length = hyperparams.UniformInt(
        lower=1,
        upper=120,
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
        default=50,
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
    count_data = hyperparams.Union[Union[bool, None]](
        configuration=collections.OrderedDict(
            user_selected=hyperparams.UniformBool(default=True),
            auto_selected=hyperparams.Hyperparameter[None](default=None),
        ),
        default="auto_selected",
        description="Whether we should label the target column as real or count (positive) "
        + "based on user input or automatic selection. For example, user might want to specify "
        + "positive only count data if target column is real-valued, but domain is >= 0",
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
    quantiles = hyperparams.Set(
        elements=hyperparams.Hyperparameter[float](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of quantiles for which to return estimates from forecast distribution"
    )
    number_samples = hyperparams.UniformInt(
        lower=1,
        upper=1000,
        default=100,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="number of samples to draw at each timestep from forecast distribution"
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
            "id": "3410d709-0a13-4187-a1cb-159dd24b584b",
            "version": __version__,
            "name": "DeepAR",
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
            ],
            "python_path": "d3m.primitives.time_series_forecasting.lstm.DeepAR",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.RECURRENT_NEURAL_NETWORK,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._freq = None
        self._is_fit = False
        self.preds = None

    def get_params(self) -> Params:
        return Params(
            deepar_dataset = self._deepar_dataset,
            input_frame = self._input_frame,
            timestamp_column = self._timestamp_column,
            real_cols = self._real_columns,
            group_cols = self._grouping_columns,
            cat_cols = self._cat_columns,
            output_column = self._output_column,
            freq = self._freq,
            reind_freq = self._reind_freq,
            is_fit = self._is_fit,
            max_train = self._max_trains
        )

    def set_params(self, *, params: Params) -> None:
        self._deepar_dataset = params['deepar_dataset']
        self._input_frame = params['input_frame']
        self._timestamp_column = params['timestamp_column']
        self._real_columns = params['real_cols']
        self._grouping_columns = params['group_cols']
        self._cat_columns = params['cat_cols']
        self._output_column = params['output_column']
        self._freq = params['freq']
        self._reind_freq = params['reind_freq']
        self._is_fit = params['is_fit']
        self._max_trains = params['max_train']

    def _get_col_names(self, col_idxs, all_col_names):
        """ transform column indices to column names """
        return [all_col_names[i] for i in col_idxs]
        
    def _process_special_col(self, col_list, col_type):
        """ private util function that warns if multiple special columns 
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
            self._reind_freq = 'D'
        else:
            frame[time_name] = pd.to_datetime(frame[time_name], unit = 's')

        return frame.sort_values(by = time_name)

    def _set_freq(self, frame, norm_value = 10e8):
        """ sets frequency using differences in timestamp column in data frame
            ASSUMPTION: frequency is the same across all grouped time series
        """

        if len(self._grouping_columns) == 0:
            if self._freq is None:
                diff = frame.iloc[1, self._timestamp_column] - frame.iloc[0, self._timestamp_column]
                #diff /= norm_value
                self._freq, self._reind_freq = calculate_time_frequency(diff, model = 'deep_ar')
        else:
            if self._freq is None:
                g_cols = self._get_col_names(self._grouping_columns, frame.columns)
                for g, df in frame.groupby(g_cols):
                    diff = df.iloc[1, self._timestamp_column] - df.iloc[0, self._timestamp_column]
                    break
                self._freq, self._reind_freq = calculate_time_frequency(diff, model = 'deep_ar')

    def _robust_reindex(self, frame):
        """ reindex dataframe IFF it has > 1 row, interpolate real-valued columns, forward-filling
            categorical and grouping columns """ 
        
        original_times = frame.iloc[:, self._timestamp_column]
        frame = frame.drop_duplicates(subset = frame.columns[self._timestamp_column])
        frame.index = frame.iloc[:, self._timestamp_column]
        if frame.shape[0] > 1:
            frame = frame.reindex(
                pd.date_range(
                    frame.index[0], 
                    frame.index[-1], 
                    freq = self._reind_freq,
                )
            )
        frame.iloc[:, self._real_columns] = frame.iloc[:, self._real_columns].interpolate()
        frame.iloc[:, self._cat_columns + self._grouping_columns] = \
            frame.iloc[:, self._cat_columns + self._grouping_columns].ffill()

        return frame, original_times

    def _reindex(self, frame):
        """ reindex data, keeping NA values for target column, but interpolating feature columns
        """ 

        if len(self._grouping_columns) == 0:
            df, original_times = self._robust_reindex(frame)
            return df, df.index[-1], df.shape[0], original_times
        else:
            all_dfs, max_trains, original_times = [], [], []
            max_train_length = 0
            g_cols = self._get_col_names(self._grouping_columns, frame.columns)
            for _, df in frame.groupby(g_cols):
                df, orig_times = self._robust_reindex(df)
                if df.shape[0] > max_train_length:
                    max_train_length = df.shape[0]
                max_trains.append(df.index[-1])
                all_dfs.append(df)
                original_times.append(orig_times)
            return pd.concat(all_dfs), max_trains, max_train_length, original_times

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

        self._target_column = self._process_special_col(
            target_columns, "target column"
        )

        # get timestamp idx (first column by default)
        timestamp_columns = input_metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
                "http://schema.org/DateTime",
            )
        )
        self._timestamp_column = self._process_special_col(
            timestamp_columns, "timestamp column"
        )

        # get grouping idx 
        self._grouping_columns = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/GroupingKey",)
        )
        suggested_group_cols = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey",)
        )
        if len(self._grouping_columns) == 0:
            self._grouping_columns = suggested_group_cols

        def diff(li1, li2): 
            return list(set(li1) - set(li2))

        # categorical columns
        self._cat_columns = input_metadata.list_columns_with_semantic_types(
            ("https://metadata.datadrivendiscovery.org/types/CategoricalData",)
        )
        self._cat_columns = diff(self._cat_columns, self._grouping_columns + suggested_group_cols)

        # real valued columns
        self._real_columns = input_metadata.list_columns_with_semantic_types(
            ("http://schema.org/Integer", "http://schema.org/Float")
        )

        self._real_columns = diff(
            self._real_columns, 
            [self._timestamp_column] + [self._target_column] + self._grouping_columns
        )

        # determine whether targets are count data
        self._target_semantic_types = input_metadata.query_column_field(
            self._target_column, "semantic_types"
        )

    def _check_window_support(self, max_train_length):
        """ ensures that at least one series of target series is >= context_length """

        if max_train_length < self.hyperparams['prediction_length']:
            raise ValueError(
                f"This training set does not support a prediction length of {self.hyperparams['prediction_length']} " +
                f"because its longest series has length {max_train_length} observations. Please " +
                f"choose a shorter prediction length."
            )

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ Sets primitive's training data
        
            Arguments:
                inputs {Inputs} -- D3M dataframe containing attributes
                outputs {Outputs} -- D3M dataframe containing targets
            
            Raises:
                ValueError: If multiple columns are annotated with 'Time' or 'DateTime' metadata
        """

        self._output_column = outputs.columns[0]
        self._input_frame = inputs.copy()

        frame = inputs.append_columns(outputs)
        self._get_cols(frame)
        frame = self._sort_by_timestamp(frame)
        self._set_freq(frame)
        frame, self._max_trains, max_train_length, _ = self._reindex(frame)
        self._check_window_support(max_train_length)

        self._deepar_dataset = DeepARDataset(
            frame, 
            self._grouping_columns,
            self._cat_columns,
            self._real_columns,
            self._timestamp_column,
            self._target_column,
            self._freq,
            self.hyperparams['prediction_length'],
            self.hyperparams['context_length'],
            self._target_semantic_types,
            self.hyperparams['count_data']
        )
        self._train_data = self._deepar_dataset.get_data()

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

        estimator = DeepAREstimator(
            freq=self._freq,
            prediction_length=self.hyperparams['prediction_length'],
            context_length=self.hyperparams['context_length'],
            use_feat_static_cat=self._deepar_dataset.has_cat_cols() or self._deepar_dataset.has_group_cols(),
            use_feat_dynamic_real=self._deepar_dataset.has_real_cols(),
            cardinality=self._deepar_dataset.get_cardinality(),
            distr_output=self._deepar_dataset.get_distribution_type(),
            dropout_rate=self.hyperparams['dropout_rate'],
            trainer=Trainer(
                epochs=iterations,
                learning_rate=self.hyperparams['learning_rate'], 
                batch_size=self.hyperparams['batch_size'],
                num_batches_per_epoch=self.hyperparams['steps_per_epoch']
            )
        )

        logger.info(f"Fitting for {iterations} iterations")
        start_time = time.time()
        predictor = estimator.train(self._train_data)
        self._is_fit = True
        logger.info(f"Fit for {iterations} epochs, took {time.time() - start_time}s")

        if not os.path.isdir(self.hyperparams['weights_dir']):
            os.mkdir(self.hyperparams['weights_dir'])
        predictor.serialize(Path(self.hyperparams['weights_dir']))

        return CallResult(None, has_finished=has_finished)

    def _get_pred_intervals(self, original_times):
        """ private util function that retrieves unevenly spaced prediction intervals from data frame 
        """

        if len(self._grouping_columns) == 0:
            intervals = discretize_time_difference(
                original_times,
                self._max_trains,
                self._freq, 
                zero_index = True
            )
            all_intervals = [np.array(intervals)]
        else:
            all_intervals = []
            g_cols = self._deepar_dataset.get_group_names()
            for times, max_train in zip(original_times, self._max_trains):
                intervals = discretize_time_difference(
                    times,
                    max_train,
                    self._freq, 
                    zero_index = True
                )
                all_intervals.append(np.array(intervals))
        return all_intervals

    def _check_pred_support(self, pred_intervals):
        """ check if trained model supports called for prediction length """

        max_pred_length = max([np.max(interval) for interval in pred_intervals])
        if max_pred_length >= self.hyperparams['prediction_length']:
            raise ValueError(
                f"Asking for a prediction {max_pred_length + 1} steps into the future from a model " + 
                f"that was trained to predict a maximum of {self.hyperparams['prediction_length']} " + 
                f"steps into the future. Please train a model with a longer prediction horizon or " +
                f"only ask for predictions <= {self.hyperparams['prediction_length']} into the future."
            )

    def _produce(self, inputs: Inputs):
        """ internal produce method to support produce() and produce_confidence_intervals() methods """
        
        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        test_frame = inputs.copy()
        deepar_forecast = DeepARForecast(
            self._deepar_dataset,
            self.hyperparams['weights_dir'],
            self.hyperparams['output_mean'],
            self.hyperparams['number_samples'],
            self.hyperparams['quantiles']
        )
    
        test_frame = self._sort_by_timestamp(test_frame)
        test_frame, _, _, original_times = self._reindex(test_frame)
        pred_intervals = self._get_pred_intervals(original_times)

        st = time.time()
        if inputs.equals(self._input_frame):
            logger.info('Making in-sample predictions for all training observations. ' +
                'This can take a while for training sets with a large number of series.')
            preds = deepar_forecast.predict_in_sample()
            logger.info(f'Making in-sample predictions took {time.time() - st}s')
        else:
            self._check_pred_support(pred_intervals)
            logger.info('Making out-of-sample predictions...')
            preds = deepar_forecast.predict_out_of_sample(test_frame)
            logger.info(f'Making out-of-sample predictions took {time.time() - st}s')

        return preds, pred_intervals
        
    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ Produce primitive's predictions for specific time series at specific future time instances
            * these specific timesteps / series are specified implicitly by input dataset

            Arguments:
                inputs {Inputs} -- D3M dataframe containing attributes
            
            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Raises:
                PrimitiveNotFittedError: if primitive not fit
            
            Returns:
                CallResult[Outputs] -- (N, 2) dataframe with d3m_index and value for each prediction slice requested.
                    prediction slice = specific horizon idx for specific series in specific regression 
        """

        all_preds, pred_intervals = self._produce(inputs)
        point_estimates = np.concatenate([series[0][idxs] for series, idxs in zip(all_preds, pred_intervals)])
        
        result_df = container.DataFrame(
            {self._output_column: point_estimates},
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
        """ produce quantiles for each prediction timestep in dataframe
        
        Arguments:
            inputs {Inputs} -- D3M dataframe containing attributes
        
        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, considered (default: {None})
        
        Raises:
            PrimitiveNotFittedError: 
        
        Returns:
            CallResult[Outputs] -- 

            Ex. 
                0.50 | 0.05 | 0.95
                -------------------
                 5   |   3  |   7
                 6   |   4  |   8
                 5   |   3  |   7
                 6   |   4  |   8
        """

        all_preds, pred_intervals = self._produce(inputs)
        all_quantiles = [[] for q in range(len(self.hyperparams['quantiles']) + 1)]
        for series, idxs in zip(all_preds, pred_intervals):
            for i, quantile in enumerate(series):
                all_quantiles[i].append(quantile[idxs])
        all_quantiles = [np.concatenate(quantile) for quantile in all_quantiles]
        col_names = [0.5] + self.hyperparams['quantiles']
        result_df = container.DataFrame(
            {col_name: quantile for col_name, quantile in zip(col_names, all_quantiles)},
            generate_metadata=True,
        )

        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            ("https://metadata.datadrivendiscovery.org/types/PredictedTarget"),
        )

        return CallResult(result_df, has_finished=self._is_fit)
