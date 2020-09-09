import sys
import os
from pathlib import Path
import logging
import time
from typing import List, Union, Dict, Tuple, Any
from collections import OrderedDict

import numpy as np
import pandas as pd
import mxnet as mx
from gluonts.model.n_beats import NBEATSEnsembleEstimator
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
from .nbeats_dataset import NBEATSDataset
from .nbeats_forecast import NBEATSForecast
from .nbeats_predictor import NBEATSEnsembleEstimatorHook

__author__ = "Distil"
__version__ = "1.2.0"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)

class Params(params.Params):
    nbeats_dataset: NBEATSDataset
    is_fit: bool
    timestamp_column: int
    freq: str
    reind_freq: str
    group_cols: List[int]
    output_column: str
    target_column: int
    min_trains: Union[
        List[pd._libs.tslibs.timestamps.Timestamp], 
        Dict[str, pd._libs.tslibs.timestamps.Timestamp],
        Dict[Any, pd._libs.tslibs.timestamps.Timestamp]
    ]

class Hyperparams(hyperparams.Hyperparams):
    weights_dir= hyperparams.Hyperparameter[str](
        default='nbeats_weights',
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="weights of trained model will be saved to this filepath",
    )
    prediction_length = hyperparams.UniformInt(
        lower=1,
        upper=1000,
        default=30,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of future timesteps to predict",
    )
    interpretable = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to build interpretable architecture",
    )
    num_context_lengths = hyperparams.UniformInt(
        lower=1,
        upper=6,
        default=2,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="number of different context lengths to use for training estimators"
    )
    num_estimators = hyperparams.UniformInt(
        lower=1,
        upper=20,
        default=2,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="number of different estimators to train for each combination of context "
            + "length and loss function (3). The total number of estimators is num_estimators * "
            + "num_context_lengths * 3"
    )
    epochs = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=10,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of training epochs for each estimator",
    )
    steps_per_epoch = hyperparams.UniformInt(
        lower=1,
        upper=200,
        default=50,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of steps per epoch for each estimator",
    )
    learning_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=1e-4,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="learning rate for each estimator",
    )
    training_batch_size = hyperparams.UniformInt(
        lower=1,
        upper=256,
        default=32,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="training batch size for each estimator",
    )
    inference_batch_size = hyperparams.UniformInt(
        lower=1,
        upper=1024,
        default=256,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="inference batch size",
    )
    output_mean = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to output mean (or median) forecasts from ensemble estimators. "
        + "If `interpretable` is `True`, `output_mean` will automatically be `True` to preserve "
        + "the additive decomposition of the trend and seasonality forecast components"
    )
    # quantiles = hyperparams.Set(
    #     elements=hyperparams.Hyperparameter[float](-1),
    #     default=(),
    #     semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
    #     description="A set of quantiles for which to return estimates from forecast distribution"
    # )


class NBEATSPrimitive(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Primitive that applies the NBEATS (Neural basis expansion analysis for interpretable time
        series forecasting) method for time series forecasting. The implementation is based off of
        this paper: https://arxiv.org/abs/1905.10437 and this repository: https://gluon-ts.mxnet.io/index.html

        Training inputs: 1) Feature dataframe, 2) Target dataframe
        Outputs: Dataframe with predictions for specific time series at specific future time instances 
    
        Arguments:
            hyperparams {Hyperparams} -- D3M Hyperparameter object
        
        Keyword Arguments:
            random_seed {int} -- random seed (default: {0})
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "3952a074-145e-406d-9cee-80232ae8f3ae",
            "version": __version__,
            "name": "NBEATS",
            "keywords": [
                "time series",
                "forecasting",
                "deep neural network",
                "fully-connected",
                "residual network",
                "interpretable"
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
            "python_path": "d3m.primitives.time_series_forecasting.feed_forward_neural_net.NBEATS",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.DEEP_NEURAL_NETWORK,
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
            nbeats_dataset = self._nbeats_dataset,
            timestamp_column = self._timestamp_column,
            group_cols = self._grouping_columns,
            output_column = self._output_column,
            target_column = self._target_column,
            freq = self._freq,
            reind_freq = self._reind_freq,
            is_fit = self._is_fit,
            min_trains = self._min_trains
        )

    def set_params(self, *, params: Params) -> None:
        self._nbeats_dataset = params['nbeats_dataset']
        self._timestamp_column = params['timestamp_column']
        self._grouping_columns = params['group_cols']
        self._output_column = params['output_column']
        self._target_column = params['target_column']
        self._freq = params['freq']
        self._reind_freq = params['reind_freq']
        self._is_fit = params['is_fit']
        self._min_trains = params['min_trains']

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ Sets primitive's training data
        
            Arguments:
                inputs {Inputs} -- D3M dataframe containing attributes
                outputs {Outputs} -- D3M dataframe containing targets
            
            Raises:
                ValueError: If multiple columns are annotated with 'Time' or 'DateTime' metadata
        """

        self._output_column = outputs.columns[0]

        frame = inputs.append_columns(outputs)
        self._get_cols(frame)
        self._set_freq(frame)
        frame, self._min_trains, max_train_length, _ = self._reindex(frame)
        self._check_window_support(max_train_length)

        self._nbeats_dataset = NBEATSDataset(
            frame, 
            self._grouping_columns,
            self._timestamp_column,
            self._target_column,
            self._freq,
            self.hyperparams['prediction_length'],
            self.hyperparams['num_context_lengths']
        )
        self._train_data = self._nbeats_dataset.get_data()

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ Fits NBEATS model using training data from set_training_data and hyperparameters
            
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

        if self.hyperparams['interpretable']:
            num_stacks = 2
            num_blocks = [1]
            widths = [256,2048]
            sharing = [True]
            expansion_coefficient_lengths = [3]
            stack_types = ["T", "S"]
            estimator_class = NBEATSEnsembleEstimatorHook
        else:
            num_stacks = 30
            num_blocks = [3]
            widths = [512]
            sharing = [False]
            expansion_coefficient_lengths = [32]
            stack_types = ["G"]
            estimator_class = NBEATSEnsembleEstimator

        estimator = estimator_class(
            freq=self._freq,
            prediction_length=self.hyperparams['prediction_length'],
            meta_context_length=[
                i for i in range(2, self.hyperparams['num_context_lengths'] + 2)
            ],
            meta_loss_function = ['sMAPE', 'MASE', 'MAPE'],
            meta_bagging_size = self.hyperparams['num_estimators'],
            num_stacks=num_stacks,
            num_blocks=num_blocks,
            widths=widths,
            sharing=sharing,
            expansion_coefficient_lengths=expansion_coefficient_lengths,
            stack_types=stack_types,
            trainer=Trainer(
                epochs=iterations,
                learning_rate=self.hyperparams['learning_rate'], 
                batch_size=self.hyperparams['training_batch_size'],
                num_batches_per_epoch=self.hyperparams['steps_per_epoch']
            ),
        )

        logger.info(f"Fitting for {iterations} iterations")
        start_time = time.time()
        predictor = estimator.train(self._train_data)
        predictor.batch_size = self.hyperparams['inference_batch_size']
        predictor.set_aggregation_method('none')
        self._is_fit = True
        logger.info(f"Fit for {iterations} epochs, took {time.time() - start_time}s")

        if not os.path.isdir(self.hyperparams['weights_dir']):
            os.mkdir(self.hyperparams['weights_dir'])
        predictor.serialize(Path(self.hyperparams['weights_dir']))

        return CallResult(None, has_finished=has_finished)
        
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

        if self.hyperparams['interpretable']:
            all_components = [[] for c in range(3)]
            for series, idxs in zip(all_preds, pred_intervals):
                for i, component in enumerate(series):
                    all_components[i].append(component[idxs])
            all_components = [np.concatenate(component) for component in all_components]

            col_names = (self._output_column, 'trend-component', 'seasonality-component')
            df_data = {
                col_name: component 
                for col_name, component 
                in zip(col_names, all_components)
            }

        else:
            point_estimates = np.concatenate(
                [series[0][idxs] for series, idxs in zip(all_preds, pred_intervals)]
            )
            df_data = {self._output_column: point_estimates}
        
        result_df = container.DataFrame(
            df_data,
            generate_metadata=True,
        )

        result_df.metadata = result_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            ("https://metadata.datadrivendiscovery.org/types/PredictedTarget"),
        )
        return CallResult(result_df, has_finished=self._is_fit)

    # def produce_confidence_intervals(
    #     self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    # ) -> CallResult[Outputs]:
    #     """ produce quantiles for each prediction timestep in dataframe
        
    #     Arguments:
    #         inputs {Inputs} -- D3M dataframe containing attributes
        
    #     Keyword Arguments:
    #         timeout {float} -- timeout, not considered (default: {None})
    #         iterations {int} -- iterations, considered (default: {None})
        
    #     Raises:
    #         PrimitiveNotFittedError: 
        
    #     Returns:
    #         CallResult[Outputs] -- 

    #         Ex. 
    #             0.50 | 0.05 | 0.95
    #             -------------------
    #              5   |   3  |   7
    #              6   |   4  |   8
    #              5   |   3  |   7
    #              6   |   4  |   8
    #     """

    #     all_preds, pred_intervals = self._produce(inputs)
    #     all_quantiles = [[] for q in range(len(self.hyperparams['quantiles']) + 1)]
    #     for series, idxs in zip(all_preds, pred_intervals):
    #         for i, quantile in enumerate(series):
    #             all_quantiles[i].append(quantile[idxs])
    #     all_quantiles = [np.concatenate(quantile) for quantile in all_quantiles]

    #     col_names = (0.5,) + self.hyperparams['quantiles']
    #     result_df = container.DataFrame(
    #         {col_name: quantile for col_name, quantile in zip(col_names, all_quantiles)},
    #         generate_metadata=True,
    #     )

    #     result_df.metadata = result_df.metadata.add_semantic_type(
    #         (metadata_base.ALL_ELEMENTS, 0),
    #         ("https://metadata.datadrivendiscovery.org/types/PredictedTarget"),
    #     )

    #     return CallResult(result_df, has_finished=self._is_fit)

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
        """ private util function: convert to pd datetime and sort
        """

        time_name = frame.columns[self._timestamp_column]
        new_frame = frame.copy()

        if "http://schema.org/Integer" in frame.metadata.query_column_field(
            self._timestamp_column, "semantic_types"
        ):
            new_frame.iloc[:, self._timestamp_column] = pd.to_datetime(
                new_frame.iloc[:, self._timestamp_column] - 1, 
                unit = 'D'
            )
            self._freq = 'D'
            self._reind_freq = 'D'
        else:
            new_frame.iloc[:, self._timestamp_column] = pd.to_datetime(
                new_frame.iloc[:, self._timestamp_column], 
                unit = 's'
            )
        return new_frame.sort_values(by = time_name)

    def _set_freq(self, frame):
        """ sets frequency using differences in timestamp column in data frame
            ASSUMPTION: frequency is the same across all grouped time series
        """

        if len(self._grouping_columns) == 0:
            if self._freq is None:
                diff = frame.iloc[1, self._timestamp_column] - frame.iloc[0, self._timestamp_column]
                self._freq, self._reind_freq = calculate_time_frequency(diff, model = 'gluon')
        else:
            if self._freq is None:
                g_cols = self._get_col_names(self._grouping_columns, frame.columns)
                for g, df in frame.groupby(g_cols, sort = False):
                    diff = df.iloc[1, self._timestamp_column] - df.iloc[0, self._timestamp_column]
                    break
                self._freq, self._reind_freq = calculate_time_frequency(diff, model = 'gluon')

    def _robust_reindex(self, frame):
        """ reindex dataframe IFF it has > 1 row, interpolate target column """ 

        frame = self._sort_by_timestamp(frame)
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

        # only interpolate when target exists during training
        if self._target_column < frame.shape[1]:
            frame.iloc[:, self._target_column] = frame.iloc[:, self._target_column].interpolate() 
        frame.iloc[:, self._grouping_columns] = frame.iloc[:, self._grouping_columns].ffill()

        return frame, original_times

    def _reindex(self, frame):
        """ reindex data, interpolating target columns
        """ 

        if len(self._grouping_columns) == 0:
            df, original_times = self._robust_reindex(frame)
            return df, [df.index[0]], df.shape[0], original_times
        else:
            all_dfs, min_trains, original_times = [], {}, OrderedDict()
            max_train_length = 0
            g_cols = self._get_col_names(self._grouping_columns, frame.columns)
            for grp, df in frame.groupby(g_cols, sort = False):
                df, orig_times = self._robust_reindex(df)
                if df.shape[0] > max_train_length:
                    max_train_length = df.shape[0]
                all_dfs.append(df)
                min_trains[grp] = df.index[0]
                original_times[grp] = orig_times
            return pd.concat(all_dfs), min_trains, max_train_length, original_times

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

    def _check_window_support(self, max_train_length):
        """ ensures that at least one series of target series is >= context_length """

        if max_train_length < self.hyperparams['prediction_length']:
            raise ValueError(
                f"This training set does not support a prediction length of {self.hyperparams['prediction_length']} " +
                f"because its longest series has length {max_train_length} observations. Please " +
                f"choose a shorter prediction length."
            )

    def _get_pred_intervals(self, original_times):
        """ private util function that retrieves unevenly spaced prediction intervals from data frame 
        """

        if len(self._grouping_columns) == 0:
            intervals = discretize_time_difference(
                original_times,
                self._min_trains[0],
                self._freq, 
                zero_index = True
            )
            all_intervals = [np.array(intervals) + 1]
        else:
            all_intervals = []
            for grp, times in original_times.items():
                if grp in self._min_trains.keys():
                    intervals = discretize_time_difference(
                        times,
                        self._min_trains[grp],
                        self._freq, 
                        zero_index = True
                    )
                else:
                    logger.info(
                        f'Series with category {grp} did not exist in training data, ' +
                        f'These predictions will be returned as np.nan.'
                    )
                    intervals = np.zeros(times.shape[0]).astype(int)
                all_intervals.append(np.array(intervals) + 1)
        return all_intervals

    def _produce(self, inputs: Inputs):
        """ internal produce method to support produce() and produce_confidence_intervals() methods """
        
        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        test_frame = inputs.copy()
        nbeats_forecast = NBEATSForecast(
            self._nbeats_dataset,
            self.hyperparams['weights_dir'],
            self.hyperparams['interpretable'],
            self.hyperparams['output_mean'],
            #self.hyperparams['quantiles']
        )
        test_frame, _, _, original_times = self._reindex(test_frame)
        pred_intervals = self._get_pred_intervals(original_times)

        st = time.time()
        preds = nbeats_forecast.predict(test_frame, pred_intervals)
        logger.info(f'Making predictions took {time.time() - st}s')
        return preds, pred_intervals