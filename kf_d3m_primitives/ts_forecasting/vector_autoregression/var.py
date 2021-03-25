import sys
import os
import collections
from datetime import timedelta
from typing import List, Union, Any, Tuple
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.exceptions import PrimitiveNotFittedError
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from statsmodels.tsa.api import VAR as vector_ar
from statsmodels.tsa.vector_ar.var_model import VARResultsWrapper
import statsmodels.api as sm
import scipy.stats as stats

from ..utils.time_utils import calculate_time_frequency, discretize_time_difference
from .arima import Arima

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)

__author__ = "Distil"
__version__ = "1.2.0"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

MAX_INT = np.finfo("d").max - 1


class Params(params.Params):
    integer_time: bool
    time_column: str
    targets: List[str]
    target_indices: List[int]
    X_train: Union[List[d3m_DataFrame], List[pd.DataFrame]]
    X_train_names: Any

    filter_idxs: List[str]
    interpolation_ranges: Union[pd.Series, None, pd.DataFrame]
    freq: str
    is_fit: bool

    fits: Union[
        List[VARResultsWrapper], List[Arima], List[Union[VARResultsWrapper, Arima]]
    ]
    values: List[np.ndarray]
    values_diff: List[np.ndarray]
    lag_order: Union[
        List[None],
        List[np.int64],
        List[int],
        List[Union[np.int64, None]],
        List[Union[int, None]],
        List[Union[np.int64, int, None]],
    ]


class Hyperparams(hyperparams.Hyperparams):
    max_lag_order = hyperparams.Union[Union[int, None]](
        configuration=collections.OrderedDict(
            user_selected=hyperparams.UniformInt(lower=0, upper=100, default=1),
            auto_selected=hyperparams.Hyperparameter[None](
                default=None,
                description="Lag order of regressions automatically selected",
            ),
        ),
        default="user_selected",
        description="The lag order to apply to regressions. If user-selected, the same lag will be "
        + "applied to all regressions. If auto-selected, different lags can be selected for different "
        + "regressions.",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
    )
    seasonal = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to perform ARIMA prediction with seasonal component",
    )
    seasonal_differencing = hyperparams.UniformInt(
        lower=1,
        upper=365,
        default=1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="period of seasonal differencing to use in ARIMA prediction",
    )
    dynamic = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to perform dynamic in-sample prediction with ARIMA model",
    )
    interpret_value = hyperparams.Enumeration(
        default="lag_order",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["series", "lag_order"],
        description="whether to return weight coefficients for each series or each lag order "
        + "separately in the regression",
    )
    interpret_pooling = hyperparams.Enumeration(
        default="avg",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["avg", "max"],
        description="whether to pool weight coefficients via average or max",
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
        + "returns a 95%% confdience interval from alpha / 2 to 1 - (alpha / 2) . "
        + "Exposed through auxiliary 'produce_confidence_intervals' method",
    )


class VarPrimitive(
    SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]
):
    """
    This primitive applies a vector autoregression (VAR) multivariate forecasting model to time series data.
    It defaults to an ARIMA model if the time series is univariate. The VAR
    implementation comes from the statsmodels library. The lag order and AR, MA, and
    differencing terms for the VAR and ARIMA models respectively are selected automatically
    and independently for each regression. User can override automatic selection with 'max_lag_order' HP.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "76b5a479-c209-4d94-92b5-7eba7a4d4499",
            "version": __version__,
            "name": "VAR",
            "keywords": ["Time Series"],
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
            "python_path": "d3m.primitives.time_series_forecasting.vector_autoregression.VAR",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.VECTOR_AUTOREGRESSION
            ],
            "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # track metadata about times, targets, indices, grouping keys
        self._filter_idxs = None
        self._targets = None
        self._key = None
        self._integer_time = False
        self._target_indices = None

        # information about interpolation
        self._freq = None
        self._interpolation_ranges = None

        # data needed to fit model and reconstruct predictions
        self._X_train_names = []
        self._X_train = None
        self._mins = None
        self._lag_order = []
        self._values = None
        self._fits = []
        self._is_fit = False

    def get_params(self) -> Params:
        if not self._is_fit:
            return Params(
                integer_time=None,
                time_column=None,
                targets=None,
                target_indices=None,
                X_train=None,
                fits=None,
                values=None,
                values_diff=None,
                lag_order=None,
                positive=None,
                filter_idxs=None,
                interpolation_ranges=None,
                freq=None,
                is_fit=None,
                X_train_names=None,
            )

        return Params(
            integer_time=self._integer_time,
            time_column=self._time_column,
            targets=self._targets,
            target_indices=self._target_indices,
            X_train=self._X_train,
            fits=self._fits,
            values=self._values,
            values_diff=self._values_diff,
            lag_order=self._lag_order,
            filter_idxs=self._filter_idxs,
            interpolation_ranges=self._interpolation_ranges,
            freq=self._freq,
            is_fit=self._is_fit,
            X_train_names=self._X_train_names,
        )

    def set_params(self, *, params: Params) -> None:
        self._integer_time = params["integer_time"]
        self._time_column = params["time_column"]
        self._targets = params["targets"]
        self._target_indices = params["target_indices"]
        self._X_train = params["X_train"]
        self._fits = params["fits"]
        self._values = params["values"]
        self._values_diff = params["values_diff"]
        self._lag_order = params["lag_order"]
        self._filter_idxs = params["filter_idxs"]
        self._interpolation_ranges = params["interpolation_ranges"]
        self._freq = params["freq"]
        self._is_fit = params["is_fit"]
        self._X_train_names = params["X_train_names"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """Sets primitive's training data

        Arguments:
            inputs {Inputs} -- D3M dataframe containing attributes
            outputs {Outputs} -- D3M dataframe containing targets

        Raises:
            ValueError: If multiple columns are annotated with 'Time' or 'DateTime' metadata
        """

        inputs_copy = inputs.append_columns(outputs)

        times = self._get_cols(inputs_copy)
        inputs_copy = self._convert_times(inputs_copy, times)
        num_group_keys, drop_list = self._get_grouping_keys(inputs_copy)

        inputs_copy = inputs_copy.drop(
            columns=[list(inputs_copy)[i] for i in drop_list + self._key]
        )  # drop index and extraneous grouping keys

        self._prepare_collections(inputs_copy, num_group_keys)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """If there are multiple endogenous series, primitive will fit VAR model. Otherwise it will fit an ARIMA
        model. In the VAR case, the lag order will be automatically choosen based on BIC (unless user overrides).
        In the ARIMA case, the lag order will be automatically chosen by differencing tests (again, unless user
        overrides).

        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, not considered (default: {None})

        Returns:
            CallResult[None]
        """

        # mark if data is exclusively positive
        self._values = [sequence.values for sequence in self._X_train]
        # self._positive = [True if np.min(vals) < 0 else False for vals in self._values]

        # difference data - VAR assumes data is stationary
        self._values_diff = [np.diff(sequence, axis=0) for sequence in self._X_train]

        # define models
        if self.hyperparams["max_lag_order"] is None:
            arima_max_order = 5
        else:
            arima_max_order = self.hyperparams["max_lag_order"]

        self.models = [
            vector_ar(vals, dates=original.index, freq=self._freq)
            if vals.shape[1] > 1
            else Arima(
                seasonal=self.hyperparams["seasonal"],
                seasonal_differencing=self.hyperparams["seasonal_differencing"],
                max_order=arima_max_order,
                dynamic=self.hyperparams["dynamic"],
            )
            for vals, original in zip(self._values_diff, self._X_train)
        ]

        self._robust_fit(self.models, self._values_diff, self._X_train)

        return CallResult(None, has_finished=self._is_fit)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """prediction for future time series data

        Arguments:
            inputs {Inputs} -- attribute dataframe

        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, not considered (default: {None})

        Raises:
            PrimitiveNotFittedError: if primitive not fit

        Returns:
            CallResult[Outputs] -- predictions for each prediction interval requested
        """

        return self._produce(inputs)

    def produce_confidence_intervals(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """produce confidence intervals for each series

        Arguments:
            inputs {Inputs} -- attribute dataframe

        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, not considered (default: {None})

        Raises:
            PrimitiveNotFittedError: if primitive not fit

        Returns:
            CallResult[Outputs] -- predictions for each prediction interval requested

            Ex.
                 tgt  | tgt-0.05 | tgt-0.95
                ----------------------------
                  5   |     3    |    7
                  6   |     4    |    8
                  5   |     3    |    7
                  6   |     4    |    8
        """

        return self._produce(inputs, return_conf_int=True)

    def produce_weights(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """Produce absolute values of correlation coefficients (weights) for each of the terms used in each 
            regression model. Terms must be aggregated by series or by lag order (thus the need for absolute value). 
            Pooling operation can be maximum or average (controlled by 'interpret_pooling' HP).

        Arguments:
            inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target

        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, considered (default: {None})

        Raises:
            PrimitiveNotFittedError: if primitive not fit

        Returns:
            CallResult[Outputs] -- pandas df where each row represents a unique series from one of the 
                regressions that was fit. The columns contain the coefficients for each term in the regression, 
                potentially aggregated by series or lag order. Column names will represent the lag order or 
                series to which that column refers. If the regression is an ARIMA model, the set of column 
                names will also contain AR_i (autoregressive terms) and MA_i (moving average terms).
                Columns that are not included in the regression for a specific series will have NaN 
                values in those respective columns.
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        if self.hyperparams["interpret_value"] == "series":
            logger.info(
                "You should interpret a row of the returned matrix like this: "
                + "Each row represents an endogeneous variable for which the VAR process learned an equation. "
                + "Each column represents all of the endogenous variables used in the regression equation. "
                + "Each matrix entry represents the weight of the column endogeneous variable in the equation for the "
                + "row endogenous variable."
            )

        # get correlation coefficients
        coefficients = [
            np.absolute(fit.coefs)
            if lags is not None
            else fit.get_absolute_value_params()
            for fit, lags in zip(self._fits, self._lag_order)
        ]
        trends = [
            np.absolute(fit.params[0, :].reshape(-1, 1)) if lags is not None else None
            for fit, lags in zip(self._fits, self._lag_order)
        ]

        # combine coeffcient vectors into single df
        coef_df = None
        for coef, trend, names in zip(coefficients, trends, self._X_train_names):
            # aggregate VAR coefficients based on HPs
            if trend is not None:
                if self.hyperparams["interpret_value"] == "series":
                    if self.hyperparams["interpret_pooling"] == "avg":
                        coef = np.mean(coef, axis=0)  # K x K
                    else:
                        coef = np.max(coef, axis=0)  # K x K
                    colnames = names
                else:
                    # or axis = 2, I believe symmetrical
                    if self.hyperparams["interpret_pooling"] == "avg":
                        coef = np.mean(coef, axis=1).T  # K x p + 1
                    else:
                        coef = np.max(coef, axis=1).T  # K x p + 1
                    coef = np.concatenate((trend, coef), axis=1)
                    colnames = ["trend_0"] + [
                        "ar_" + str(i + 1) for i in range(coef.shape[1] - 1)
                    ]
                new_df = pd.DataFrame(coef, columns=colnames, index=names)
                coef_df = pd.concat([coef_df, new_df], sort=True)

            # add index to ARIMA params
            else:
                coef.index = names
                if self.hyperparams["interpret_value"] == "lag_order":
                    coef_df = pd.concat([coef_df, coef], sort=True)

        if coef_df is None:
            logger.info(
                f"There was only one variable in each grouping of time series, "
                + "therefore only ARIMA models were fit. Additionally, becasue the 'interpret_value' "
                + "hyperparameter is set to series, this will return an empty dataframe."
            )

        return CallResult(
            container.DataFrame(coef_df, generate_metadata=True),
            has_finished=self._is_fit,
        )

    def _get_cols(self, frame):
        """private util function: get indices of important columns from metadata"""

        # mark datetime column
        times = frame.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
                "http://schema.org/DateTime",
            )
        )
        if len(times) != 1:
            raise ValueError(
                f"There are {len(times)} indices marked as datetime values. Please only specify one"
            )
        self._time_column = list(frame)[times[0]]

        # mark key variable
        self._key = frame.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
        )

        # mark target variables
        self._targets = frame.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/Target",
            )
        )
        self._targets = [list(frame)[t] for t in self._targets]

        return times

    def _convert_times(self, frame, times):
        """private util function: convert to pd datetime

        if datetime columns are integers, parse as # of days
        """

        if (
            "http://schema.org/Integer"
            in frame.metadata.query_column(times[0])["semantic_types"]
        ):
            self._integer_time = True
            frame[self._time_column] = pd.to_datetime(
                frame[self._time_column] - 1, unit="D"
            )
        else:
            frame[self._time_column] = pd.to_datetime(
                frame[self._time_column], unit="s"
            )

        return frame

    def _get_grouping_keys(self, frame):
        """see if 'GroupingKey' has been marked
        otherwise fall through to use 'SuggestedGroupingKey' to intelligently calculate grouping key order
        we sort keys so that VAR can operate on as many series as possible simultaneously (reverse order)

        return the number of grouping columns and list of extraneous columns that should be dropped
        """

        grouping_keys = frame.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = frame.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
            drop_list = []
        else:
            drop_list = suggested_grouping_keys

        grouping_keys_counts = [
            frame.iloc[:, key_idx].nunique() for key_idx in grouping_keys
        ]
        grouping_keys = [
            group_key
            for count, group_key in sorted(zip(grouping_keys_counts, grouping_keys))
        ]
        self._filter_idxs = [list(frame)[key] for key in grouping_keys]

        return len(grouping_keys), drop_list

    def _prepare_collections(self, frame, num_group_keys=0):
        """prepare separate collections of series on which to fit separate VAR or ARIMA models"""

        # check whether no grouping keys are labeled
        if num_group_keys == 0:

            # avg across duplicated time indices if necessary and re-index
            if sum(frame[self._time_column].duplicated()) > 0:
                frame = frame.groupby(self._time_column).mean()
            else:
                frame = frame.set_index(self._time_column)

            # interpolate
            self._freq = calculate_time_frequency(frame.index[1] - frame.index[0])
            frame = frame.interpolate(method="time", limit_direction="both")

            # set X train
            self._target_indices = [
                i for i, col_name in enumerate(list(frame)) if col_name in self._targets
            ]
            self._X_train = [frame]
            self._X_train_names = [frame.columns]

        else:
            # find interpolation range from outermost grouping key
            if num_group_keys == 1:
                date_ranges = frame.agg({self._time_column: ["min", "max"]})
                indices = frame[self._filter_idxs[0]].unique()
                self._interpolation_ranges = pd.Series(
                    [date_ranges] * len(indices), index=indices
                )
                self._X_train = [None]
                self._X_train_names = [None]
            else:
                self._interpolation_ranges = frame.groupby(
                    self._filter_idxs[:-1], sort=False
                ).agg({self._time_column: ["min", "max"]})
                self._X_train = [
                    None for i in range(self._interpolation_ranges.shape[0])
                ]
                self._X_train_names = [
                    None for i in range(self._interpolation_ranges.shape[0])
                ]

            for name, group in frame.groupby(self._filter_idxs, sort=False):
                if num_group_keys > 2:
                    group_value = name[:-1]
                elif num_group_keys == 2:
                    group_value = name[0]
                else:
                    group_value = name
                if num_group_keys > 1:
                    training_idx = np.where(
                        self._interpolation_ranges.index.to_flat_index() == group_value
                    )[0][0]
                else:
                    training_idx = 0
                group = group.drop(columns=self._filter_idxs)

                # avg across duplicated time indices if necessary and re-index
                group = group.sort_values(by=[self._time_column])
                if sum(group[self._time_column].duplicated()) > 0:
                    group = group.groupby(self._time_column).mean()
                else:
                    group = group.set_index(self._time_column)

                # interpolate
                min_date = self._interpolation_ranges.loc[group_value][
                    self._time_column
                ]["min"]
                max_date = self._interpolation_ranges.loc[group_value][
                    self._time_column
                ]["max"]
                # assume frequency is the same across all time series
                if self._freq is None:
                    self._freq = calculate_time_frequency(
                        group.index[1] - group.index[0]
                    )

                group = group.reindex(
                    pd.date_range(min_date, max_date, freq=self._freq),
                )
                group = group.interpolate(method="time", limit_direction="both")

                # add to training data under appropriate top-level grouping key
                self._target_indices = [
                    i
                    for i, col_name in enumerate(list(group))
                    if col_name in self._targets
                ]
                if self._X_train[training_idx] is None:
                    self._X_train[training_idx] = group
                else:
                    self._X_train[training_idx] = pd.concat(
                        [self._X_train[training_idx], group], axis=1
                    )
                if self._X_train_names[training_idx] is None:
                    self._X_train_names[training_idx] = [name]
                else:
                    self._X_train_names[training_idx].append(name)

    def _robust_fit(self, models, training_data, training_times):
        """fit models, robustly recover from matrix decomposition errors and other fitting
        errors
        """

        for vals, model, original in zip(training_data, models, training_times):
            # VAR
            if vals.shape[1] > 1:
                try:
                    lags = model.select_order(
                        maxlags=self.hyperparams["max_lag_order"]
                    ).bic
                    logger.info(
                        "Successfully performed model order selection. Optimal order = {} lags".format(
                            lags
                        )
                    )
                except np.linalg.LinAlgError as e:
                    lags = 0
                    logger.info(f"Matrix decomposition error.  Using lag order of 0")
                except ValueError as e:
                    lags = 0
                    logger.info("ValueError: " + str(e) + ". Using lag order of 0")
                self._lag_order.append(lags)
                self._fits.append(model.fit(maxlags=lags))

            # ARIMA
            else:
                X_train = pd.Series(
                    data=vals.reshape((-1,)), index=original.index[: vals.shape[0]]
                )
                model.fit(X_train)
                self._lag_order.append(None)
                self._fits.append(model)

        self._is_fit = True

    def _calculate_prediction_intervals(self, inputs: Inputs, num_group_keys: int):
        """private util function that uses learned grouping keys to extract horizon,
        horizon intervals, and forecast_idxs
        """

        # check whether no grouping keys are labeled
        if num_group_keys == 0:
            group_tuple = ((self._X_train_names[0][0], inputs),)
        else:
            group_tuple = inputs.groupby(self._filter_idxs, sort=False)

        # groupby learned filter_idxs and extract n_periods, interval and d3mIndex information
        n_periods = [0 for i in range(len(self._X_train))]
        forecast_idxs = []
        intervals = []
        for name, group in group_tuple:
            if num_group_keys > 2:
                group_value = name[:-1]
            elif num_group_keys == 2:
                group_value = name[0]
            else:
                group_value = name

            if num_group_keys > 1:
                testing_idx = np.where(
                    self._interpolation_ranges.index.to_flat_index() == group_value
                )[0][0]
            else:
                testing_idx = 0

            col_idxs = [
                i
                for i, tupl in enumerate(self._X_train_names[testing_idx])
                if tupl == name
            ]

            if not len(col_idxs):
                logger.info(
                    f"Series with category {name} did not exist in training data, "
                    + f"These predictions will be returned as np.nan."
                )
                col_idx = -1
            else:
                col_idx = col_idxs[0]
            forecast_idxs.append((testing_idx, col_idx))

            min_train_idx = self._X_train[testing_idx].index[0]
            local_intervals = discretize_time_difference(
                group[self._time_column], min_train_idx, self._freq
            )
            intervals.append(local_intervals)

            num_p = int(max(local_intervals) - self._X_train[testing_idx].shape[0] + 1)
            if n_periods[testing_idx] < num_p:
                n_periods[testing_idx] = num_p

        return n_periods, forecast_idxs, intervals

    def _forecast(self, n_periods, return_conf_int=False):
        """make future forecasts using models, prepend in-sample predictions, inverse transformations
        using information extracted from prediction intervals
        """

        forecasts = []
        for fit, lags, vals, vals_diff, horizon in zip(
            self._fits, self._lag_order, self._values, self._values_diff, n_periods
        ):
            if lags is None:
                preds = np.concatenate(
                    (vals[:1], vals_diff[:1], fit.predict_in_sample().reshape(-1, 1)),
                    axis=0,
                )
            else:
                preds = np.concatenate(
                    (vals[:1], vals_diff[:lags], fit.fittedvalues), axis=0
                )
            in_sample_len = preds.shape[0]

            if horizon > 0:

                if return_conf_int:
                    alpha = self.hyperparams["confidence_interval_alpha"]
                    means, lowers, uppers = [], [], []
                    if lags is not None and lags > 0:
                        mean, lower, upper = fit.forecast_interval(
                            y=vals_diff[-fit.k_ar :], steps=horizon, alpha=alpha
                        )
                    elif lags == 0:
                        q = stats.norm.ppf(1 - alpha / 2)
                        sigma = np.sqrt(fit._forecast_vars(horizon))
                        mean = np.repeat(fit.params, horizon, axis=0)
                        lower = np.repeat(fit.params - q * sigma, horizon, axis=0)
                        upper = np.repeat(fit.params + q * sigma, horizon, axis=0)
                    else:
                        mean, lower, upper = fit.predict(
                            n_periods=horizon, return_conf_int=True, alpha=alpha
                        )
                        if len(mean.shape) == 1:
                            mean = mean.reshape(-1, 1)
                            lower = lower.reshape(-1, 1)
                            upper = upper.reshape(-1, 1)

                    preds = [
                        np.concatenate((preds, mean), axis=0),
                        np.concatenate((preds, lower), axis=0),
                        np.concatenate((preds, upper), axis=0),
                    ]
                    preds = [p.cumsum(axis=0) for p in preds]
                    preds[1][:in_sample_len] = np.nan
                    preds[2][:in_sample_len] = np.nan

                else:
                    if lags is not None and lags > 0:
                        mean = fit.forecast(y=vals_diff[-fit.k_ar :], steps=horizon)
                    elif lags == 0:
                        mean = np.repeat(fit.params, horizon, axis=0)
                    else:
                        mean = fit.predict(n_periods=horizon).reshape(-1, 1)

                    preds = np.concatenate((preds, mean), axis=0)
                    preds = [preds.cumsum(axis=0)]

            else:
                preds = [preds.cumsum(axis=0)]

                if return_conf_int:
                    nan_array = np.empty(preds[0].shape)
                    nan_array[:] = np.nan
                    preds.append(nan_array)
                    preds.append(nan_array)

            preds = [pd.DataFrame(p) for p in preds]
            forecasts.append(preds)

        return forecasts

    def _produce(
        self, inputs: Inputs, return_conf_int: bool = False
    ) -> CallResult[Outputs]:
        """prediction for future time series data"""
        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        # make copy of input data!
        inputs_copy = inputs.copy()

        # if datetime columns are integers, parse as # of days
        if self._integer_time:
            inputs_copy[self._time_column] = pd.to_datetime(
                inputs_copy[self._time_column] - 1, unit="D"
            )
        else:
            inputs_copy[self._time_column] = pd.to_datetime(
                inputs_copy[self._time_column], unit="s"
            )

        # find marked 'GroupingKey' or 'SuggestedGroupingKey'
        grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
        else:
            inputs_copy = inputs_copy.drop(
                columns=[list(inputs_copy)[i] for i in suggested_grouping_keys]
            )

        # extract n_periods, interval
        n_periods, forecast_idxs, all_intervals = self._calculate_prediction_intervals(
            inputs_copy, len(grouping_keys)
        )
        forecasts = self._forecast(n_periods, return_conf_int=return_conf_int)

        a = self.hyperparams["confidence_interval_alpha"]
        if return_conf_int:
            columns = [[t, f"{t}-{a/2}", f"{t}-{1-a/2}"] for t in self._targets]
            columns = [cols for cols in columns]
        else:
            columns = self._targets

        var_df = []
        for (grp_idx, col_idx), intervals in zip(forecast_idxs, all_intervals):
            forecast = forecasts[grp_idx]
            if col_idx == -1:

                nan_array = np.empty(
                    (len(intervals), len(forecast) * len(self._target_indices))
                )
                nan_array[:] = np.nan
                data = pd.DataFrame(nan_array)
            else:
                col_idxs = [col_idx + target for target in self._target_indices]
                data = [f.iloc[intervals, col_idxs] for f in forecast]
                data = pd.concat(data, axis=1)
            data.columns = columns
            var_df.append(data)
        var_df = pd.concat(var_df, axis=0, ignore_index=True)
        var_df = d3m_DataFrame(var_df)

        # assign target metadata and round appropriately
        idx = 0
        for tgt_name in self._targets:
            col_dict = dict(var_df.metadata.query((metadata_base.ALL_ELEMENTS, idx)))
            col_dict["structural_type"] = type("1")
            col_dict["name"] = tgt_name
            col_dict["semantic_types"] = (
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
                "http://schema.org/Float",
            )
            var_df.metadata = var_df.metadata.update(
                (metadata_base.ALL_ELEMENTS, idx), col_dict
            )

            if return_conf_int:
                col_dict_lower = dict(
                    var_df.metadata.query((metadata_base.ALL_ELEMENTS, idx + 1))
                )
                col_dict_upper = dict(
                    var_df.metadata.query((metadata_base.ALL_ELEMENTS, idx + 2))
                )
                col_dict_lower["structural_type"] = type("1")
                col_dict_upper["structural_type"] = type("1")
                col_dict_lower["semantic_types"] = ("http://schema.org/Float",)
                col_dict_upper["semantic_types"] = ("http://schema.org/Float",)
                col_dict_lower["name"] = f"{tgt_name}-{a/2}"
                col_dict_upper["name"] = f"{tgt_name}-{1-a/2}"
                var_df.metadata = var_df.metadata.update(
                    (metadata_base.ALL_ELEMENTS, idx + 1), col_dict_lower
                )
                var_df.metadata = var_df.metadata.update(
                    (metadata_base.ALL_ELEMENTS, idx + 2), col_dict_upper
                )

                idx += 3
            else:
                idx += 1

        return CallResult(var_df, has_finished=self._is_fit)