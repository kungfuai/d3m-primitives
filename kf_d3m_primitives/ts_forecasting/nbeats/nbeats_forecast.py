from pathlib import Path
import logging
from typing import List

import numpy as np
import pandas as pd
from gluonts.model.predictor import GluonPredictor
from gluonts.gluonts_tqdm import tqdm
from gluonts.dataset.common import ListDataset
from gluonts.core.serde import load_json

from .nbeats_dataset import NBEATSDataset

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class NBEATSForecast:
    def __init__(
        self,
        train_dataset: NBEATSDataset,
        predictor_filepath: str,
        interpretable: bool = True,
        mean: bool = True,
        nan_padding: bool = True,
    ):
        """constructs NBEATS forecast object

        if mean False, will return median point estimates
        """

        self.train_dataset = train_dataset
        self.train_frame = train_dataset.get_frame()
        self.predictor = GluonPredictor.deserialize(Path(predictor_filepath))
        self.interpretable = interpretable
        if interpretable:
            self.mean = True
        else:
            self.mean = mean
        self.prediction_length = train_dataset.get_pred_length()
        self.context_length = train_dataset.get_context_length()
        self.nan_padding = nan_padding

        self.data = []
        self.series_idxs = []
        self.max_intervals = []
        self.pre_pad_lens = []
        self.total_in_samples = []

    def predict(self, test_frame, pred_intervals):
        """makes in-sample, out-of-sample, or both in-sample and out-of-sample
        predictions using test_frame for all timesteps included in pred_intervals
        """

        if not self.train_dataset.has_group_cols():
            interval = pred_intervals[0]
            targets = self.train_dataset.get_targets(self.train_frame)
            min_interval = np.min(interval)
            max_interval = np.max(interval)
            self.max_intervals.append(max_interval)
            self._iterate_over_series(
                0,
                targets,
                min_interval,
                max_interval,
            )
        else:
            group_cols = self.train_dataset.get_group_names()
            for series_idx, ((group, test_df), interval) in enumerate(
                zip(test_frame.groupby(group_cols, sort=False), pred_intervals)
            ):
                if len(group_cols) == 1:
                    group = [group]
                query_list = [
                    f'{grp_col}=="{grp}"' for grp_col, grp in zip(group_cols, group)
                ]
                train_df = self.train_frame.query(" & ".join(query_list))
                min_interval = np.min(interval)
                max_interval = np.max(interval)
                self.max_intervals.append(max_interval)
                if not train_df.shape[0]:
                    self.series_idxs.append(-1)
                    self.pre_pad_lens.append(0)
                    self.total_in_samples.append(0)
                else:
                    targets = self.train_dataset.get_targets(train_df)
                    self._iterate_over_series(
                        series_idx, targets, min_interval, max_interval
                    )
        self.series_idxs = np.array(self.series_idxs)
        self.data = ListDataset(self.data, freq=self.train_dataset.get_freq())
        forecasts = self._forecast()
        forecasts = self._pad(forecasts)
        return forecasts  # Num Series, 1/3, Horizon

    def _iterate_over_series(self, series_idx, targets, min_interval, max_interval):
        """ iterate over a single series to make forecasts using min_interval and max_interval"""

        logger.debug(
            f"min: {min_interval}, max: {max_interval}, total_in_sample: {targets.shape[0]}"
        )
        if max_interval < targets.shape[0]:  # all in-sample
            start = 0
            stop = targets.shape[0] - self.context_length
        elif min_interval < targets.shape[0]:  # some in-sample, some out-of-sample
            train_l_cutoff = targets.shape[0] - self.context_length
            train_batches = (min_interval - targets.shape[0]) // self.prediction_length
            start = max(
                train_l_cutoff % self.prediction_length,
                train_l_cutoff + self.prediction_length * train_batches,
            )
            stop = targets.shape[0]
        else:  # out-of-sample
            start = targets.shape[0] - self.context_length
            stop = targets.shape[0]

        logger.debug(f"start: {start}, stop: {stop}")
        if start >= 0 and stop > start:
            for start_idx in range(start, stop, self.prediction_length):
                logger.debug(
                    f"context start: {start_idx} context end: {start_idx + self.context_length}, "
                    + f"pred end: {start_idx + self.context_length + self.prediction_length}"
                )
                self.data.append(
                    self.train_dataset.get_series(
                        targets, start_idx=start_idx, test=True
                    )
                )
                self.series_idxs.append(series_idx)
        else:
            self.series_idxs.append(-1)
            logger.info(
                f"This model was trained to use a context length of {self.context_length}, but there are "
                + f"only {targets.shape[0]} in this series. These predictions will be returned as np.nan"
            )

        self.total_in_samples.append(targets.shape[0])
        self.pre_pad_lens.append(start + self.context_length)

    def _forecast(self):
        """ make forecasts for all series contained in data """

        all_forecasts = []
        with tqdm(
            self.predictor.predict(self.data),
            total=len(self.data),
            desc="Making Predictions",
        ) as forecasts, np.errstate(invalid="ignore"):
            for forecast in forecasts:
                if self.mean:
                    point_estimate = np.mean(forecast.samples, axis=0)
                else:
                    point_estimate = np.median(forecast.samples, axis=0)

                all_forecasts.append(point_estimate)
        all_forecasts = np.array(all_forecasts)
        if self.interpretable and len(self.data) > 0:
            trends = []
            for predictor in self.predictor.predictors:
                trends.append(predictor.prediction_net.get_trend_forecast())
                predictor.prediction_net.clear_trend_forecast()
            trends = np.stack(trends)
            trends = np.mean(trends, axis=0)
            trends = np.expand_dims(trends, axis=1)
            seasonalities = all_forecasts - trends

            all_forecasts = np.concatenate(
                (all_forecasts, trends, seasonalities), axis=1
            )
        return all_forecasts  # Batch/Series, Components, Prediction Length

    def _pad(self, forecasts):
        """ resize forecasts according to pre_pad_len """

        padded_forecasts = []
        series_idx = 0
        clean_series_idxs = self.series_idxs[self.series_idxs != -1]
        for series_val in self.series_idxs:
            if series_val == -1:
                dim_0 = 3 if self.interpretable else 1
                series_forecasts = np.empty(
                    (
                        dim_0,
                        self.max_intervals[series_idx]
                        + self.total_in_samples[series_idx]
                        + 1,
                    )
                )
                if self.nan_padding:
                    series_forecasts[:] = np.nan
                else:
                    series_forecasts[:] = 0
            elif series_val < series_idx:
                continue
            else:
                idxs = np.where(clean_series_idxs == series_val)[0]
                series_forecasts = forecasts[idxs]
                series_forecasts = np.stack(series_forecasts, axis=1).reshape(
                    series_forecasts.shape[1], -1
                )  # Components, In-Sample Horizon
                if self.pre_pad_lens[series_idx] > 0:
                    padding = np.empty(
                        (series_forecasts.shape[0], self.pre_pad_lens[series_idx])
                    )
                    padding[:] = np.nan
                    series_forecasts = np.concatenate(
                        (padding, series_forecasts), axis=1
                    )  # Components, Context Length + Horizon
                if self.max_intervals[series_idx] >= series_forecasts.shape[1]:
                    padding = np.empty(
                        (
                            series_forecasts.shape[0],
                            self.max_intervals[series_idx]
                            - series_forecasts.shape[1]
                            + 1,
                        )
                    )
                    if self.nan_padding:
                        padding[:] = np.nan
                    else:
                        padding[:] = series_forecasts[:, -1][:, np.newaxis]
                    series_forecasts = np.concatenate(
                        (series_forecasts, padding), axis=1
                    )  # Components, Context Length + Horizon + Post Padding
                    logger.info(
                        f"Asking for a prediction {self.max_intervals[series_idx] - self.total_in_samples[series_idx]} "
                        + f"steps into the future from a model that was trained to predict a maximum of {self.prediction_length} "
                        + "steps into the future. This prediction will be returned as np.nan"
                    )
            padded_forecasts.append(series_forecasts)
            series_idx += 1
        return padded_forecasts