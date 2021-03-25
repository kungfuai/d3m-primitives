from pathlib import Path
import logging
from typing import List

import numpy as np
import pandas as pd
from gluonts.model.predictor import GluonPredictor
from gluonts.gluonts_tqdm import tqdm
from gluonts.dataset.common import ListDataset
from gluonts.core.serde import load_json

from .deepar_dataset import DeepARDataset

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class DeepARForecast:
    def __init__(
        self,
        train_dataset: DeepARDataset,
        predictor_filepath: str,
        mean: bool = True,
        num_samples: int = 100,
        quantiles: List[float] = [],
        nan_padding: bool = True,
    ):
        """constructs DeepAR forecast object

        if mean False, will return median point estimates
        """

        self.train_dataset = train_dataset
        self.train_frame = train_dataset.get_frame()
        self.predictor = GluonPredictor.deserialize(Path(predictor_filepath))
        self.mean = mean
        self.prediction_length = train_dataset.get_pred_length()
        self.context_length = train_dataset.get_context_length()
        self.num_samples = num_samples
        self.quantiles = quantiles
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
            feat_df = self.train_dataset.get_features(self.train_frame)
            targets = self.train_dataset.get_targets(self.train_frame)
            min_interval = np.min(interval)
            max_interval = np.max(interval)
            self.max_intervals.append(max_interval)
            if np.max(interval) >= targets.shape[0]:
                feat_df = pd.concat((feat_df, test_frame))
            self._iterate_over_series(
                0,
                feat_df,
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
                    feat_df = self.train_dataset.get_features(train_df)
                    targets = self.train_dataset.get_targets(train_df)
                    if np.max(interval) >= targets.shape[0]:
                        feat_df = pd.concat((feat_df, test_df))
                    self._iterate_over_series(
                        series_idx, feat_df, targets, min_interval, max_interval
                    )
        self.series_idxs = np.array(self.series_idxs)
        self.data = ListDataset(self.data, freq=self.train_dataset.get_freq())
        forecasts = self._forecast()
        forecasts = self._pad(forecasts)
        return forecasts  # Num Series, Quantiles, Horizon

    def _iterate_over_series(
        self, series_idx, feat_df, targets, min_interval, max_interval
    ):
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
                        targets, feat_df, start_idx=start_idx, test=True
                    )
                )
                self.series_idxs.append(series_idx)
        else:
            self.series_idxs.append(-1)
            logger.info(
                f"This model was trained to use a context length of {self.context_length}, but there are "
                + f"only  {targets.shape[0]} in this series. These predictions will be returned as np.nan"
            )

        self.total_in_samples.append(targets.shape[0])
        self.pre_pad_lens.append(start + self.context_length)

    def _forecast(self):
        """ make forecasts for all series contained in data """

        all_forecasts = []
        with tqdm(
            self.predictor.predict(self.data, num_samples=self.num_samples),
            total=len(self.data),
            desc="Making Predictions",
        ) as forecasts, np.errstate(invalid="ignore"):
            for forecast in forecasts:
                point_estimate = forecast.mean if self.mean else forecast.quantile(0.5)
                quantiles = np.vstack(
                    [point_estimate] + [forecast.quantile(q) for q in self.quantiles]
                )
                all_forecasts.append(quantiles)
        return np.array(all_forecasts)  # Batch/Series, Quantiles, Prediction Length

    def _pad(self, forecasts):
        """ resize forecasts according to pre_pad_len """

        padded_forecasts = []
        series_idx = 0
        clean_series_idxs = self.series_idxs[self.series_idxs != -1]
        for series_val in self.series_idxs:
            if series_val == -1:
                series_forecasts = np.empty(
                    (
                        len(self.quantiles) + 1,
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
                )  # Quantiles, In-Sample Horizon
                if self.pre_pad_lens[series_idx] > 0:
                    padding = np.empty(
                        (series_forecasts.shape[0], self.pre_pad_lens[series_idx])
                    )
                    padding[:] = np.nan
                    series_forecasts = np.concatenate(
                        (padding, series_forecasts), axis=1
                    )  # Quantiles, Context Length + Horizon
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
                    )  # Quantiles, Context Length + Horizon + Post Padding
                    logger.info(
                        f"Asking for a prediction {self.max_intervals[series_idx] - self.total_in_samples[series_idx]} "
                        + f"steps into the future from a model that was trained to predict a maximum of {self.prediction_length} "
                        + "steps into the future. This prediction will be returned as np.nan"
                    )
            padded_forecasts.append(series_forecasts)
            series_idx += 1
        return padded_forecasts