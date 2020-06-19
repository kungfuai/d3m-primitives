from pathlib import Path
import logging
from typing import List

import numpy as np
import pandas as pd
from gluonts.model.predictor import GluonPredictor
from gluonts.gluonts_tqdm import tqdm
from gluonts.dataset.common import ListDataset

from .deepar_dataset import DeepARDataset

logger = logging.getLogger(__name__)

class DeepARForecast:

    def __init__(
        self,
        train_dataset: DeepARDataset,
        predictor_filepath: str,
        mean: bool = True,
        num_samples: int = 100,
        quantiles: List[float] = []
    ):
        """ constructs DeepAR forecast object
        
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

    def predict(self, test_frame, pred_intervals):
        """ makes in-sample, out-of-sample, or both in-sample and out-of-sample 
            predictions using test_frame for all timesteps included in pred_intervals
        """

        if not self.train_dataset.has_group_cols():
            interval = pred_intervals[0]
            feat_df = self.train_dataset.get_features(self.train_frame)
            targets = self.train_dataset.get_targets(self.train_frame)
            min_interval = np.min(interval)
            max_interval = np.max(interval)
            if np.max(interval) >= targets.shape[0]:
                feat_df = pd.concat((feat_df, test_frame))
            all_series = [
                self._iterate_over_series(
                    feat_df, 
                    targets, 
                    min_interval, 
                    max_interval, 
                )
            ]
        else:
            all_series = []
            group_cols = self.train_dataset.get_group_names()
            for (group, test_df), interval in zip(
                test_frame.groupby(group_cols, sort = False), 
                pred_intervals
            ):            
                if len(group_cols) == 1:
                    group = [group]
                query_list = [
                    f'{grp_col}=="{grp}"' for grp_col, grp in zip(group_cols, group)
                ]
                train_df = self.train_frame.query(' & '.join(query_list))
                if not train_df.shape[0]:
                    forecasts = np.empty((len(self.quantiles) + 1, 2)) 
                    forecasts[:] = np.nan
                else:
                    feat_df = self.train_dataset.get_features(train_df)
                    targets = self.train_dataset.get_targets(train_df)
                    min_interval = np.min(interval)
                    max_interval = np.max(interval)
                    if np.max(interval) >= targets.shape[0]:
                        feat_df = pd.concat((feat_df, test_df))
                    forecasts = self._iterate_over_series(
                        feat_df, 
                        targets, 
                        min_interval, 
                        max_interval, 
                    )
                all_series.append(forecasts)
        
        return all_series # Num Series, Quantiles, Horizon

    def _iterate_over_series(
        self, 
        feat_df, 
        targets, 
        min_interval, 
        max_interval, 
    ):
        """ iterate over a single series to make forecasts using min_interval and max_interval"""

        data = []
        #print(f'context: {self.context_length}, pred: {self.prediction_length}')
        #print(f'min: {min_interval}, max: {max_interval}, total_in_sample: {targets.shape[0]}')
        if max_interval < targets.shape[0]: # all in-sample
            start = 0
            stop = targets.shape[0] - self.context_length
        elif min_interval < targets.shape[0]: # some in-sample, some out-of-sample
            train_l_cutoff = targets.shape[0] - self.context_length
            train_batches = (min_interval - targets.shape[0]) // self.prediction_length
            start = max(
                train_l_cutoff % self.prediction_length, 
                train_l_cutoff + self.prediction_length * train_batches
            )
            stop = targets.shape[0]
        else: # out-of-sample
            start = targets.shape[0] - self.context_length
            stop = targets.shape[0]
        #print(f'start: {start}, stop: {stop}')
        if start >= 0 and stop > start:
            for start_idx in range(start, stop, self.prediction_length):
                #print(f'context start: {start_idx} context end: {start_idx + self.context_length}, pred end: {start_idx + self.context_length + self.prediction_length}')
                data.append(
                    self.train_dataset.get_series(
                        targets,
                        feat_df,
                        start_idx = start_idx, 
                        test = True
                    )
                )

            data = ListDataset(data, freq = self.train_dataset.get_freq())
            forecasts = self._forecast(data)
        else:
            forecasts = np.empty((1, len(self.quantiles) + 1, self.prediction_length))
            forecasts[:] = np.nan
            logger.info(
                f"This model was trained to use a context length of {self.context_length}, but there are " + 
                f"only  {targets.shape[0]} in this series. These predictions will be returned as np.nan"
            )
        #print(f'forecast shape: {forecasts.shape}')
        return self._pad(
            forecasts, 
            max_interval, 
            self.context_length + start,
            targets.shape[0]
        )

    def _forecast(self, data):
        """ make forecasts for all series contained in data """

        all_forecasts = []
        with tqdm(
            self.predictor.predict(data, num_samples = self.num_samples),
            total=len(data),
            desc="Making Predictions"
        ) as it, np.errstate(invalid='ignore'):
            for forecast in it:
                point_estimate = forecast.mean if self.mean else forecast.quantile(0.5)
                quantiles = np.vstack(
                    [point_estimate] + 
                    [forecast.quantile(q) for q in self.quantiles]
                )
                all_forecasts.append(quantiles)
        return np.array(all_forecasts) # Batch/Series, Quantiles, Prediction Length

    def _pad(self, forecasts, max_interval, pre_pad_len, total_in_sample):
        """ resize forecasts according to pre_pad_len """
        #print(f'pre pad: {pre_pad_len}')
        forecasts = np.stack(forecasts, axis=1).reshape(forecasts.shape[1], -1) # Quantiles, In-Sample Horizon
        #print(f'f shape: {forecasts.shape}')
        if pre_pad_len > 0:
            padding = np.empty((forecasts.shape[0], pre_pad_len))
            padding[:] = np.nan
            forecasts = np.concatenate((padding, forecasts), axis = 1) # Quantiles, Context Length + Horizon
        #print(f'f shape: {forecasts.shape}')
        if max_interval >= forecasts.shape[1]:
            padding = np.empty((forecasts.shape[0], max_interval - forecasts.shape[1] + 1))
            padding[:] = np.nan
            forecasts = np.concatenate((forecasts, padding), axis = 1) # Quantiles, Context Length + Horizon + Post Padding
            logger.info(
                f"Asking for a prediction {max_interval - total_in_sample} steps into the future " + 
                f"from a model that was trained to predict a maximum of {self.prediction_length} steps " +
                "into the future. This prediction will be returned as np.nan"
            )
        #print(f'f shape: {forecasts.shape}')
        return forecasts 