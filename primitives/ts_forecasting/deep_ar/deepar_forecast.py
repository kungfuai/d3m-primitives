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
# logger.setLevel(logging.DEBUG)

class DeepARForecast:

    def __init__(
        self,
        train_dataset: DeepARDataset,
        predictor_filepath: str,
        mean: bool = True,
        num_samples: int = 100,
        quantiles: List[float] = []
    ):
        """ if mean False, will return median point estimates """ 

        self.train_dataset = train_dataset
        self.train_frame = train_dataset.get_frame()
        self.predictor = GluonPredictor.deserialize(Path(predictor_filepath))
        self.mean = mean
        self.prediction_length = train_dataset.get_pred_length()
        self.context_length = train_dataset.get_context_length()
        self.num_samples = num_samples
        self.quantiles = quantiles

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

    def _resize_in_sample(self, forecasts, total_in_sample):
        """ resize in-sample forecasts """
        forecasts = np.stack(forecasts, axis=1).reshape(forecasts.shape[1], -1) # Quantiles, In-Sample Horizon
        forecasts = forecasts[:, :total_in_sample-self.context_length] # Quantiles, In-Sample Horizon - Context Length
        forecasts = np.concatenate(
            (np.zeros((forecasts.shape[0], self.context_length)), forecasts), 
            axis = 1
        ) # Quantiles, In-Sample Horizon
        return forecasts 

    def _iterate_in_sample(self, df):
        """ iterate through a single series to make in-sample forecasts"""
        data = []
        for stop_idx in range(
            self.context_length, 
            df.shape[0], 
            self.prediction_length
        ):
            #print(f'stop i: {stop_idx}, start i: {stop_idx - self.context_length}')
            data.append(
                self.train_dataset.get_series(
                    self.train_dataset.get_targets(df),
                    self.train_dataset.get_features(df),
                    start_idx = stop_idx - self.context_length, 
                    test = True
                )
            )

        data = ListDataset(data, freq = self.train_dataset.get_freq())
        forecasts = self._forecast(data)
        return self._resize_in_sample(forecasts, df.shape[0])

    def predict_in_sample(self):
        """ make in sample predictions"""

        if self.train_dataset.has_group_cols():
            all_series = []
            for _, df in self.train_frame.groupby(
                self.train_dataset.get_group_names(), sort = False
            ):
                series_forecast = self._iterate_in_sample(df)
                all_series.append(series_forecast)

        else:
            all_series = [self._iterate_in_sample(self.train_frame)]
        return all_series # Num Series, Quantiles, Horizon

    def predict_out_of_sample(self, future_feat):
        """ make forecasts for future time steps """
        
        train_feat = self.train_dataset.get_features(self.train_frame)
        all_feat = pd.concat((train_feat, future_feat))
        test_data = self.train_dataset.get_data(all_feat, test = True)
        return self._forecast(test_data) # Num Series, Quantiles, Horizon
