from typing import List, Union

import pandas as pd
import numpy as np
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.distribution import NegativeBinomialOutput, StudentTOutput


class NBEATSDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        group_cols: List[int],
        time_col: int,
        target_col: int,
        freq: str,
        prediction_length: int,
        num_context_lengths: int,
    ):
        """initialize NBEATSDataset object"""

        self.frame = frame
        self.group_cols = group_cols
        self.time_col = time_col
        self.target_col = target_col
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = (num_context_lengths + 1) * prediction_length

        if self.has_group_cols():
            g_cols = self.get_group_names()
            self.targets = frame.groupby(g_cols, sort=False)[frame.columns[target_col]]
        else:
            self.targets = self.get_targets(frame)

    def get_targets(self, df):
        """ gets targets from df using target_col of object """
        return df.iloc[:, self.target_col]

    def get_series(
        self,
        targets: pd.Series,
        test=False,
        start_idx=0,
    ):
        """creates dictionary of start time, targets for one individual time series

        if test, creates dictionary from subset of indices using start_idx
        """

        if not test:
            start_idx = 0

        features = {FieldName.START: targets.index[start_idx]}

        if test:
            features[FieldName.TARGET] = targets.iloc[
                start_idx : start_idx + self.context_length
            ].values
        else:
            features[FieldName.TARGET] = targets.values

        return features

    def get_data(self):
        """ creates train dataset object """

        if self.has_group_cols():
            data = []
            g_cols = self.get_group_names()
            for (_, targets) in self.targets:
                data.append(self.get_series(targets))
        else:
            data = [self.get_series(self.targets)]

        return ListDataset(data, freq=self.freq)

    def get_group_names(self):
        """ transform column indices to column names """
        return [self.frame.columns[i] for i in self.group_cols]

    def has_group_cols(self):
        """ does this NBEATS dataset have grouping columns """
        return len(self.group_cols) != 0

    def get_frame(self):
        """ get data frame associated with this NBEATS dataset """
        return self.frame

    def get_freq(self):
        """ get frequency associated with this NBEATS dataset """
        return self.freq

    def get_pred_length(self):
        """ get prediction length associated with this NBEATS dataset """
        return self.prediction_length

    def get_context_length(self):
        """ get context length associated with this NBEATS dataset """
        return self.context_length

    def get_time_col(self):
        """ get time column associated with this NBEATS dataset """
        return self.time_col