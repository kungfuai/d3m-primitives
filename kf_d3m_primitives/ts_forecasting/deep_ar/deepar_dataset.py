from typing import List, Union

import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.distribution import NegativeBinomialOutput, StudentTOutput


class DeepARDataset:
    def __init__(
        self,
        frame: pd.DataFrame,
        group_cols: List[int],
        cat_cols: List[int],
        real_cols: List[int],
        time_col: int,
        target_col: int,
        freq: str,
        prediction_length: int,
        context_length: int,
        target_semantic_types: List[str],
        count_data: Union[None, bool],
    ):
        """initialize DeepARDataset object"""

        self.frame = frame
        self.group_cols = group_cols
        self.cat_cols = cat_cols
        self.real_cols = real_cols
        self.time_col = time_col
        self.target_col = target_col
        self.freq = freq
        self.prediction_length = prediction_length
        self.context_length = context_length
        self.target_semantic_types = target_semantic_types
        self.count_data = count_data

        if self.has_group_cols():
            g_cols = self.get_group_names()
            self.targets = frame.groupby(g_cols, sort=False)[frame.columns[target_col]]
        else:
            self.targets = self.get_targets(frame)
        self.train_feat_df = self.get_features(frame)

        if self.has_cat_cols() or self.has_group_cols():
            self.enc = OrdinalEncoder().fit(self.frame.iloc[:, cat_cols + group_cols])

    def get_targets(self, df):
        """ gets targets from df using target_col of object """
        return df.iloc[:, self.target_col]

    def get_features(self, df):
        """ gets features from df using target col of object (all cols but) """
        return df.drop(df.columns[self.target_col], axis=1)

    def get_series(
        self,
        targets: pd.Series,
        feat_df: pd.DataFrame,
        test=False,
        start_idx=0,
    ):
        """creates dictionary of start time, features, targets for one individual time series

        if test, creates dictionary from subset of indices using start_idx
        """

        if not test:
            start_idx = 0

        assert feat_df.index[start_idx] == targets.index[start_idx]
        features = {FieldName.START: feat_df.index[start_idx]}

        if test:
            features[FieldName.TARGET] = targets.iloc[
                start_idx : start_idx + self.context_length
            ].values
        else:
            features[FieldName.TARGET] = targets.values

        if self.has_real_cols():
            if test:
                total_length = self.context_length + self.prediction_length
                real_features = feat_df.iloc[
                    start_idx : start_idx + total_length, self.real_cols
                ]
                if real_features.shape[0] < total_length:
                    real_features = self._pad_future_features(
                        real_features, total_length - real_features.shape[0]
                    )
            else:
                real_features = feat_df.iloc[:, self.real_cols]
            features[FieldName.FEAT_DYNAMIC_REAL] = real_features.values.reshape(
                len(self.real_cols), -1
            )

        if self.has_cat_cols() or self.has_group_cols():
            features[FieldName.FEAT_STATIC_CAT] = self.enc.transform(
                feat_df.iloc[0, self.cat_cols + self.group_cols].values.reshape(1, -1)
            ).reshape(
                -1,
            )

        return features

    def get_data(self):
        """ creates train dataset object """

        if self.has_group_cols():
            data = []
            g_cols = self.get_group_names()
            for (_, features), (_, targets) in zip(
                self.train_feat_df.groupby(g_cols, sort=False), self.targets
            ):
                data.append(self.get_series(targets, features))
        else:
            data = [self.get_series(self.targets, self.train_feat_df)]

        return ListDataset(data, freq=self.freq)

    def get_group_names(self):
        """ transform column indices to column names """
        return [self.frame.columns[i] for i in self.group_cols]

    def has_group_cols(self):
        """ does this DeepAR dataset have grouping columns """
        return len(self.group_cols) != 0

    def has_cat_cols(self):
        """ does this DeepAR dataset have categorical columns """
        return len(self.cat_cols) != 0

    def has_real_cols(self):
        """ does this DeepAR dataset have real valued columns """
        return len(self.real_cols) != 0

    def get_frame(self):
        """ get data frame associated with this DeepAR dataset """
        return self.frame

    def get_freq(self):
        """ get frequency associated with this DeepAR dataset """
        return self.freq

    def get_pred_length(self):
        """ get prediction length associated with this DeepAR dataset """
        return self.prediction_length

    def get_context_length(self):
        """ get context length associated with this DeepAR dataset """
        return self.context_length

    def get_time_col(self):
        """ get time column associated with this DeepAR dataset """
        return self.time_col

    def get_cardinality(self):
        """ get the cardinalities of categorical columns of dataset """
        if len(self.group_cols + self.cat_cols) != 0:
            return [self.frame.iloc[:, col].nunique() for col in self.group_cols]
        else:
            return None

    def get_distribution_type(self):
        """ get distribution type of dataset """
        if self.count_data:
            return NegativeBinomialOutput()
        elif self.count_data == False:
            return StudentTOutput()
        elif "http://schema.org/Integer" in self.target_semantic_types:
            if np.min(self.frame.iloc[:, self.target_col]) >= 0:
                return NegativeBinomialOutput()
            else:
                return StudentTOutput()
        elif "http://schema.org/Float" in self.target_semantic_types:
            return StudentTOutput()
        else:
            raise ValueError("Target column is not of type 'Integer' or 'Float'")

    def _pad_future_features(self, df, pad_length):
        """ pads feature df for test predictions that extend past support"""
        return df.append(df.iloc[[-1] * pad_length])