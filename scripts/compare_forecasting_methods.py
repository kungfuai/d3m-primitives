from typing import List, Union, Tuple
import json
import pickle
from os import path
from argparse import ArgumentParser
from pathlib import Path

import pandas as pd
import numpy as np
import mxnet as mx
from sklearn.preprocessing import OrdinalEncoder
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.dataset.common import ListDataset
from gluonts.dataset.field_names import FieldName
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.lstnet import LSTNetEstimator
from gluonts.model.n_beats import NBEATSEstimator, NBEATSEnsembleEstimator
from gluonts.model.npts import NPTSPredictor
from gluonts.model.prophet import ProphetPredictor
from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.trainer import Trainer
from gluonts.distribution import NegativeBinomialOutput
from gluonts.model.predictor import Predictor
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.evaluation import Evaluator

from explore_time_series import (
    sort_time,
    datetime_format_strs,
    time_cols,
    grouping_cols,
    target_cols,
    freqs,
    reind_freqs,
    plot_hist,
)

Model = Union[
    DeepAREstimator,
    DeepFactorEstimator,
    DeepStateEstimator,
    NBEATSEstimator,
    MQCNNEstimator,
    MQRNNEstimator,
    WaveNetEstimator,
]

pred_lengths = {
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": [10, 20],
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": [10, 25],
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": [10, 25],
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": [10, 25],
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": [10, 15],
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": [5, 8],
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": [5, 10],
}

date_cutoffs = {
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": [30, 45],
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": [30, 60],
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": [30, 60],
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": [30, 60],
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": [45, 45],
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": ["2018-01-01", "2018-01-01"],
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": ["2018-01-01", "2018-01-01"],
}


def encode_categoricals(
    data: pd.DataFrame, group_cols: List[str]
) -> (pd.DataFrame, OrdinalEncoder):
    enc = OrdinalEncoder()
    data[group_cols] = enc.fit_transform(data[group_cols].values)
    return data, enc


def prep_datasets(
    pred_length: int, dataset_name: str, date_cutoff: int
) -> Tuple[Tuple[ListDataset], Tuple[ListDataset], List[int]]:

    if path.exists(f"ts_datasets/{dataset_name}_{pred_length}_training.pkl"):

        training_datasets = pickle.load(
            open(f"ts_datasets/{dataset_name}_{pred_length}_training.pkl", "rb")
        )
        validation_datasets = pickle.load(
            open(f"ts_datasets/{dataset_name}_{pred_length}_validation.pkl", "rb")
        )
        cardinalities = pickle.load(
            open(f"ts_datasets/{dataset_name}_cardinalities.pkl", "rb")
        )

    else:
        data = pd.read_csv(
            f"../datasets/seed_datasets_current/{dataset_name}/TRAIN/dataset_TRAIN/tables/learningData.csv"
        )
        data = sort_time(data, dataset_name)

        group_cols = [data.columns[c] for c in grouping_cols[dataset_name]]
        data, _ = encode_categoricals(data, group_cols)
        cardinalities = [data[c].nunique() for c in group_cols]
        plot_hist(
            data[data.columns[time_cols[dataset_name]]].values,
            "Distribution of training times",
        )
        date_cutoff = pd.to_datetime(
            date_cutoff, format=datetime_format_strs[dataset_name]
        )
        if freqs[dataset_name] == "D":
            cutoff_end = date_cutoff + pd.DateOffset(pred_length)
        elif freqs[dataset_name] == "M":
            cutoff_end = date_cutoff + pd.DateOffset(months=pred_length)
        elif freqs[dataset_name] == "W":
            cutoff_end = date_cutoff + pd.DateOffset(weeks=pred_length)

        train_dataset, val_dataset = [], []
        train_deepf, val_deepf = [], []
        train_interpolate, val_interpolate = [], []
        drop_ct = 0
        val_obs = []
        for i, (grp_name, df) in enumerate(data.groupby(group_cols)):

            df = df.drop_duplicates(subset=data.columns[time_cols[dataset_name]])

            if df.shape[0] == 1:
                print(f"Dropped series {grp_name} because it only had 1 observation")
                drop_ct += 1
                continue
            if df.index[0] >= date_cutoff:
                print(
                    f"Dropped series {grp_name} because it has no observations in training set"
                )
                drop_ct += 1
                continue
            if df.index[-1] < date_cutoff:
                print(
                    f"Dropped series {grp_name} because it does not have any observations in validation range"
                )
                drop_ct += 1
                continue

            df = df.reindex(
                pd.date_range(df.index[0], df.index[-1], freq=reind_freqs[dataset_name])
            )

            target_col = df.columns[target_cols[dataset_name]]
            grp_query = [
                f'{grp_col}=="{key}"' for grp_col, key in zip(group_cols, grp_name)
            ]
            feat_static_cat = df.query(" & ".join(grp_query))[group_cols].values[0]
            target_interpolated = df[target_col][:date_cutoff].interpolate()

            train_dataset.append(
                {
                    FieldName.START: df.index[0],
                    FieldName.TARGET: df[target_col][:date_cutoff].values,
                    FieldName.FEAT_STATIC_CAT: feat_static_cat,
                }
            )
            train_deepf.append(
                {
                    FieldName.START: df.index[0],
                    FieldName.TARGET: target_interpolated.values,
                    FieldName.FEAT_STATIC_CAT: [i],
                }
            )
            train_interpolate.append(
                {
                    FieldName.START: df.index[0],
                    FieldName.TARGET: target_interpolated.values,
                    FieldName.FEAT_STATIC_CAT: feat_static_cat,
                }
            )

            val_dataset.append(
                {
                    FieldName.START: df[:cutoff_end].index[0],
                    FieldName.TARGET: df[target_col][:cutoff_end].values,
                    FieldName.FEAT_STATIC_CAT: feat_static_cat,
                }
            )
            val_deepf.append(
                {
                    FieldName.START: df[:cutoff_end].index[0],
                    FieldName.TARGET: target_interpolated.append(
                        df[target_col][date_cutoff:cutoff_end]
                    ).values,
                    FieldName.FEAT_STATIC_CAT: [i],
                }
            )
            val_interpolate.append(
                {
                    FieldName.START: df[:cutoff_end].index[0],
                    FieldName.TARGET: target_interpolated.append(
                        df[target_col][date_cutoff:cutoff_end]
                    ).values,
                    FieldName.FEAT_STATIC_CAT: feat_static_cat,
                }
            )

            val_obs.append(df[target_col][date_cutoff:cutoff_end].count())

        print(f"Dropped {drop_ct} series, there are now {len(train_dataset)} series.")
        print(
            f"There are an average of {np.mean(val_obs)} validation observations per series"
        )
        plot_hist(val_obs, f"Distribution of Number of Validation Observations")

        training_datasets = (
            ListDataset(train_dataset, freq=freqs[dataset_name]),
            ListDataset(train_deepf, freq=freqs[dataset_name]),
            ListDataset(train_interpolate, freq=freqs[dataset_name]),
        )
        validation_datasets = (
            ListDataset(val_dataset, freq=freqs[dataset_name]),
            ListDataset(val_deepf, freq=freqs[dataset_name]),
            ListDataset(val_interpolate, freq=freqs[dataset_name]),
        )

        pickle.dump(
            training_datasets,
            open(f"ts_datasets/{dataset_name}_{pred_length}_training.pkl", "wb"),
        )
        pickle.dump(
            validation_datasets,
            open(f"ts_datasets/{dataset_name}_{pred_length}_validation.pkl", "wb"),
        )
        pickle.dump(
            cardinalities, open(f"ts_datasets/{dataset_name}_cardinalities.pkl", "wb")
        )

    return (training_datasets, validation_datasets, cardinalities)


def prep_estimators(
    pred_length: int,
    dataset_name: str,
    num_series: int,
    cardinalities: List[int],
    epochs: int,
) -> List[Model]:

    trainer = Trainer(epochs=epochs)

    models = [
        # DeepAREstimator(
        #     freq = freqs[dataset_name],
        #     prediction_length = pred_length,
        #     trainer = trainer,
        #     use_feat_static_cat = True,
        #     cardinality = cardinalities,
        #     distr_output=NegativeBinomialOutput()
        # ),
        # DeepFactorEstimator(
        #     freq = freqs[dataset_name],
        #     prediction_length = pred_length,
        #     trainer = trainer,
        #     cardinality = [num_series],
        #     distr_output=NegativeBinomialOutput(),
        # ),
        # DeepStateEstimator(
        #     freq = freqs[dataset_name],
        #     prediction_length = pred_length,
        #     trainer = trainer,
        #     cardinality = cardinalities,
        # ),
        # NBEATSEstimator(
        #     freq = freqs[dataset_name],
        #     prediction_length = pred_length,
        #     trainer = trainer,
        #     # TODO is the loss function/evaluation metric we want to use?
        #     loss_function = 'MAPE',
        # ),
        NBEATSEnsembleEstimator(
            freq=freqs[dataset_name],
            prediction_length=pred_length,
            trainer=trainer,
            meta_bagging_size=1,
        ),
        NBEATSEnsembleEstimator(
            freq=freqs[dataset_name],
            prediction_length=pred_length,
            trainer=trainer,
            num_stacks=2,
            num_blocks=[3],
            widths=[256, 2048],
            sharing=[True],
            expansion_coefficient_lengths=[3],
            stack_types=["T", "S"],
            meta_bagging_size=1,
        ),
        # MQCNNEstimator(
        #     freq = freqs[dataset_name],
        #     prediction_length = pred_length,
        #     trainer = trainer,
        # ),
        # MQRNNEstimator(
        #     freq = freqs[dataset_name],
        #     prediction_length = pred_length,
        #     trainer = trainer,
        # ),
        # WaveNetEstimator(
        #     freq = freqs[dataset_name],
        #     prediction_length = pred_length,
        #     trainer = trainer,
        #     cardinality = cardinalities
        # ),
    ]
    return models


def fit_estimators(
    all_estimators: List[Model],
    training_datasets: Tuple[ListDataset],
    dataset_name: str,
    pred_length: int,
) -> List[Predictor]:

    train_data, train_deepf, train_interpolate = training_datasets

    predictors = []
    for estimator in all_estimators:
        print(f"Fitting {type(estimator)}")
        if type(estimator) is DeepFactorEstimator:
            predictor = estimator.train(train_deepf)
        elif type(estimator) is WaveNetEstimator:
            predictor = estimator.train(train_data)
        else:
            predictor = estimator.train(train_interpolate)
        predictors.append(predictor)

    predictors += [
        NPTSPredictor(
            freq=freqs[dataset_name],
            prediction_length=pred_length,
        ),
        #     # ProphetPredictor(
        #     #     freq = freqs[dataset_name],
        #     #     prediction_length = pred_length,
        #     # )
    ]

    return predictors


def evaluate_predictors(
    all_predictors: List[Predictor],
    validation_datasets: List[ListDataset],
    dataset_name: str,
    pred_length: int,
    epochs: int,
    predictor_names: List[str] = [
        # 'DeepAR',
        # 'DeepFactor',
        # 'DeepState',
        # 'NBEATS',
        "NBEATS-E",
        "NBEATS-EI",
        # 'MQCNN',
        # 'MQRNN',
        # 'WaveNet',
        # 'NPTS',
        #'Prophet'
    ],
):
    evaluator = Evaluator(quantiles=[0.5])

    val_data, val_deepf, val_interpolate = validation_datasets

    if path.exists("ts_metrics_nbeats.csv"):
        metrics = pd.read_csv(
            "ts_metrics_nbeats.csv",
            usecols=[
                "Dataset",
                "Predictor",
                "Pred_Length",
                "Epochs",
                "sMAPE",
                "MAPE",
                "RMSE",
            ],
        )
    else:
        metrics = pd.DataFrame(
            columns=[
                "Dataset",
                "Predictor",
                "Pred_Length",
                "Epochs",
                "sMAPE",
                "MAPE",
                "RMSE",
            ]
        )

    for predictor, name in zip(all_predictors, predictor_names):
        print(f"Evaluating {name}")
        if name == "DeepFactor":
            val_dataset = val_deepf
        elif name == "WaveNet":
            val_dataset = val_data
        else:
            val_dataset = val_interpolate

        forecast_it, ts_it = make_evaluation_predictions(
            dataset=val_dataset, predictor=predictor, num_samples=100
        )
        agg_metrics, _ = evaluator(ts_it, forecast_it)

        query_list = [
            f'Dataset=="{dataset_name}"',
            f'Predictor=="{name}"',
            f'Pred_Length=="{pred_length}"',
            f'Epochs=="{epochs}"',
        ]
        df_row = metrics.query(" & ".join(query_list))
        if df_row.shape[0] == 1:
            metrics["sMAPE"][df_row.index] = agg_metrics["sMAPE"]
            metrics["MAPE"][df_row.index] = agg_metrics["MAPE"]
            metrics["RMSE"][df_row.index] = agg_metrics["RMSE"]
        else:
            metrics.loc[metrics.shape[0]] = [
                dataset_name,
                name,
                pred_length,
                epochs,
                agg_metrics["sMAPE"],
                agg_metrics["MAPE"],
                agg_metrics["RMSE"],
            ]
        metrics.to_csv("ts_metrics_nbeats.csv")


def main(args):

    # set random seeds for reproducibility
    mx.random.seed(0)
    np.random.seed(0)

    dataset_name = args.dataset_name
    epochs = args.epochs

    for dataset_name in [
        "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA",
        # 'LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA',
        # 'LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA',
        # 'LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA',
        # 'LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA',
        # 'LL1_PHEM_Monthly_Malnutrition_MIN_METADATA',
        # 'LL1_PHEM_weeklyData_malnutrition_MIN_METADATA',
    ]:

        if args.pred_length == "short":
            pred_length = pred_lengths[dataset_name][0]
            date_cutoff = date_cutoffs[dataset_name][0]
        elif args.pred_length == "long":
            pred_length = pred_lengths[dataset_name][1]
            date_cutoff = date_cutoffs[dataset_name][1]

        train_datasets, val_datasets, cardinalities = prep_datasets(
            pred_length, dataset_name, date_cutoff
        )
        estimators = prep_estimators(
            pred_length, dataset_name, len(train_datasets[0]), cardinalities, epochs
        )
        predictors = fit_estimators(
            estimators, train_datasets, dataset_name, pred_length
        )
    # evaluate_predictors(
    #     predictors,
    #     val_datasets,
    #     dataset_name,
    #     pred_length,
    #     epochs
    # )


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset-name",
        type=str,
        default="LL1_PHEM_weeklyData_malnutrition_MIN_METADATA",
        help="The name of the dataset to explore",
    )
    parser.add_argument(
        "-e",
        "--epochs",
        type=int,
        default=1,
        help="The number of epochs for which to fit estimators",
    )
    parser.add_argument(
        "-p",
        "--pred-length",
        type=str,
        default="short",
        help='The prediction length. Current options are "short" and "long"',
    )
    args = parser.parse_args()
    main(args)