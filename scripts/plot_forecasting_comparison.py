from typing import List

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style="whitegrid")

from compare_forecasting_methods import pred_lengths


def to_query(
    elements: List[str],
):
    if len(elements) == 1:
        return elements[0]
    else:
        return "(" + " or ".join(elements) + ")"


def plot(
    metrics: str = "ts_metrics.csv",
    datasets: str = "Sorghum",
    horizon: str = "Short",
    metric: str = "MAPE",
    predictors: List[str] = [
        "DeepAR",
        "DeepFactor",
        "DeepState",
        "NBEATS",
        "NBEATS-Interp",
        "MQCNN",
        "MQRNN",
        "WaveNet",
        "NPTS",
    ],
):

    metrics = pd.read_csv("ts_metrics.csv")

    if datasets == "Sorghum":
        dataset_names = [
            "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA",
            "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA",
            "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA",
            "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA",
            "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA",
        ]
    elif datasets == "Malnutrition":
        dataset_names = [
            "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA",
            "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA",
        ]
    else:
        raise ValueError("'Datasets' must be one of 'Sorghum' or 'Malnutrition'")

    if horizon == "Short":
        pred_ls = [pred_lengths[dataset_name][0] for dataset_name in dataset_names]
    elif horizon == "Long":
        pred_ls = [pred_lengths[dataset_name][1] for dataset_name in dataset_names]
    else:
        raise ValueError("'Horizon' must be one of 'Short' or 'Long'")

    pred_list = to_query([f'Pred_Length=="{pred_l}"' for pred_l in pred_ls])
    dataset_list = to_query(
        [f'Dataset=="{dataset_name}"' for dataset_name in dataset_names]
    )
    predictor_list = to_query([f'Predictor=="{predictor}"' for predictor in predictors])
    query_list = pred_list + " and " + dataset_list + " and " + predictor_list

    df_slice = metrics.query(query_list)
    plt.clf()
    sns.barplot(x="Predictor", y=metric, data=df_slice)
    plt.xticks(rotation=45)
    plt.subplots_adjust(bottom=0.3)
    plt.xlabel("Forecasting Method")
    plt.title(f"Average {metric} on {datasets} Datasets with {horizon} Horizon")
    plt.savefig(f"{datasets}_{horizon}.png")


plot(
    datasets="Sorghum",
    horizon="Short",
    metric="MAPE",
)
plot(
    datasets="Sorghum",
    horizon="Long",
    metric="MAPE",
)
plot(
    datasets="Malnutrition",
    horizon="Short",
    metric="MAPE",
)
plot(
    datasets="Malnutrition",
    horizon="Long",
    metric="MAPE",
)
