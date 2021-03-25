"""
Script to explore Terra and Phem time series datasets
"""

from typing import List
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tslearn.clustering import TimeSeriesKMeans
from tslearn import utils as ts_utils

""" Summary statistics
        -number of series 
        -frequency
        -distribution of training lengths
        -distribution of proportion missing observations 
        -distribution of proportion 0 observations
        -distribution of prediction lengths
        -distribution of training lengths (at lower grouping level)
        -look at examples from clusters
"""

""" Observed Similarities/Differences Between Terra/Phem Datasets
"""


datetime_format_strs = {
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": "%j",
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": "%j",
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": "%j",
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": "%j",
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": "%j",
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": "%Y-%m-%d",
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": "%Y-%m-%d",
}

freqs = {
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": "D",
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": "D",
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": "D",
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": "D",
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": "D",
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": "M",
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": "W",
}

reind_freqs = {
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": "D",
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": "D",
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": "D",
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": "D",
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": "D",
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": "M",
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": "W-MON",
}

target_cols = {
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": 4,
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": 4,
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": 4,
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": 4,
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": 4,
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": 5,
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": 5,
}

time_cols = {
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": 3,
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": 3,
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": 3,
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": 3,
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": 3,
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": 4,
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": 4,
}

grouping_cols = {
    "LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA": [1, 2],
    "LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA": [1, 2],
    "LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA": [1, 2],
    "LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA": [1, 2],
    "LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA": [1, 2],
    "LL1_PHEM_Monthly_Malnutrition_MIN_METADATA": [1, 2, 3],
    "LL1_PHEM_weeklyData_malnutrition_MIN_METADATA": [1, 2, 3],
}


def sort_time(df: pd.DataFrame, dataset_name: str):
    time_col = time_cols[dataset_name]
    df.iloc[:, time_col] = pd.to_datetime(
        df.iloc[:, time_col], format=datetime_format_strs[dataset_name]
    )
    df = df.sort_values(by=df.columns[time_col])
    df.index = df.iloc[:, time_col]
    return df


def read_data(dataset_name: str):
    train = pd.read_csv(
        f"../datasets/seed_datasets_current/{dataset_name}/TRAIN/dataset_TRAIN/tables/learningData.csv"
    )
    test = pd.read_csv(
        f"../datasets/seed_datasets_current/{dataset_name}/SCORE/dataset_SCORE/tables/learningData.csv"
    )
    return sort_time(train, dataset_name), sort_time(test, dataset_name)


def df_summary_stats(train: pd.DataFrame, test: pd.DataFrame, dataset_name: str):
    """Explore:
    -number of training series
    -number of test series
    -frequency of series
    """
    group_cols = [train.columns[c] for c in grouping_cols[dataset_name]]
    print(f"Exploring {dataset_name}")
    print(f"Number of training series: {train.groupby(group_cols).ngroups}")
    print(f"Number of testing series: {test.groupby(group_cols).ngroups}")
    print(f"Frequency: {freqs[dataset_name]}")


def train_test_dists(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    group_cols: List[str],
    dataset_name: str,
):
    series_lens = []
    num_obs = []
    num_0_obs = []
    pred_lens = []
    all_train_dfs = []
    all_test_dfs = []
    for (grp_name, test_f) in test_df.groupby(group_cols):

        query_list = [
            f'{grp_col}=="{key}"' for grp_col, key in zip(group_cols, grp_name)
        ]
        train_f = train_df.query(" & ".join(query_list))

        ## TODO in DeepAR sum instead of drop duplicates
        train_f = train_f.drop_duplicates(
            subset=train_f.columns[time_cols[dataset_name]]
        )
        num_obs.append(train_f.shape[0])
        num_0_obs.append((train_f.iloc[:, target_cols[dataset_name]] == 0).sum().sum())

        if train_f.shape[0] > 1:
            train_f = train_f.reindex(
                pd.date_range(
                    train_f.index[0], train_f.index[-1], freq=reind_freqs[dataset_name]
                )
            )
        train_f.iloc[:, grouping_cols[dataset_name]] = train_f.iloc[
            :, grouping_cols[dataset_name]
        ].ffill()

        test_f = test_f.reindex(
            pd.date_range(
                train_f.index[-1], test_f.index[-1], freq=reind_freqs[dataset_name]
            )
        )
        test_f.iloc[:, grouping_cols[dataset_name]] = test_f.iloc[
            :, grouping_cols[dataset_name]
        ].bfill()
        series_lens.append(train_f.shape[0])
        pred_len = (test_f.index[-1] - train_f.index[-1]).days
        if freqs[dataset_name] == "W":
            pred_len = pred_len / 7
        elif freqs[dataset_name] == "M":
            pred_len = pred_len / 30.5
        pred_lens.append(pred_len)
        all_train_dfs.append(train_f)
        all_test_dfs.append(test_f)
    prop_obs = [1 - (n / l) for n, l in zip(num_obs, series_lens)]
    prop_0_obs = [n_0 / n for n, n_0 in zip(num_obs, num_0_obs)]
    return (
        series_lens,
        prop_obs,
        prop_0_obs,
        pred_lens,
        pd.concat(all_train_dfs),
        pd.concat(all_test_dfs),
    )


def plot_hist(data: List[int], title: str):
    plt.clf()
    plt.hist(data)
    plt.title(title)
    plt.savefig(f"ts_explore/{title}")


def series_dists(train: pd.DataFrame, test: pd.DataFrame, dataset_name: str):
    """Explore:
    -distribution of training lengths
    -distribution of proportion missing observations
    -distribution of proportion 0 observations
    -distribution of prediction lengths
    """
    group_cols = [train.columns[c] for c in grouping_cols[dataset_name]]

    (
        series_lens,
        prop_obs,
        prop_0_obs,
        pred_lens,
        all_train_dfs,
        all_test_dfs,
    ) = train_test_dists(train, test, group_cols, dataset_name)

    plot_hist(series_lens, "Distribution of Training Series Lengths")
    plot_hist(prop_obs, "Distribution of Proportion of Missing Values")
    plot_hist(
        prop_0_obs,
        "Distribution of Proportion of 0 Values (of non-missing obserations)",
    )
    plot_hist(pred_lens, "Distribution of Prediction Lengths")

    return all_train_dfs, all_test_dfs


def grouping_dists(train: pd.DataFrame, dataset_name: str):
    """Explore:
    -distribution of training lengths (at lower grouping levels)
    """

    group_cols = [train.columns[c] for c in grouping_cols[dataset_name]]

    for i in range(len(group_cols) - 1):
        num_series = train.groupby(group_cols[: i + 1])["d3mIndex"].agg("count")
        plot_hist(
            num_series,
            f"Distribution of Training Series Lengths, Grouping Level: {i+1}",
        )


def plot_ts(time_vals: pd.Series, title: str):
    plt.clf()
    plt.scatter(time_vals.index, time_vals.values)
    plt.title(title)
    plt.xticks(rotation=45, fontsize="x-small")
    plt.subplots_adjust(bottom=0.2)
    plt.savefig(f"ts_explore/{title}")


def cluster_examples(
    train: pd.DataFrame,
    test: pd.DataFrame,
    dataset_name: str,
    nclusters: int = 5,
    n_examples: int = 3,
):
    """Explore:
    -cluster series and look at examples from each cluster
    """
    clusterer = TimeSeriesKMeans(n_clusters=nclusters)

    group_cols = [train.columns[c] for c in grouping_cols[dataset_name]]
    train_groups = train.groupby(group_cols)
    test_groups = test.groupby(group_cols)
    max_l = max(
        [len(trg) + len(teg) for (_, trg), (_, teg) in zip(train_groups, test_groups)]
    )

    timeseries = []
    keys = []
    for (group_name, train_group), (_, test_group) in zip(train_groups, test_groups):
        t_values = train_group.iloc[:, target_cols[dataset_name]].astype(float)
        t_values = t_values.append(
            test_group.iloc[:, target_cols[dataset_name]].astype(float)
        )
        t_padded = t_values.append(
            pd.Series([np.nan] * (max_l - t_values.shape[0])),
        )
        t_padded = t_padded.interpolate()
        assert len(t_padded) == max_l
        timeseries.append(t_padded)
        keys.append(group_name)

    timeseries_dataset = ts_utils.to_time_series_dataset(timeseries)
    clusters = clusterer.fit_predict(timeseries_dataset)

    plot_hist(clusters, "Distribution of Clusters")

    for i in range(nclusters):
        print(f"Looking at examples from cluster {i}")
        idxs = np.where(clusters == i)[0]
        examples = np.random.choice(idxs, size=n_examples, replace=False)
        for j, ex in enumerate(examples):
            query_list = [
                f'{grp_col}=="{key}"' for grp_col, key in zip(group_cols, keys[ex])
            ]
            values = train.query(" & ".join(query_list)).iloc[
                :, target_cols[dataset_name]
            ]
            # values = values.append(
            #     test.query(' & '.join(query_list)).iloc[:, target_cols[dataset_name]]
            # )
            plot_ts(values, f"Example {j} of cluster {i}")


def main(args):
    dataset_name = args.dataset_name
    train, test = read_data(dataset_name)
    df_summary_stats(train, test, dataset_name)
    all_train_dfs, all_test_dfs = series_dists(train, test, dataset_name)
    grouping_dists(train, dataset_name)
    cluster_examples(all_train_dfs, all_test_dfs, dataset_name)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "-d", "--dataset-name", help="The name of the dataset to explore"
    )
    args = parser.parse_args()
    main(args)

"""
Observations
    LL1_terra_canopy_height_long_form_s4_70_MIN_METADATA
        -715 series
        -20-25 length
        -lots missing
        -no 0s
        -pred length 70-100
        -10-25 series per grouping
        -c0 = very small growth, huge by May
        -c2 = big growth, asymptote by May
    LL1_terra_canopy_height_long_form_s4_80_MIN_METADATA
        -65 length
        -pred length 40-50
        -10-35 series per grouping
        -c0:4 = more intermediate observations
    LL1_terra_canopy_height_long_form_s4_90_MIN_METADATA
        -70 length
    LL1_terra_canopy_height_long_form_s4_100_MIN_METADATA
        -70-80 length
        -pred length 30-40
    LL1_terra_leaf_angle_mean_long_form_s4_MIN_METADATA
        -751 serixes
        -20-60 length
        -pred length 20-80
        -c0:2 = no pattern, high growth in future
        -c3:4 = some pattern of upward growth in train
    LL1_PHEM_Monthly_Malnutrition_MIN_METADATA
        -1242 series
        -20s length
        -pred length 5-6
        -some 0 values
        -20s series/grouping level 3
        -hard to see discernable pattern in clusters, maybe seasonal??, 
        definitely spiky
    LL1_PHEM_weeklyData_malnutrition_MIN_METADATA
        -100s length (more densely sampled)
        -slightly more missing values (could be from reindexing?)
        -pred length 20s
"""