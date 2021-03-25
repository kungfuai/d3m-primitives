"""
Script to generate plot of DeepAR D3M forecast
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set(style="darkgrid")


def prep_frame(train_csv: str, test_csv: str, label: str = "prediction"):
    train = pd.read_csv(train_csv)
    test = pd.read_csv(test_csv)
    data = pd.concat((train, test)).reset_index(drop=True)
    data["label"] = label
    if label == "prediction":
        return data
    else:
        return data, test["year-month"][0]


def reshape_frame(data: pd.DataFrame):
    intervals = []
    for quantile in ["0.1", "0.5", "0.9"]:
        df = data[[quantile, "label", "year-month"]]
        df = df.rename(columns={quantile: "sunspots"})
        intervals.append(df)
    return pd.concat(intervals)


def split(data: pd.DataFrame, time_start: str):
    return data[data["year-month"] >= time_start]


def plot(
    data: pd.DataFrame, train_test_cut: str, x_label_sample: int = 12, save: bool = True
):
    fig, ax = plt.subplots()
    chart = sns.lineplot(x="year-month", y="sunspots", data=data, hue="label")
    for ind, label in enumerate(chart.get_xticklabels()):
        if ind % x_label_sample == 0:
            label.set_visible(True)
        else:
            label.set_visible(False)

    plt.xticks(rotation=45, fontsize="x-small")
    plt.subplots_adjust(bottom=0.2)
    plt.xlabel("Date")
    plt.ylabel("Sunspots")
    plt.title("DeepAR Forecast on Monthly Sunspots Problem")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles=handles[1:], labels=labels[1:])
    ax.axvline(x=train_test_cut, color="g")
    if save:
        plt.savefig("sunspots.png")
    else:
        plt.show()


trues, train_test_cut = prep_frame("train_trues.csv", "test_trues.csv", "true")
preds = prep_frame("train_preds.csv", "test_preds.csv", "prediction")
preds["year-month"] = trues["year-month"]
preds = reshape_frame(preds)
preds = preds.rename(columns={"0.5": "sunspots"})
data = pd.concat((trues, preds))
data = split(data, "2012-01")
plot(data, train_test_cut)
