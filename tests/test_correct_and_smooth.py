from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pandas as pd
import numpy as np
from d3m.container import DataFrame as d3m_DataFrame

from kf_d3m_primitives.semi_supervised.correct_and_smooth.correct_and_smooth import (
    CorrectAndSmoothPrimitive,
    Hyperparams as cas_hp,
)


def load_mnist_ss(start_idx=0):

    X, y = fetch_openml("mnist_784", version=1, return_X_y=True, as_frame=False)
    # np.save("X.npy", X)
    # np.save("y.npy", y)
    # X = np.load("X.npy", allow_pickle=True)
    # y = np.load("y.npy", allow_pickle=True)

    X = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))

    # shuffle
    p = np.random.permutation(X.shape[0])
    X, y = X[p], y[p]

    # subset
    n = 5000
    X, y = X[start_idx : start_idx + n], y[start_idx : start_idx + n]
    return X, y


def split_mnist(X, y):
    """ prepare mnist data as d3m dataframe, fit linear SVC as baseline"""

    # train/test split
    n_class = len(set(y))
    n_samples_per_class = 2

    idxs = np.arange(X.shape[0])
    X_train, X_test, y_train, y_test, _, idx_test = train_test_split(
        X, y, idxs, train_size=n_class * n_samples_per_class, stratify=y
    )
    y[idx_test] = ""

    # Linear SVC
    global svc
    svc = LinearSVC().fit(X_train, y_train)
    svc_preds = svc.predict(X_test)
    svc_acc = (y_test == svc_preds).mean()

    features_df = pd.DataFrame(X)
    labels_df = pd.DataFrame({"target": y})

    features_df = d3m_DataFrame(features_df)
    labels_df = d3m_DataFrame(labels_df)

    return features_df, labels_df, svc_acc, idx_test, y_test


def test_mnist():

    X, y = load_mnist_ss()

    features_df, labels_df, svc_acc, idx_test, y_test = split_mnist(X, y)

    cas = CorrectAndSmoothPrimitive(
        hyperparams=cas_hp(
            cas_hp.defaults(),
        )
    )
    cas.set_training_data(inputs=features_df, outputs=labels_df)
    cas.fit()

    global cas_params
    cas_params = cas.get_params()

    preds = cas.produce(inputs=features_df).value
    cas_acc = (y_test == preds["target"][idx_test]).mean()

    print(f"svc acc: {svc_acc}")
    print(f"cas acc: {cas_acc}")
    assert cas_acc > svc_acc


def test_produce_mnist():

    X, y = load_mnist_ss(start_idx=5000)

    features_df = pd.DataFrame(X)

    cas = CorrectAndSmoothPrimitive(
        hyperparams=cas_hp(
            cas_hp.defaults(),
        )
    )
    cas.set_params(params=cas_params)

    preds = cas.produce(inputs=features_df).value
    cas_acc = (y == preds["target"]).mean()

    svc_preds = svc.predict(X)
    svc_acc = (y == svc_preds).mean()

    print(f"svc acc: {svc_acc}")
    print(f"cas acc: {cas_acc}")
    assert cas_acc > svc_acc


def _test_serialize(dataset, normalize_features=False):

    pipeline = CorrectAndSmoothPipeline(normalize_features=normalize_features)
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()


def test_serialization_dataset_sylva_prior():
    # 0.67 vs. 0.96
    _test_serialize("SEMI_1040_sylva_prior_MIN_METADATA", normalize_features=True)


def test_serialization_dataset_eye_movements():
    # 0.48 vs. 0.42
    _test_serialize("SEMI_1044_eye_movements_MIN_METADATA", normalize_features=True)


def test_serialization_dataset_software_defects():
    # 0.34 vs. 0.55
    _test_serialize("SEMI_1053_jm1_MIN_METADATA")


def test_serialization_dataset_click_prediction():
    # 0.09 vs. 0.52
    _test_serialize(
        "SEMI_1217_click_prediction_small_MIN_METADATA", normalize_features=True
    )


def test_serialization_dataset_artificial_characters():
    # 0.45 vs. 0.61
    _test_serialize(
        "SEMI_1459_artificial_characters_MIN_METADATA", normalize_features=True
    )
