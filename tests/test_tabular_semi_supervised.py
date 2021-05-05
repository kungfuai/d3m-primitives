import random

from sklearn.datasets import make_moons
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from d3m.container import DataFrame as d3m_DataFrame

from kf_d3m_primitives.semi_supervised.tabular_semi_supervised.tabular_semi_supervised import (
    TabularSemiSupervisedPrimitive,
    Hyperparams as tss_hp,
)
from kf_d3m_primitives.semi_supervised.tabular_semi_supervised.tabular_semi_supervised_pipeline import (
    TabularSemiSupervisedPipeline,
)

np.random.seed(5)
torch.manual_seed(5 + 111)
torch.cuda.manual_seed(5 + 222)
random.seed(5 + 333)


def load_moons(labeled_sample=10):

    data, label = make_moons(1000, False, 0.1, random_state=5)

    if labeled_sample == 0:
        return data, label

    else:
        l0_data = np.random.permutation(data[(label == 0)])
        l1_data = np.random.permutation(data[(label == 1)])

        X_l = np.concatenate(
            [l0_data[: labeled_sample // 2], l1_data[: labeled_sample // 2]]
        )
        y_l = np.concatenate(
            [np.zeros(labeled_sample // 2), np.ones(labeled_sample // 2)]
        )
        X_u = np.concatenate(
            [l0_data[labeled_sample // 2 :], l1_data[labeled_sample // 2 :]]
        )
        y_u = np.concatenate([np.zeros(X_u.shape[0] // 2), np.ones(X_u.shape[0] // 2)])

        return X_l, y_l, X_u, y_u


def test_moons(labeled_sample=10):

    X_l, y_l, X_u, y_u = load_moons(labeled_sample)

    X = np.vstack((X_l, X_u)).astype(str)
    y = np.concatenate((y_l, y_u)).astype(str)
    y[labeled_sample:] = ""

    features_df = pd.DataFrame(X)
    labels_df = pd.DataFrame({"target": y})

    features_df = d3m_DataFrame(features_df)
    labels_df = d3m_DataFrame(labels_df)

    global tss_params
    tss_params = {}

    accs = {}
    for algorithm in ["PseudoLabel", "VAT", "ICT"]:
        tss = TabularSemiSupervisedPrimitive(
            hyperparams=tss_hp(
                tss_hp.defaults(),
                epochs=50,
                algorithm=algorithm,
                weights_filepath=f"{algorithm}.pth",
            ),
            random_seed=5,
        )
        tss.set_training_data(inputs=features_df, outputs=labels_df)
        tss.fit()

        tss_params[algorithm] = tss.get_params()

        preds = tss.produce(inputs=features_df).value
        acc = (y_u == preds["target"][labeled_sample:].astype(float)).mean()
        print(f"{algorithm}: {acc}")
        accs[algorithm] = acc

    assert accs["VAT"] > accs["PseudoLabel"]
    assert accs["VAT"] > accs["ICT"]
    assert accs["PseudoLabel"] > accs["ICT"]


def test_new_moons():

    X, y = load_moons(labeled_sample=0)

    features_df = pd.DataFrame(X)
    features_df = d3m_DataFrame(features_df)

    accs = {}
    for algorithm in ["PseudoLabel", "VAT", "ICT"]:
        tss = TabularSemiSupervisedPrimitive(
            hyperparams=tss_hp(
                tss_hp.defaults(),
                algorithm=algorithm,
                weights_filepath=f"{algorithm}.pth",
            ),
            random_seed=5,
        )
        tss.set_params(params=tss_params[algorithm])

        preds = tss.produce(inputs=features_df).value
        acc = (y == preds["target"].astype(float)).mean()
        print(f"{algorithm}: {acc}")
        accs[algorithm] = acc

    assert accs["VAT"] > accs["PseudoLabel"]
    assert accs["VAT"] > accs["ICT"]
    assert accs["PseudoLabel"] > accs["ICT"]


def _test_serialize(dataset, algorithm="PseudoLabel"):

    pipeline = TabularSemiSupervisedPipeline(algorithm=algorithm)
    pipeline.write_pipeline()
    pipeline.fit_serialize(dataset)
    pipeline.deserialize_score(dataset)
    pipeline.delete_pipeline()
    pipeline.delete_serialized_pipeline()


def test_serialization_dataset_sylva_prior_pseudolabel():
    _test_serialize("SEMI_1040_sylva_prior_MIN_METADATA", algorithm="PseudoLabel")


def test_serialization_dataset_sylva_prior_vat():
    _test_serialize("SEMI_1040_sylva_prior_MIN_METADATA", algorithm="VAT")


def test_serialization_dataset_sylva_prior_ict():
    _test_serialize("SEMI_1040_sylva_prior_MIN_METADATA", algorithm="ICT")


def test_serialization_dataset_eye_movements_pseudolabel():
    _test_serialize("SEMI_1044_eye_movements_MIN_METADATA", algorithm="PseudoLabel")


def test_serialization_dataset_eye_movements_vat():
    _test_serialize("SEMI_1044_eye_movements_MIN_METADATA", algorithm="VAT")


def test_serialization_dataset_eye_movements_ict():
    _test_serialize("SEMI_1044_eye_movements_MIN_METADATA", algorithm="ICT")


def test_serialization_dataset_software_defects_pseudolabel():
    _test_serialize("SEMI_1053_jm1_MIN_METADATA", algorithm="PseudoLabel")


def test_serialization_dataset_software_defects_vat():
    _test_serialize("SEMI_1053_jm1_MIN_METADATA", algorithm="VAT")


def test_serialization_dataset_software_defects_ict():
    _test_serialize("SEMI_1053_jm1_MIN_METADATA", algorithm="ICT")


def test_serialization_dataset_click_prediction_pseudolabel():
    _test_serialize(
        "SEMI_1217_click_prediction_small_MIN_METADATA", algorithm="PseudoLabel"
    )


def test_serialization_dataset_click_prediction_vat():
    _test_serialize("SEMI_1217_click_prediction_small_MIN_METADATA", algorithm="VAT")


def test_serialization_dataset_click_prediction_ict():
    _test_serialize("SEMI_1217_click_prediction_small_MIN_METADATA", algorithm="ICT")


def test_serialization_dataset_artificial_characters_pseudolabel():
    _test_serialize(
        "SEMI_1459_artificial_characters_MIN_METADATA", algorithm="PseudoLabel"
    )


def test_serialization_dataset_artificial_characters_vat():
    _test_serialize("SEMI_1459_artificial_characters_MIN_METADATA", algorithm="VAT")


def test_serialization_dataset_artificial_characters_ict():
    _test_serialize("SEMI_1459_artificial_characters_MIN_METADATA", algorithm="ICT")
