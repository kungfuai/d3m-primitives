import sys
import os.path
import typing
from typing import List

import numpy as np
import pandas
import hdbscan
from sklearn.cluster import DBSCAN
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params

from ..utils.dataframe_utils import select_rows

__author__ = "Distil"
__version__ = "1.0.2"
__contact__ = "mailto:cbethune@uncharted.software"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(
        default="HDBSCAN",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["DBSCAN", "HDBSCAN"],
        description="type of clustering algorithm to use",
    )
    eps = hyperparams.Uniform(
        lower=0,
        upper=sys.maxsize,
        default=0.5,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="maximum distance between two samples for them to be considered as in \
        the same neigborhood, used in DBSCAN algorithm",
    )
    min_cluster_size = hyperparams.UniformInt(
        lower=2,
        upper=sys.maxsize,
        default=5,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="the minimum size of clusters",
    )
    min_samples = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=5,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="The number of samples in a neighbourhood for a point to be considered a core point.",
    )
    cluster_selection_method = hyperparams.Enumeration(
        default="eom",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        values=["leaf", "eom"],
        description="Determines how clusters are selected from the cluster hierarchy tree for HDBSCAN",
    )
    required_output = hyperparams.Enumeration(
        default="feature",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["prediction", "feature"],
        description="Determines whether the output is a dataframe with just predictions,\
            or an additional feature added to the input dataframe.",
    )


class HdbscanPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    This primitive applies hierarchical density-based and density-based clustering algorithms.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "ca014488-6004-4b54-9403-5920fbe5a834",
            "version": __version__,
            "name": "hdbscan",
            "keywords": ["Clustering"],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": [
                    "https://github.com/kungfuai/d3m-primitives",
                ],
            },
            "installation": [
                {"type": "PIP", "package": "cython", "version": "0.29.16"},
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/kungfuai/d3m-primitives.git@{git_commit}#egg=kf-d3m-primitives".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            "python_path": "d3m.primitives.clustering.hdbscan.Hdbscan",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.DBSCAN,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.CLUSTERING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        if self.hyperparams["algorithm"] == "HDBSCAN":
            self.clf = hdbscan.HDBSCAN(
                min_cluster_size=self.hyperparams["min_cluster_size"],
                min_samples=self.hyperparams["min_samples"],
                cluster_selection_method=self.hyperparams["cluster_selection_method"],
            )
        else:
            self.clf = DBSCAN(
                eps=self.hyperparams["eps"], min_samples=self.hyperparams["min_samples"]
            )

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs: D3M dataframe with attached metadata for semi-supervised or unsupervised data

        Returns
        ----------
        Outputs:
            The output depends on the required_output hyperparameter and is either a dataframe
            containing a single column where each entry is the cluster ID, or the input daatframe
            with the cluster ID of each row added as an additional feature.
        """

        # find target and index variables
        targets = inputs.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/TrueTarget"
        )
        if not len(targets):
            targets = inputs.metadata.get_columns_with_semantic_type(
                "https://metadata.datadrivendiscovery.org/types/Target"
            )
        if not len(targets):
            targets = inputs.metadata.get_columns_with_semantic_type(
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget"
            )
        target_names = [list(inputs)[t] for t in targets]
        index = inputs.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
        )
        index_names = [list(inputs)[i] for i in index]

        X_test = inputs.copy()
        if len(index):
            X_test = X_test.drop(columns=list(inputs)[index[0]])
        if len(target_names):
            X_test = X_test.drop(columns=target_names)
        X_test = X_test.values

        # special semi-supervised case - during training, only produce rows with labels
        series = inputs[target_names] != ""
        if series.any().any():
            inputs = select_rows(inputs, np.flatnonzero(series))
            X_test = X_test[np.flatnonzero(series)]

        if self.hyperparams["required_output"] == "feature":

            hdb_df = d3m_DataFrame(
                pandas.DataFrame(
                    self.clf.fit_predict(X_test), columns=["cluster_labels"]
                )
            )

            # just add last column of last column ('clusters')
            col_dict = dict(hdb_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
            col_dict["structural_type"] = type(1)
            col_dict["name"] = "cluster_labels"
            col_dict["semantic_types"] = (
                "http://schema.org/Integer",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
            hdb_df.metadata = hdb_df.metadata.update(
                (metadata_base.ALL_ELEMENTS, 0), col_dict
            )

            df_dict = dict(hdb_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
            df_dict_1 = dict(hdb_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
            df_dict["dimension"] = df_dict_1
            df_dict_1["name"] = "columns"
            df_dict_1["semantic_types"] = (
                "https://metadata.datadrivendiscovery.org/types/TabularColumn",
            )
            df_dict_1["length"] = 1
            hdb_df.metadata = hdb_df.metadata.update(
                (metadata_base.ALL_ELEMENTS,), df_dict
            )

            return CallResult(inputs.append_columns(hdb_df))
        else:

            hdb_df = d3m_DataFrame(
                pandas.DataFrame(
                    self.clf.fit_predict(X_test), columns=[target_names[0]]
                )
            )

            hdb_df = pandas.concat([inputs.d3mIndex, hdb_df], axis=1)

            col_dict = dict(hdb_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
            col_dict["structural_type"] = type(1)
            col_dict["name"] = index_names[0]
            col_dict["semantic_types"] = (
                "http://schema.org/Integer",
                "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
            )
            hdb_df.metadata = hdb_df.metadata.update(
                (metadata_base.ALL_ELEMENTS, 0), col_dict
            )

            col_dict = dict(hdb_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
            col_dict["structural_type"] = type(1)
            col_dict["name"] = target_names[0]
            col_dict["semantic_types"] = (
                "http://schema.org/Integer",
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
            )
            hdb_df.metadata = hdb_df.metadata.update(
                (metadata_base.ALL_ELEMENTS, 1), col_dict
            )

            df_dict = dict(hdb_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
            df_dict_1 = dict(hdb_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
            df_dict["dimension"] = df_dict_1
            df_dict_1["name"] = "columns"
            df_dict_1["semantic_types"] = (
                "https://metadata.datadrivendiscovery.org/types/TabularColumn",
            )
            df_dict_1["length"] = 2
            hdb_df.metadata = hdb_df.metadata.update(
                (metadata_base.ALL_ELEMENTS,), df_dict
            )

            return CallResult(hdb_df)
