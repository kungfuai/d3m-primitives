import sys
import os.path
import typing
from typing import List

import numpy as np
import pandas
from sklearn.cluster import SpectralClustering as SC
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params

from ..utils.dataframe_utils import select_rows

__author__ = "Distil"
__version__ = "1.0.0"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    n_clusters = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=8,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="The dimension of the projection space",
    )

    n_init = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=10,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="Number of times the k-means algorithm will be run with different centroid seeds",
    )

    n_neighbors = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=10,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description='Number of neighbors when constructing the affintiy matrix using n-neighbors, \
            ignored for affinity="rbf"',
    )

    affinity = hyperparams.Enumeration(
        default="rbf",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        values=["rbf", "nearest_neighbors"],
        description="method to construct affinity matrix",
    )

    task_type = hyperparams.Enumeration(
        default="classification",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["clustering", "classification"],
        description="Determines whether the output is a dataframe with just predictions,\
            or an additional feature added to the input dataframe.",
    )


class SpectralClusteringPrimitive(
    TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]
):
    """
    This primitive applies scikit-learn's spectral clustering algorithm to unsupervised, supervised
    or semi-supervised datasets.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "d13a4529-f0ba-44ee-a867-e0fdbb71d6e2",
            "version": __version__,
            "name": "tsne",
            "keywords": ["Clustering", "Graph Clustering"],
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
            "python_path": "d3m.primitives.clustering.spectral_graph.SpectralClustering",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.SPECTRAL_CLUSTERING,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.CLUSTERING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self.sc = SC(
            n_clusters=self.hyperparams["n_clusters"],
            n_init=self.hyperparams["n_init"],
            n_neighbors=self.hyperparams["n_neighbors"],
            affinity=self.hyperparams["affinity"],
            random_state=self.random_seed,
        )

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        Inputs: D3M dataframe with features and labels, and D3M indices

        Returns
        ----------
        Outputs: D3M dataframe with cluster predictions and D3M indices. Clusterlabels are 
            of "suggestTarget" semantic type if the task_type hyperparameter is clustering, 
            and "Attribute" if the task_type is classification.
        """

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

        X_test = inputs.drop(columns=list(inputs)[index[0]])
        X_test = X_test.drop(columns=target_names).values

        # special semi-supervised case - during training, only produce rows with labels
        series = inputs[target_names] != ""
        if series.any().any():
            inputs = select_rows(inputs, np.flatnonzero(series))
            X_test = X_test[np.flatnonzero(series)]

        sc_df = d3m_DataFrame(
            pandas.DataFrame(self.sc.fit_predict(X_test), columns=["cluster_labels"])
        )

        # just add last column of last column ('clusters')
        col_dict = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict["structural_type"] = type(1)
        if self.hyperparams["task_type"] == "classification":
            col_dict["semantic_types"] = (
                "http://schema.org/Integer",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
            col_dict["name"] = "cluster_labels"
        else:
            col_dict["semantic_types"] = (
                "http://schema.org/Integer",
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
            )
            col_dict["name"] = target_names[0]
        sc_df.metadata = sc_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 0), col_dict
        )

        df_dict = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict_1 = dict(sc_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict["dimension"] = df_dict_1
        df_dict_1["name"] = "columns"
        df_dict_1["semantic_types"] = (
            "https://metadata.datadrivendiscovery.org/types/TabularColumn",
        )
        df_dict_1["length"] = 1
        sc_df.metadata = sc_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)

        return CallResult(inputs.append_columns(sc_df))
