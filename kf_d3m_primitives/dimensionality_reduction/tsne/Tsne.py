import sys
import os.path
import typing
from typing import List

import numpy as np
import pandas
from sklearn.manifold import TSNE
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = "Distil"
__version__ = "1.0.0"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"


Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    n_components = hyperparams.UniformInt(
        lower=1,
        upper=3,
        upper_inclusive=True,
        default=2,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="dimension of the embedded space",
    )


class TsnePrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    This primitive applies scikit-learn's T-distributed stochastic neighbour embedding algorithm.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "15586787-80d5-423e-b232-b61f55a117ce",
            "version": __version__,
            "name": "tsne",
            "keywords": ["Dimensionality Reduction"],
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
            "python_path": "d3m.primitives.dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.T_DISTRIBUTED_STOCHASTIC_NEIGHBOR_EMBEDDING,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DIMENSIONALITY_REDUCTION,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self.clf = TSNE(
            n_components=self.hyperparams["n_components"], random_state=self.random_seed
        )

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs : D3M dataframe with attached metadata for semi-supervised or unsupervised data

        Returns
        ----------
        Outputs:
            D3M dataframe with t-SNE dimensions and D3M indices
        """

        # store information on target, index variable
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

        n_ts = len(inputs.d3mIndex.unique())
        if n_ts == inputs.shape[0]:
            X_test = inputs.drop(columns=list(inputs)[index[0]])
            X_test = X_test.drop(columns=target_names).values
        else:
            ts_sz = int(inputs.shape[0] / n_ts)
            X_test = np.array(inputs.value).reshape(n_ts, ts_sz)

        # fit_transform data and create new dataframe
        n_components = self.hyperparams["n_components"]
        col_names = ["Dim" + str(c) for c in range(0, n_components)]

        tsne_df = d3m_DataFrame(
            pandas.DataFrame(self.clf.fit_transform(X_test), columns=col_names)
        )

        tsne_df = pandas.concat([inputs.d3mIndex, tsne_df], axis=1)

        # add index colmn metadata
        col_dict = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict["structural_type"] = type("1")
        col_dict["name"] = index_names[0]
        col_dict["semantic_types"] = (
            "http://schema.org/Int",
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
        )
        tsne_df.metadata = tsne_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 0), col_dict
        )

        # add dimenion columns metadata
        for c in range(1, n_components + 1):
            col_dict = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS, c)))
            col_dict["structural_type"] = type(1.0)
            col_dict["name"] = "Dim" + str(c - 1)
            col_dict["semantic_types"] = (
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
            tsne_df.metadata = tsne_df.metadata.update(
                (metadata_base.ALL_ELEMENTS, c), col_dict
            )

        df_dict = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict_1 = dict(tsne_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict["dimension"] = df_dict_1
        df_dict_1["name"] = "columns"
        df_dict_1["semantic_types"] = (
            "https://metadata.datadrivendiscovery.org/types/TabularColumn",
        )
        df_dict_1["length"] = n_components + 1
        tsne_df.metadata = tsne_df.metadata.update(
            (metadata_base.ALL_ELEMENTS,), df_dict
        )

        return CallResult(tsne_df)
