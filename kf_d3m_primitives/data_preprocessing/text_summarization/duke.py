import os.path
import typing
import logging

import numpy as np
import pandas
import pkg_resources
from Duke.agg_functions import *
from Duke.dataset_descriptor import DatasetDescriptor
from Duke.utils import mean_of_rows
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base

__author__ = "Distil"
__version__ = "1.2.0"
__contact__ = "mailto:cbethune@uncharted.software"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

logger = logging.getLogger(__name__)


class Hyperparams(hyperparams.Hyperparams):
    records_fraction = hyperparams.Uniform(
        lower=0,
        upper=1,
        default=1,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="percentage of records to sub-sample from the data frame",
    )


class DukePrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    This primitive produces abstractive summarization tags based on a word2vec model trained
    on Wikipedia data and a corresponding Wikipedia ontology.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "46612a42-6120-3559-9db9-3aa9a76eb94f",
            "version": __version__,
            "name": "duke",
            "keywords": [
                "Dataset Descriptor",
                "Text",
                "NLP",
                "Abstractive Summarization",
            ],
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
                {
                    "type": "TGZ",
                    "key": "en.model",
                    "file_uri": "http://public.datadrivendiscovery.org/en_1000_no_stem.tar.gz",
                    "file_digest": "3b1238137bba14222ae7c718f535c68a3d7190f244296108c895f1abe8549861",
                },
            ],
            "python_path": "d3m.primitives.data_cleaning.text_summarization.Duke",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.RECURRENT_NEURAL_NETWORK,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_CLEANING,
        }
    )

    def __init__(
        self,
        *,
        hyperparams: Hyperparams,
        random_seed: int = 0,
        volumes: typing.Dict[str, str] = None
    ) -> None:
        super().__init__(
            hyperparams=hyperparams, random_seed=random_seed, volumes=volumes
        )

        self.volumes = volumes
        self.random_seed = random_seed

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Produce a summary for the tabular dataset input

        Parameters
        ----------
        inputs: D3M dataframe

        Returns
        ----------
        Outputs: D3M dataframe with two columns: subject tags and confidences.
        """

        # sub-sample percentage of records from data frame
        frame = inputs.sample(
            frac=self.hyperparams["records_fraction"], random_state=self.random_seed
        )

        tmp = frame
        for i in range(frame.shape[1]):
            # not yet sure if dropping CategoticalData is ideal, but it appears to work...
            # some categorical data may contain useful information, but the d3m transformation is not reversible
            # and not aware of a way to distinguish numerical from non-numerical CategoricalData
            if (
                frame.metadata.query_column(i)["semantic_types"][0]
                == "https://metadata.datadrivendiscovery.org/types/CategoricalData"
            ):
                tmp = tmp.drop(columns=[frame.columns[i]])

        logger.info("beginning summarization... \n")

        # get the path to the ontology class tree
        resource_package = "Duke"
        resource_path = "/".join(("ontologies", "class-tree_dbpedia_2016-10.json"))
        tree_path = pkg_resources.resource_filename(resource_package, resource_path)
        embedding_path = self.volumes["en.model"] + "/en_1000_no_stem/en.model"
        row_agg_func = mean_of_rows
        tree_agg_func = parent_children_funcs(np.mean, max)
        source_agg_func = mean_of_rows
        max_num_samples = 1e6
        verbose = True

        duke = DatasetDescriptor(
            dataset=tmp,
            tree=tree_path,
            embedding=embedding_path,
            row_agg_func=row_agg_func,
            tree_agg_func=tree_agg_func,
            source_agg_func=source_agg_func,
            max_num_samples=max_num_samples,
            verbose=verbose,
        )

        logger.info("initialized duke dataset descriptor \n")

        N = 5
        out_tuple = duke.get_top_n_words(N)
        logger.info("finished summarization \n")
        out_df_duke = pandas.DataFrame.from_records(list(out_tuple)).T
        out_df_duke.columns = ["subject tags", "confidences"]

        # create metadata for the duke output dataframe
        duke_df = d3m_DataFrame(out_df_duke)
        # first column ('subject tags')
        col_dict = dict(duke_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict["structural_type"] = type("it is a string")
        col_dict["name"] = "subject tags"
        col_dict["semantic_types"] = (
            "http://schema.org/Text",
            "https://metadata.datadrivendiscovery.org/types/Attribute",
        )
        duke_df.metadata = duke_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 0), col_dict
        )
        # second column ('confidences')
        col_dict = dict(duke_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict["structural_type"] = type("1.0")
        col_dict["name"] = "confidences"
        col_dict["semantic_types"] = (
            "http://schema.org/Float",
            "https://metadata.datadrivendiscovery.org/types/Attribute",
        )
        duke_df.metadata = duke_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 1), col_dict
        )

        return CallResult(duke_df)