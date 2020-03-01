import os.path
import numpy as np
import pandas
import pickle
import requests
import ast
import typing
import pkg_resources
from json import JSONDecoder
from typing import List

from Duke.agg_functions import *
from Duke.dataset_descriptor import DatasetDescriptor
from Duke.utils import mean_of_rows

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base
from d3m.primitives.data_transformation.dataset_to_dataframe import Common as DatasetToDataFrame

__author__ = 'Distil'
__version__ = '1.2.0'
__contact__ = 'mailto:sanjeev@yonder.co'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    records = hyperparams.Uniform(lower = 0, upper = 1, default = 1, upper_inclusive = True,
    semantic_types = ['https://metadata.datadrivendiscovery.org/types/TuningParameter'],
    description = 'percentage of records to sub-sample from the data frame')
    pass

class DukePrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
        Abstractive tabulat dataset summarization using pre-trained knowledge graph
        embeddings. Uses a word2vec model trained on Wikipedia to assign abstractive summary
        tags to the detaset, as well as confidence values to each tag. Tags come from the
        corresponding Wikipedia subject ontology.
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "46612a42-6120-3559-9db9-3aa9a76eb94f",
        'version': __version__,
        'name': "duke",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Dataset Descriptor','Text', 'NLP','Abstractive Summarization'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/Yonder-OSS/D3M-Primitives",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
         'installation': [{
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/Yonder-OSS/D3M-Primitives.git@{git_commit}#egg=yonder-primitives'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),
            ),
         },
            {
            "type": "TGZ",
            "key": "en.model",
            "file_uri": "http://public.datadrivendiscovery.org/en_1000_no_stem.tar.gz",
            "file_digest":"3b1238137bba14222ae7c718f535c68a3d7190f244296108c895f1abe8549861"
        },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.data_cleaning.text_summarization.Duke',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.RECURRENT_NEURAL_NETWORK,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.DATA_CLEANING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0, volumes: typing.Dict[str,str]=None)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed, volumes=volumes)

        self.volumes = volumes

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce a summary for the tabular dataset input
        Parameters
        ----------
        inputs : Input pandas frame
        Returns
        -------
        Outputs
            The output is a string summary
        """

        """ Accept a pandas data frame, returns a string summary
        frame: a pandas data frame containing the data to be processed
        -> a string summary
        """

        # sub-sample percentage of records from data frame
        records = self.hyperparams['records']
        frame = inputs.sample(frac = records)

        # cast frame data type back to original, if numeric, to ensure
        # that duke can drop them, and not skew results (since d3m
        #  preprocessing prims turn everything into str/object)
        tmp = frame
        for i in range(frame.shape[1]):
            if (frame.metadata.query_column(i)['semantic_types'][0]=='http://schema.org/Integer'):
                tmp.ix[:,frame.columns[i]].replace('',0,inplace=True)
                tmp[frame.columns[i]] = pandas.to_numeric(tmp[frame.columns[i]],errors='coerce')
                # converting a string value like '32.0' to an int directly results in an error, so we first
                # convert everything to a float
                tmp = tmp.astype({frame.columns[i]:float})
                tmp = tmp.astype({frame.columns[i]:int})
            elif (frame.metadata.query_column(i)['semantic_types'][0]=='http://schema.org/Float'):
                tmp.ix[:,frame.columns[i]].replace('',0,inplace=True)
                tmp[frame.columns[i]] = pandas.to_numeric(tmp[frame.columns[i]],errors='coerce')
                tmp = tmp.astype({frame.columns[i]:float})
            # not yet sure if dropping CategoticalData is ideal, but it appears to work...
            # some categorical data may contain useful information, but the d3m transformation is not reversible
            # and not aware of a way to distinguish numerical from non-numerical CategoricalData
            elif (frame.metadata.query_column(i)['semantic_types'][0]=='https://metadata.datadrivendiscovery.org/types/CategoricalData'):
                tmp = tmp.drop(columns=[frame.columns[i]])

        # print('beginning summarization... \n')

        # get the path to the ontology class tree
        resource_package = "Duke"
        resource_path = '/'.join(('ontologies', 'class-tree_dbpedia_2016-10.json'))
        tree_path = pkg_resources.resource_filename(resource_package, resource_path)
        embedding_path = self.volumes['en.model']+"/en_1000_no_stem/en.model"
        row_agg_func=mean_of_rows
        tree_agg_func=parent_children_funcs(np.mean, max)
        source_agg_func=mean_of_rows
        max_num_samples = 1e6
        verbose=True

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

        print('initialized duke dataset descriptor \n')

        N = 5
        out_tuple = duke.get_top_n_words(N)
        print('finished summarization \n')
        out_df_duke = pandas.DataFrame.from_records(list(out_tuple)).T
        out_df_duke.columns = ['subject tags','confidences']


        # initialize the output dataframe as input dataframe (results will be appended to it)
        # out_df = d3m_DataFrame(inputs)

        # create metadata for the duke output dataframe
        duke_df = d3m_DataFrame(out_df_duke)
        # first column ('subject tags')
        col_dict = dict(duke_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("it is a string")
        col_dict['name'] = "subject tags"
        col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        duke_df.metadata = duke_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
         # second column ('confidences')
        col_dict = dict(duke_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1.0")
        col_dict['name'] = "confidences"
        col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        duke_df.metadata = duke_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        # concatenate final output frame -- not real consensus from program, so commenting out for now
        #out_df = utils_cp.append_columns(out_df, duke_df)


        return CallResult(duke_df)