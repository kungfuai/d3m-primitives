import os.path
import numpy as np
import pandas as pd
import typing
from typing import List
import sys

from nk_sent2vec import Sent2Vec as _Sent2Vec

from d3m import container, utils
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult

from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = "Distil"
__version__ = "1.3.0"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass

class Sent2VecPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
        Produce numerical representations (features) for short texts or sentences.

        Parameters
        ----------
        inputs : Input pandas dataframe

        Returns
        -------
        Outputs
            The output is a pandas dataframe
        """

    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "cf450079-9333-4a3f-aed4-b77a4e8c7be7",
            "version": __version__,
            "name": "sent2vec_wrapper",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": ["Sent2Vec", "Embedding", "NLP", "Natural Language Processing"],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": [
                    # Unstructured URIs.
                    "https://github.com/kungfuai/d3m-primitives"
                ],
            },
            # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
            # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
            # install a Python package first to be even able to run setup.py of another package. Or you have
            # a dependency which is not on PyPi.
            "installation": [
                {"type": "PIP", "package": "cython", "version": "0.29.16"}, 
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/kungfuai/d3m-primitives.git@{git_commit}#egg=kf-d3m-primitives".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
                {
                    "type": "FILE",
                    "key": "sent2vec_model",
                    "file_uri": "http://public.datadrivendiscovery.org/twitter_bigrams.bin",
                    "file_digest": "9e8ccfea2aaa4435ca61b05b11b60e1a096648d56fff76df984709339f423dd6",
                },
            ],
            # The same path the primitive is registered with entry points in setup.py.
            "python_path": "d3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [metadata_base.PrimitiveAlgorithmType.VECTORIZATION],
            "primitive_family": metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
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

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Produce numerical representations (features) for short texts or sentences.

        Parameters
        ----------
        inputs : Input pandas dataframe

        Returns
        -------
        Outputs
            The output is a pandas dataframe
        """

        # extract sentences from stored in nested media files
        text_columns = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/FileName')
        base_paths = [inputs.metadata.query((metadata_base.ALL_ELEMENTS, t))['location_base_uris'][0].replace('file:///', '/') for t in text_columns]
        txt_paths = [[os.path.join(base_path, filename) for filename in inputs.iloc[:,col]] for base_path, col in zip(base_paths, text_columns)]
        txt = [[open(path, 'r').read().replace('\n', '') for path in path_list] for path_list in txt_paths]
        txt_df = pd.DataFrame(np.array(txt).T)

        # concatenate with text columns that aren't stored in nested files
        local_text_columns = inputs.metadata.get_columns_with_semantic_type('http://schema.org/Text')
        local_text_columns = [col for col in local_text_columns if col not in text_columns]
        frame = pd.concat((txt_df, inputs[local_text_columns]), axis=1)
        
        # delete columns with path names of nested media files
        outputs = inputs.remove_columns(text_columns)

        try:
            vectorizer = _Sent2Vec(path=self.volumes["sent2vec_model"])
            #print('loaded sent2vec model', file = sys.__stdout__)
            output_vectors = []
            for col in range(frame.shape[1]):
                text = frame.iloc[:, col].tolist()
                embedded_sentences = vectorizer.embed_sentences(sentences=text)
                output_vectors.append(embedded_sentences)
            embedded_df = pd.DataFrame(np.array(output_vectors).reshape(len(embedded_sentences), -1))
        except ValueError:
            # just return inputs with file names deleted if vectorizing fails
            return CallResult(outputs) 
        
        #print('successfully vectorized text\n', file = sys.__stdout__)

        # create df with vectorized columns and append to input df
        embedded_df = d3m_DataFrame(embedded_df)
        for col in range(embedded_df.shape[1]):
            col_dict = dict(embedded_df.metadata.query((metadata_base.ALL_ELEMENTS, col)))
            col_dict['structural_type'] = type(1.0)
            col_dict['name'] = "vector_" + str(col)
            col_dict["semantic_types"] = (
                    "http://schema.org/Float",
                    "https://metadata.datadrivendiscovery.org/types/Attribute",
                )
            embedded_df.metadata = embedded_df.metadata.update(
                    (metadata_base.ALL_ELEMENTS, col), col_dict
                )
        df_dict = dict(embedded_df.metadata.query((metadata_base.ALL_ELEMENTS, )))
        df_dict_1 = dict(embedded_df.metadata.query((metadata_base.ALL_ELEMENTS, ))) 
        df_dict['dimension'] = df_dict_1
        df_dict_1['name'] = 'columns'
        df_dict_1['semantic_types'] = ('https://metadata.datadrivendiscovery.org/types/TabularColumn',)
        df_dict_1['length'] = embedded_df.shape[1]
        embedded_df.metadata = embedded_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)
        return CallResult(outputs.append_columns(embedded_df))

