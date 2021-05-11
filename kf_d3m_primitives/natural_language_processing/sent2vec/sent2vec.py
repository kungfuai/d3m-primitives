import os.path
from typing import Sequence, Optional, Dict

import numpy as np
import pandas as pd
from nk_sent2vec import Sent2Vec as _Sent2Vec
from d3m import container, utils
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = "Distil"
__version__ = "1.3.0"
__contact__ = "mailto:cbethune@uncharted.software"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="A set of column indices to force primitive to operate on. If any specified \
            column cannot be parsed, it is skipped.",
    )


class Sent2VecPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    This primitive produces numerical representations of text data using a model
    that was pre-trained on English Twitter bi-grams.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "cf450079-9333-4a3f-aed4-b77a4e8c7be7",
            "version": __version__,
            "name": "sent2vec_wrapper",
            "keywords": ["Sent2Vec", "Embedding", "NLP", "Natural Language Processing"],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": ["https://github.com/kungfuai/d3m-primitives"],
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
                    "type": "FILE",
                    "key": "sent2vec_model",
                    "file_uri": "http://public.datadrivendiscovery.org/twitter_bigrams.bin",
                    "file_digest": "9e8ccfea2aaa4435ca61b05b11b60e1a096648d56fff76df984709339f423dd6",
                },
            ],
            "python_path": "d3m.primitives.feature_extraction.nk_sent2vec.Sent2Vec",
            "algorithm_types": [metadata_base.PrimitiveAlgorithmType.VECTORIZATION],
            "primitive_family": metadata_base.PrimitiveFamily.FEATURE_EXTRACTION,
        }
    )

    # class instance to avoid unnecessary re-init on subsequent produce calls
    _vectorizer: Optional[_Sent2Vec] = None

    def __init__(
        self,
        *,
        hyperparams: Hyperparams,
        random_seed: int = 0,
        volumes: Dict[str, str] = None
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
        inputs: D3M dataframe

        Returns
        -------
        Outputs: Input D3M dataframe with vector components appended as additional columns
        """

        # figure out columns to operate on
        cols = self._get_operating_columns(
            inputs, self.hyperparams["use_columns"], ("http://schema.org/Text",)
        )
        frame = inputs.iloc[:, cols]
        outputs = inputs.copy()

        try:
            # lazy load the model and keep it around for subsequent produce calls
            if Sent2VecPrimitive._vectorizer is None:
                Sent2VecPrimitive._vectorizer = _Sent2Vec(
                    path=self.volumes["sent2vec_model"]
                )

            output_vectors = []
            for col in range(frame.shape[1]):
                text = frame.iloc[:, col].tolist()
                embedded_sentences = Sent2VecPrimitive._vectorizer.embed_sentences(
                    sentences=text
                )
                output_vectors.append(embedded_sentences)
            embedded_df = pd.DataFrame(
                np.array(output_vectors).reshape(len(embedded_sentences), -1)
            )
        except ValueError:
            # just return inputs with file names deleted if vectorizing fails
            return CallResult(outputs)

        # create df with vectorized columns and append to input df
        embedded_df = d3m_DataFrame(embedded_df)
        for col in range(embedded_df.shape[1]):
            col_dict = dict(
                embedded_df.metadata.query((metadata_base.ALL_ELEMENTS, col))
            )
            col_dict["structural_type"] = type(1.0)
            col_dict["name"] = "vector_" + str(col)
            col_dict["semantic_types"] = (
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
            embedded_df.metadata = embedded_df.metadata.update(
                (metadata_base.ALL_ELEMENTS, col), col_dict
            )
        df_dict = dict(embedded_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict_1 = dict(embedded_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict["dimension"] = df_dict_1
        df_dict_1["name"] = "columns"
        df_dict_1["semantic_types"] = (
            "https://metadata.datadrivendiscovery.org/types/TabularColumn",
        )
        df_dict_1["length"] = embedded_df.shape[1]
        embedded_df.metadata = embedded_df.metadata.update(
            (metadata_base.ALL_ELEMENTS,), df_dict
        )
        return CallResult(outputs.append_columns(embedded_df))

    @classmethod
    def _get_operating_columns(
        cls,
        inputs: container.DataFrame,
        use_columns: Sequence[int],
        semantic_types: Sequence[str],
        require_attribute: bool = True,
    ) -> Sequence[int]:
        # use caller supplied columns if supplied
        cols = set(use_columns)
        type_cols = set(
            inputs.metadata.list_columns_with_semantic_types(semantic_types)
        )
        if require_attribute:
            attributes = set(
                inputs.metadata.list_columns_with_semantic_types(
                    ("https://metadata.datadrivendiscovery.org/types/Attribute",)
                )
            )
            type_cols = type_cols & attributes

        if len(cols) > 0:
            cols = type_cols & cols
        else:
            cols = type_cols
        return list(cols)