import os
import numpy as np
import pandas as pd
import typing
import sys
from Simon import Simon
from Simon.penny.guesser import guess
from d3m.primitive_interfaces.unsupervised_learning import (
    UnsupervisedLearnerPrimitiveBase,
)
from d3m.primitive_interfaces.base import CallResult
from d3m.exceptions import PrimitiveNotFittedError

from d3m import container, utils
from d3m.base import utils as base_utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
import copy

import tensorflow as tf
import logging

__author__ = "Distil"
__version__ = "1.2.3"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

SIMON_ANNOTATIONS_DICT = {
    "categorical": "https://metadata.datadrivendiscovery.org/types/CategoricalData",
    "email": "http://schema.org/email",
    "text": "http://schema.org/Text",
    "uri": "https://metadata.datadrivendiscovery.org/types/FileName",
    "address": "http://schema.org/address",
    "state": "http://schema.org/State",
    "city": "http://schema.org/City",
    "postal_code": "http://schema.org/postalCode",
    "latitude": "http://schema.org/latitude",
    "longitude": "http://schema.org/longitude",
    "country": "http://schema.org/Country",
    "country_code": "http://schema.org/addressCountry",
    "boolean": "http://schema.org/Boolean",
    "datetime": "http://schema.org/DateTime",
    "float": "http://schema.org/Float",
    "int": "http://schema.org/Integer",
    "phone": "https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber",
    "ordinal": "https://metadata.datadrivendiscovery.org/types/OrdinalData"
}

class Params(params.Params):
    add_semantic_types: typing.Optional[typing.List[typing.List[str]]]
    remove_semantic_types: typing.Optional[typing.List[typing.List[str]]]


class Hyperparams(hyperparams.Hyperparams):
    detect_semantic_types = hyperparams.Set(
        elements=hyperparams.Enumeration(
            values=[
                'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text',
                'http://schema.org/DateTime', 'https://metadata.datadrivendiscovery.org/types/Time',
                "https://metadata.datadrivendiscovery.org/types/OrdinalData",
                "https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber",
                "http://schema.org/addressCountry", "http://schema.org/Country",
                "http://schema.org/longitude", "http://schema.org/latitude",
                "http://schema.org/postalCode", "http://schema.org/City",
                "http://schema.org/State", "http://schema.org/address", "http://schema.org/email", 
                "https://metadata.datadrivendiscovery.org/types/FileName", 
                "https://metadata.datadrivendiscovery.org/types/UniqueKey",
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                'https://metadata.datadrivendiscovery.org/types/UnknownType',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey',
            ],
            # Default is ignored.
            # TODO: Remove default. See: https://gitlab.com/datadrivendiscovery/d3m/issues/141
            default='http://schema.org/Boolean',
        ),
        default=(
                'http://schema.org/Boolean', 'https://metadata.datadrivendiscovery.org/types/CategoricalData',
                'http://schema.org/Integer', 'http://schema.org/Float', 'http://schema.org/Text',
                'http://schema.org/DateTime', 'https://metadata.datadrivendiscovery.org/types/Time',
                "https://metadata.datadrivendiscovery.org/types/OrdinalData",
                "https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber",
                "http://schema.org/addressCountry", "http://schema.org/Country",
                "http://schema.org/longitude", "http://schema.org/latitude",
                "http://schema.org/postalCode", "http://schema.org/City",
                "http://schema.org/State", "http://schema.org/address", "http://schema.org/email", 
                "https://metadata.datadrivendiscovery.org/types/FileName", 
                "https://metadata.datadrivendiscovery.org/types/UniqueKey",
                'https://metadata.datadrivendiscovery.org/types/Attribute',
                'https://metadata.datadrivendiscovery.org/types/TrueTarget',
                'https://metadata.datadrivendiscovery.org/types/UnknownType',
                'https://metadata.datadrivendiscovery.org/types/PrimaryKey',
                'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey'
        ),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of semantic types to detect and set. One can provide a subset of supported semantic types to limit what the primitive detects.",
    )
    remove_unknown_type = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Remove \"https://metadata.datadrivendiscovery.org/types/UnknownType\" semantic type from columns on which the primitive has detected other semantic types.",
    )
    use_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to force primitive to operate on. If any specified column cannot be detected, it is skipped.",
    )
    exclude_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="A set of column indices to not operate on. Applicable only if \"use_columns\" is not provided.",
    )
    return_result = hyperparams.Enumeration(
        values=['append', 'replace', 'new'],
        default='replace',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Should detected columns be appended, should they replace original columns, or should only detected columns be returned?",
    )
    add_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Also include primary index columns if input data has them. Applicable only if \"return_result\" is set to \"new\".",
    )
    replace_index_columns = hyperparams.UniformBool(
        default=True,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description="Replace primary index columns even if otherwise appending columns. Applicable only if \"return_result\" is set to \"append\".",
    )
    overwrite = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to overwrite manual annotations with SIMON annotations. If overwrite is set to False" +
            "only columns with `UnknownType` will be processed, otherwise all columns will be processed",
    )
    statistical_classification = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to infer categorical and ordinal annotations using rule-based classification",
    )
    multi_label_classification = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to perfrom multi-label classification and potentially append multiple annotations to metadata."
    )
    max_rows = hyperparams.UniformInt(
        lower=100,
        upper=2000,
        default=500,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="maximum number of rows from the dataset to process when inferring column semantic types",
    )
    p_threshold = hyperparams.Uniform(
        lower=0,
        upper=1.0,
        default=0.9,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="probability threshold to use when decoding classification results. Semantic types with prediction probabilities above `p_threshold`" +
            "will be added"
    )


class SimonPrimitive(UnsupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """ Simon uses a LSTM-FCN neural network trained on 18 different semantic types to infer the semantic
        type of each column. A hyperparameter `return_result` controls whether Simon's inferences replace existing metadata, 
        append new columns with inferred metadata, or return a new dataframe with only the inferred columns. 

        Simon can append multiple annotations if the hyperparameter `multi_label_classification` is set to 'True'. 
        If `statistical_classification` is set to True, Simon will use rule-based heuristics to label categorical and ordinal columns. 
        Finally, the `p_threshold` hyperparameter varies the prediction probability threshold for adding annotations. 

        The following annotations will only be considered if `statistical_classification` is set to False:
            "https://metadata.datadrivendiscovery.org/types/AmericanPhoneNumber",
            "http://schema.org/addressCountry", "http://schema.org/Country",
            "http://schema.org/longitude", "http://schema.org/latitude",
            "http://schema.org/postalCode", "http://schema.org/City",
            "http://schema.org/State", "http://schema.org/address", "http://schema.org/email", 
            "https://metadata.datadrivendiscovery.org/types/FileName"
        
        The following annotations will only be considered if `statistical_classification` is set to True:
            "https://metadata.datadrivendiscovery.org/types/OrdinalData",

        Arguments:
            hyperparams {Hyperparams} -- D3M Hyperparameter object

        Keyword Arguments:
            random_seed {int} -- random seed (default: {0})
            volumes {Dict[str, str]} -- large file dictionary containing model weights (default: {None})
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "d2fa8df2-6517-3c26-bafc-87b701c4043a",
            "version": __version__,
            "name": "simon",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": [
                "Data Type Predictor",
                "Semantic Classification",
                "Text",
                "NLP",
                "Tabular",
            ],
            "source": {
                "name": __author__,
                "contact": __contact__,
                'uris': [
                    # Unstructured URIs.
                    "https://github.com/kungfuai/d3m-primitives",
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
                    "type": "TGZ",
                    "key": "simon_models_1",
                    "file_uri": "http://public.datadrivendiscovery.org/simon_models_1.tar.gz",
                    "file_digest": "d071106b823ab1168879651811dd03b829ab0728ba7622785bb5d3541496c45f",
                },
            ],
            # The same path the primitive is registered with entry points in setup.py.
            "python_path": "d3m.primitives.data_cleaning.column_type_profiler.Simon",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.CONVOLUTIONAL_NEURAL_NETWORK,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_CLEANING,
        }
    )

    def __init__(
        self,
        *,
        hyperparams: Hyperparams,
        random_seed: int = 0,
        volumes: typing.Dict[str, str] = None,
    ) -> None:

        super().__init__(
            hyperparams=hyperparams, random_seed=random_seed, volumes=volumes
        )
        self._volumes = volumes
        self._X_train: Inputs = None
        self._add_semantic_types: typing.List[typing.List[str]] = None
        self._remove_semantic_types: typing.List[typing.List[str]] = None

    def set_training_data(self, *, inputs: Inputs) -> None:
        """ Sets primitive's training data

            Arguments:
                inputs {Inputs} -- D3M dataframe
        """
        self._X_train = inputs
        self._is_fit = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ Learns column annotations using training data. Saves to apply to testing data.

            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Returns:
                CallResult[None]
        """

        true_target_columns = self._X_train.metadata.list_columns_with_semantic_types(['https://metadata.datadrivendiscovery.org/types/TrueTarget'])
        index_columns = self._X_train.metadata.get_index_columns()

        # Target and index columns should be set only once, if they are set.
        self.has_set_target_columns = False
        self.has_set_index_column = False

        columns_to_use = self._get_columns(self._X_train.metadata)

        self._add_semantic_types = []
        self._remove_semantic_types = []

        # compute SIMON annotations
        self.simon_annotations = self._produce_annotations(inputs=self._X_train)
        logger.debug(f"simon annotations: {self.simon_annotations}")

        for col_idx in columns_to_use:

            # Target and index columns should be set only once, if they are set.
            self.has_set_target_columns = False
            self.has_set_index_column = False

            input_column = self._X_train.select_columns([col_idx])
            column_metadata = self._X_train.metadata.query_column(col_idx)
            column_name = column_metadata.get('name', str(col_idx))
            column_semantic_types = list(column_metadata.get('semantic_types', []))

            # We might be here because column has a known type, but it has "https://metadata.datadrivendiscovery.org/types/SuggestedTarget" set.
            has_unknown_type = not column_semantic_types or 'https://metadata.datadrivendiscovery.org/types/UnknownType' in column_semantic_types

            # A normalized copy of semantic types, which always includes unknown type.
            normalized_column_semantic_types = copy.copy(column_semantic_types)

            # If we are processing this column and it does not have semantic type then it has missing semantic types,
            # we first set it, to normalize the input semantic types. If we will add any other semantic type,
            # we will then remove this semantic type.
            if has_unknown_type \
                    and 'https://metadata.datadrivendiscovery.org/types/UnknownType' in self.hyperparams['detect_semantic_types'] \
                    and 'https://metadata.datadrivendiscovery.org/types/UnknownType' not in normalized_column_semantic_types:
                normalized_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/UnknownType')

            # A working copy of semantic types.
            new_column_semantic_types = copy.copy(normalized_column_semantic_types)

            # append simon labels
            if has_unknown_type:
                new_column_semantic_types = self._append_simon_annotations(new_column_semantic_types, col_idx)

            # handle target columns
            new_column_semantic_types = self._set_target_column(new_column_semantic_types, true_target_columns)

            if has_unknown_type:

                # handle index columns
                if not index_columns and not self.has_set_index_column:
                    new_column_semantic_types = self._set_index_column(new_column_semantic_types, column_name)
            
                # handle attribute columns
                new_column_semantic_types = self._set_attribute_column(new_column_semantic_types)

                # handle additional time label
                new_column_semantic_types = self._set_additional_time_label(new_column_semantic_types)

                # Have we added any other semantic type besides unknown type?
                if new_column_semantic_types != normalized_column_semantic_types:
                    if self.hyperparams['remove_unknown_type'] and 'https://metadata.datadrivendiscovery.org/types/UnknownType' in new_column_semantic_types:
                        new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/UnknownType')
            
            new_column_semantic_types_set = set(new_column_semantic_types)
            column_semantic_types_set = set(column_semantic_types)

            self._add_semantic_types.append(sorted(new_column_semantic_types_set - column_semantic_types_set))
            self._remove_semantic_types.append(sorted(column_semantic_types_set - new_column_semantic_types_set))

        assert len(self._add_semantic_types) == len(columns_to_use)
        assert len(self._remove_semantic_types) == len(columns_to_use)
        self._is_fit = True
        return CallResult(None)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Inputs]:
        """ Add SIMON annotations 

            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target

            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Raises:
                PrimitiveNotFittedError: if primitive not fit

            Returns:
                CallResult[Outputs] -- Input pd frame with metadata augmented 

        """
        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        ## BEGIN originally from from d3m.primitives.schema_discovery.profiler.Common """
        assert self._add_semantic_types is not None
        assert self._remove_semantic_types is not None

        columns_to_use, output_columns = self._produce_columns(inputs, self._add_semantic_types, self._remove_semantic_types)
        
        if self.hyperparams['replace_index_columns'] and self.hyperparams['return_result'] == 'append':
            assert len(columns_to_use) == len(output_columns)

            index_columns = inputs.metadata.get_index_columns()

            index_columns_to_use = []
            other_columns_to_use = []
            index_output_columns = []
            other_output_columns = []
            for column_to_use, output_column in zip(columns_to_use, output_columns):
                if column_to_use in index_columns:
                    index_columns_to_use.append(column_to_use)
                    index_output_columns.append(output_column)
                else:
                    other_columns_to_use.append(column_to_use)
                    other_output_columns.append(output_column)

            outputs = base_utils.combine_columns(inputs, index_columns_to_use, index_output_columns, return_result='replace', add_index_columns=self.hyperparams['add_index_columns'])
            outputs = base_utils.combine_columns(outputs, other_columns_to_use, other_output_columns, return_result='append', add_index_columns=self.hyperparams['add_index_columns'])
        else:
            outputs = base_utils.combine_columns(inputs, columns_to_use, output_columns, return_result=self.hyperparams['return_result'], add_index_columns=self.hyperparams['add_index_columns'])
        ## EMD originally from from d3m.primitives.schema_discovery.profiler.Common """

        return CallResult(outputs, has_finished = self._is_fit)

    def produce_metafeatures(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ Produce primitive's best guess for the structural type of each input column.

            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target

            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})

            Raises:
                PrimitiveNotFittedError: if primitive not fit

            Returns:
                CallResult[Outputs] -- dataframe with two columns: "semantic type classifications" and "probabilities"
                    Each row represents a column in the original dataframe. The column "semantic type
                    classifications" contains a list of all semantic type labels and the column
                    "probabilities" contains a list of the model's confidence in assigning each
                    respective semantic type label
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        out_df = self._produce_annotations(inputs=inputs)

        # add metadata to output data frame
        simon_df = d3m_DataFrame(out_df)
        # first column list of ('semantic types')
        col_dict = dict(simon_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict["structural_type"] = typing.List[str]
        col_dict["name"] = "semantic types"
        col_dict["semantic_types"] = (
            "http://schema.org/Text",
            "https://metadata.datadrivendiscovery.org/types/Attribute",
        )
        simon_df.metadata = simon_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 0), col_dict
        )
        # second column ('probabilities')
        col_dict = dict(simon_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict["structural_type"] = typing.List[float]
        col_dict["name"] = "probabilities"
        col_dict["semantic_types"] = (
            "http://schema.org/Text",
            "https://metadata.datadrivendiscovery.org/types/Attribute",
            "https://metadata.datadrivendiscovery.org/types/FloatVector"
        )
        simon_df.metadata = simon_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 1), col_dict
        )

        return CallResult(simon_df, has_finished=self._is_fit)

    def _can_use_column(self, inputs_metadata: metadata_base.DataMetadata, column_index: int) -> bool:
        """ originally from from d3m.primitives.schema_discovery.profiler.Common """

        column_metadata = inputs_metadata.query_column(column_index)

        semantic_types = column_metadata.get('semantic_types', [])

        # We detect only on columns which have no semantic types or where it is explicitly set as unknown.
        if not semantic_types or 'https://metadata.datadrivendiscovery.org/types/UnknownType' in semantic_types:
            return True

        # A special case to handle setting "https://metadata.datadrivendiscovery.org/types/TrueTarget".
        if 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in semantic_types:
            return True

        return False

    def _get_columns(self, inputs_metadata: metadata_base.DataMetadata) -> typing.List[int]:
        """ originally from from d3m.primitives.schema_discovery.profiler.Common """

        def can_use_column(column_index: int) -> bool:
            # if overwrite, we detect on all columns
            if self.hyperparams['overwrite']:
                return True

            return self._can_use_column(inputs_metadata, column_index)

        columns_to_use, columns_not_to_use = base_utils.get_columns_to_use(inputs_metadata, self.hyperparams['use_columns'], self.hyperparams['exclude_columns'], can_use_column)

        # We are OK if no columns ended up being parsed.
        # "base_utils.combine_columns" will throw an error if it cannot work with this.

        if self.hyperparams['use_columns'] and columns_not_to_use:
            self.logger.warning("Not all specified columns can parsed. Skipping columns: %(columns)s", {
                'columns': columns_not_to_use,
            })

        return columns_to_use

    def _append_simon_annotations(self, new_column_semantic_types, col_idx):

        simon_labels = self.simon_annotations["semantic types"][col_idx]
        simon_probabilities = self.simon_annotations["probabilities"][col_idx]

        # filter labels and probs by those specified in HP
        filtered_labels, filtered_probabilities = [], []
        for label, prob in zip(simon_labels, simon_probabilities):
            if SIMON_ANNOTATIONS_DICT[label] in self.hyperparams['detect_semantic_types']:
                filtered_labels.append(SIMON_ANNOTATIONS_DICT[label])
                filtered_probabilities.append(prob)

        if self.hyperparams["multi_label_classification"]:
            new_column_semantic_types.extend(filtered_labels)
        else:
            if len(filtered_labels) > 0:
                new_column_semantic_types.append(filtered_labels[np.argmax(filtered_probabilities)])
        return new_column_semantic_types

    def _produce_annotations(self, inputs: Inputs) -> Outputs:
        """ generates dataframe with semantic type classifications and classification probabilities
            for each column of original dataframe

        Arguments:
            inputs {Inputs} -- D3M dataframe

        Returns:
            Outputs -- dataframe with two columns: "semantic type classifications" and "probabilities"
                       Each row represents a column in the original dataframe. The column "semantic type
                       classifications" contains a list of all semantic type labels and the column
                       "probabilities" contains a list of the model's confidence in assigning each
                       respective semantic type label
        """

        # load model checkpoint
        checkpoint_dir = (
            self._volumes["simon_models_1"] + "/simon_models_1/pretrained_models/"
        )
        if self.hyperparams["statistical_classification"]:
            execution_config = "Base.pkl"
            category_list = "/Categories.txt"
        else:
            execution_config = "Base_stat_geo.pkl"
            category_list = "/Categories_base_stat_geo.txt"
        with open(
            self._volumes["simon_models_1"] + "/simon_models_1" + category_list, "r"
        ) as f:
            Categories = f.read().splitlines()

        # create model object
        Classifier = Simon(encoder={})
        config = Classifier.load_config(execution_config, checkpoint_dir)
        encoder = config["encoder"]
        checkpoint = config["checkpoint"]
        model = Classifier.generate_model(20, self.hyperparams["max_rows"], len(Categories))
        Classifier.load_weights(checkpoint, None, model, checkpoint_dir)
        model.compile(
            loss="binary_crossentropy", optimizer="adam", metrics=["binary_accuracy"]
        )

        # prepare data and make predictions
        frame = inputs.copy()
        prepped_data = encoder.encodeDataFrame(frame)
        preds = model.predict_on_batch(tf.constant(prepped_data))
        logger.debug('------------Reverse label encoding------------')
        decoded_preds = encoder.reverse_label_encode(
            preds, self.hyperparams["p_threshold"]
        )

        # apply statistical / ordinal classification if desired
        if self.hyperparams["statistical_classification"]:
            logger.debug("Beginning Guessing categorical/ordinal classifications...")
            raw_data = frame.values
            guesses = [
                guess(raw_data[:, i], for_types="category")
                for i in np.arange(raw_data.shape[1])
            ]

            # probability of rule-based statistical / ordinal classifications = min probability of existing classifications
            for i, g in enumerate(guesses):
                if g[0] == "category":
                    if len(decoded_preds[1][i]) == 0:
                        guess_prob = self.hyperparams['p_threshold']
                    else:
                        guess_prob = min(decoded_preds[1][i])
                    decoded_preds[0][i] += ("categorical",)
                    decoded_preds[1][i].append(guess_prob)
                    if (
                        ("int" in decoded_preds[1][i])
                        or ("float" in decoded_preds[1][i])
                        or ("datetime" in decoded_preds[1][i])
                    ):
                        decoded_preds[0][i] += ("ordinal",)
                        decoded_preds[1][i].append(guess_prob)
            logger.debug("Done with statistical variable guessing")

        # clear tf session, remove unnecessary files
        Classifier.clear_session()
        os.remove('unencoded_chars.json')

        out_df = pd.DataFrame.from_records(list(decoded_preds)).T
        out_df.columns = ["semantic types", "probabilities"]
        return out_df

    def _set_target_column(self, new_column_semantic_types, true_target_columns):
        """ originally from from d3m.primitives.schema_discovery.profiler.Common """

        if not true_target_columns \
                and not self.has_set_target_columns \
                and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' in self.hyperparams['detect_semantic_types'] \
                and 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget' in new_column_semantic_types:
             # It should not be set because there are no columns with this semantic type in whole DataFrame.
            assert 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in new_column_semantic_types
            new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/TrueTarget')
            if 'https://metadata.datadrivendiscovery.org/types/Target' not in new_column_semantic_types:
                new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Target')
            if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
            self.has_set_target_columns = True
        return new_column_semantic_types

    def _set_index_column(self, new_column_semantic_types, column_name):
        """ originally from from d3m.primitives.schema_discovery.profiler.Common """

        if 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' in self.hyperparams['detect_semantic_types'] \
                and column_name == 'd3mIndex' \
                and 'https://metadata.datadrivendiscovery.org/types/UniqueKey' in new_column_semantic_types:
            # It should not be set because there are no columns with this semantic type in whole DataFrame.
            assert 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types
            assert 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types
            new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
            new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/UniqueKey')
            if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
            self.has_set_index_column = True
        elif 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' in self.hyperparams['detect_semantic_types'] \
                and column_name == 'd3mIndex':
            assert 'https://metadata.datadrivendiscovery.org/types/UniqueKey' not in new_column_semantic_types
            # It should not be set because there are no columns with this semantic type in whole DataFrame.
            assert 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types
            assert 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types
            new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey')
            if 'https://metadata.datadrivendiscovery.org/types/Attribute' in new_column_semantic_types:
                new_column_semantic_types.remove('https://metadata.datadrivendiscovery.org/types/Attribute')
            self.has_set_index_column = True
        return new_column_semantic_types

    def _set_attribute_column(self, new_column_semantic_types):
        """ originally from from d3m.primitives.schema_discovery.profiler.Common """

        if 'https://metadata.datadrivendiscovery.org/types/Attribute' in self.hyperparams['detect_semantic_types'] \
                and 'https://metadata.datadrivendiscovery.org/types/TrueTarget' not in new_column_semantic_types \
                and 'https://metadata.datadrivendiscovery.org/types/PrimaryKey' not in new_column_semantic_types \
                and 'https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey' not in new_column_semantic_types \
                and 'https://metadata.datadrivendiscovery.org/types/Attribute' not in new_column_semantic_types:
            new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Attribute')
        return new_column_semantic_types

    def _set_additional_time_label(self, new_column_semantic_types):
        """ originally from from d3m.primitives.schema_discovery.profiler.Common """

        if 'https://metadata.datadrivendiscovery.org/types/Time' in self.hyperparams['detect_semantic_types'] \
                and 'http://schema.org/DateTime' in new_column_semantic_types \
                and 'https://metadata.datadrivendiscovery.org/types/Time' not in new_column_semantic_types:
            new_column_semantic_types.append('https://metadata.datadrivendiscovery.org/types/Time')
        return new_column_semantic_types

    def _produce_columns(
        self, inputs: Inputs,
        add_semantic_types: typing.List[typing.List[str]],
        remove_semantic_types: typing.List[typing.List[str]],
    ) -> typing.Tuple[typing.List[int], typing.List[Outputs]]:

        """ originally from from d3m.primitives.schema_discovery.profiler.Common """
        columns_to_use = self._get_columns(inputs.metadata)

        assert len(add_semantic_types), len(remove_semantic_types)

        if len(columns_to_use) != len(add_semantic_types):
            raise exceptions.InvalidStateError("Producing on a different number of columns than fitting.")

        output_columns = []

        for col_index, column_add_semantic_types, column_remove_semantic_types in zip(columns_to_use, add_semantic_types, remove_semantic_types):
            output_column = inputs.select_columns([col_index])

            for remove_semantic_type in column_remove_semantic_types:
                output_column.metadata = output_column.metadata.remove_semantic_type((metadata_base.ALL_ELEMENTS, 0), remove_semantic_type)
            for add_semantic_type in column_add_semantic_types:
                output_column.metadata = output_column.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, 0), add_semantic_type)

            output_columns.append(output_column)

        assert len(output_columns) == len(columns_to_use)

        return columns_to_use, output_columns

    def get_params(self) -> Params:
        if not self._is_fit:
            return Params(
                add_semantic_types=None,
                remove_semantic_types=None,
            )

        return Params(
            add_semantic_types=self._add_semantic_types,
            remove_semantic_types=self._remove_semantic_types,
        )

    def set_params(self, *, params: Params) -> None:
        self._add_semantic_types = params['add_semantic_types']
        self._remove_semantic_types = params['remove_semantic_types']
        self._is_fit = all(param is not None for param in params.values())
