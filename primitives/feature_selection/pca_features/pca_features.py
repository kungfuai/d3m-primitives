import os.path
import pandas
import sys
import typing
from punk.feature_selection import PCAFeatures
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = 'Distil'
__version__ = '3.1.2'
__contact__ = 'mailto:numa@yonder.co'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    bestFeatures: typing.List[int]

class Hyperparams(hyperparams.Hyperparams):
    threshold = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 0.0,
        upper_inclusive = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
       description = 'pca score threshold for feature selection')
    only_numeric_cols = hyperparams.UniformBool(default = True, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
       description="consider only numeric columns for feature selection")

class PcaFeaturesPrimitive(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Perform principal component analysis on all numeric data in the dataset
        and then use each original features contribution to the first principal
        component as a proxy for the 'score' of that feature. Returns a dataframe
        that only contains features whose score is above a threshold (HP)
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "04573880-d64f-4791-8932-52b7c3877639",
        'version': __version__,
        'name': "PCA Features",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Rank and score numeric features based on principal component analysis'],
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
        }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.feature_selection.pca_features.Pcafeatures',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.PRINCIPAL_COMPONENT_ANALYSIS,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,

    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.pca_df = None
        self.input_cols = None
        self.bestFeatures = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits pcafeatures feature selection algorithm on the training set. applies same feature selection to test set
        for consistency with downstream classifiers
        '''
        # take best features with threshold
        bestFeatures = [int(row[1]) for row in self.pca_df.itertuples() if float(row[2]) > self.hyperparams['threshold']]

        # add suggested targets to dataset containing best features
        self.bestFeatures = [self.inputs_cols[bf] for bf in bestFeatures]
        return CallResult(None)

    def get_params(self) -> Params:
        return Params(
            bestFeatures = self.bestFeatures
        )

    def set_params(self, *, params: Params) -> None:
        self.bestFeatures = params['bestFeatures']

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data
        Parameters
        ----------
        inputs = D3M dataframe
        '''
        # remove primary key and targets from feature selection
        inputs_primary_key = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        inputs_target = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        if not len(inputs_target):
            inputs_target = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Target')
        if not len(inputs_target):
            inputs_target = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')

        # extract numeric columns and suggested target
        if self.hyperparams['only_numeric_cols']:
            inputs_float = inputs.metadata.get_columns_with_semantic_type('http://schema.org/Float')
            inputs_integer = inputs.metadata.get_columns_with_semantic_type('http://schema.org/Integer')
            inputs_numeric = [*inputs_float, *inputs_integer]
            self.inputs_cols = [x for x in inputs_numeric if x not in inputs_primary_key and x not in inputs_target]
        else:
            self.inputs_cols = [x for x in range(inputs.shape[1]) if x not in inputs_primary_key and x not in inputs_target]

        # generate feature ranking
        self.pca_df = PCAFeatures().rank_features(inputs = inputs.iloc[:, self.inputs_cols])


    def produce_metafeatures(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        -------
        inputs : Input pandas frame
        Returns
        -------
        Outputs : pandas frame with list of original features in first column, ordered
            by their contribution to the first principal component, and scores in
            the second column.
        """

        # add metadata to output data frame
        pca_df = d3m_DataFrame(PCAFeatures().rank_features(inputs = inputs))
        # first column ('features')
        col_dict = dict(pca_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = 'features'
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        pca_df.metadata = pca_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)
        # second column ('scores')
        col_dict = dict(pca_df.metadata.query((metadata_base.ALL_ELEMENTS, 1)))
        col_dict['structural_type'] = type("1.0")
        col_dict['name'] = 'scores'
        col_dict['semantic_types'] = ('http://schema.org/Float', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        pca_df.metadata = pca_df.metadata.update((metadata_base.ALL_ELEMENTS, 1), col_dict)

        return CallResult(pca_df)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Parameters
        -------
        inputs : Input pandas frame
        Returns
        -------
        Outputs : pandas frame with list of original features in first column, ordered
            by their contribution to the first principal component, and scores in
            the second column.
        """

        index = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')

        # add index to dataset containing best features (suggested targets added at a later pipeline step)
        bestFeatures = index + self.bestFeatures

        # drop all columns below threshold value
        result = inputs.select_columns(bestFeatures)
        return CallResult(result)