import os.path
import typing

import pandas
from punk.feature_selection import RFFeatures
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params

__author__ = 'Distil'
__version__ = '3.1.2'
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    bestFeatures: typing.List[str]

class Hyperparams(hyperparams.Hyperparams):
    proportion_of_features = hyperparams.Uniform(lower = 0.0, upper = 1.0, default = 1.0,
        upper_inclusive = True, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
        description = 'proportion of top features from input dataset to keep')
    only_numeric_cols = hyperparams.UniformBool(default = False, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/TuningParameter'],
       description="consider only numeric columns for feature selection")

class RfFeaturesPrimitive(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Perform supervised recursive feature elimination using random forests to generate an ordered
        list of features
        Parameters
        ----------
        inputs : Input D3M pandas frame
        Returns
        -------
        Outputs : D3M frame with top num_features selected by algorithm
        """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "ef6f3887-b253-4bfd-8b35-ada449efad0c",
        'version': __version__,
        'name': "RF Features",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Rank and score numeric features based on Random Forest and Recursive Feature Elimination'],
        'source': {
            'name': __author__,
            'contact': __contact__,
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
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.feature_selection.rffeatures.Rffeatures',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.RANDOM_FOREST,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.FEATURE_SELECTION,

    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)
        self.rff_features = None
        self.num_features = None
        self.bestFeatures = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits rffeatures feature selection algorithm on the training set. applies same feature selection to test set
        for consistency with downstream classifiers
        '''
        # set threshold for top features
        self.bestFeatures = self.rff_features.iloc[0:self.num_features].values
        self.bestFeatures = [row[0] for row in self.bestFeatures]
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

        self.num_features = int(inputs.shape[1] * self.hyperparams['proportion_of_features'])

        # remove primary key and targets from feature selection
        inputs_primary_key = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        inputs_target = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Target')

        # extract numeric columns and suggested target
        if self.hyperparams['only_numeric_cols']:
            inputs_float = inputs.metadata.get_columns_with_semantic_type('http://schema.org/Float')
            inputs_integer = inputs.metadata.get_columns_with_semantic_type('http://schema.org/Integer')
            inputs_numeric = [*inputs_float, *inputs_integer]
            inputs_cols = [x for x in inputs_numeric if x not in inputs_primary_key and x not in inputs_target]
        else:
            inputs_cols = [x for x in range(inputs.shape[1]) if x not in inputs_primary_key and x not in inputs_target]

        # generate feature ranking
        self.rff_features = pandas.DataFrame(RFFeatures().rank_features(inputs = inputs.iloc[:, inputs_cols], targets = pandas.DataFrame(inputs.iloc[:, inputs_target])), columns=['features'])


    def produce_metafeatures(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Perform supervised recursive feature elimination using random forests to generate an ordered
        list of features
        Parameters
        ----------
        inputs : Input pandas frame, NOTE: Target column MUST be the last column
        Returns
        -------
        Outputs : pandas frame with ordered list of original features in first column
        """
        # add metadata to output dataframe
        rff_df = d3m_DataFrame(RFFeatures().rank_features(inputs = inputs.iloc[:,:-1], targets = pandas.DataFrame(inputs.iloc[:,-1])), columns=['features'])
        # first column ('features')
        col_dict = dict(rff_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("it is a string")
        col_dict['name'] = 'features'
        col_dict['semantic_types'] = ('http://schema.org/Text', 'https://metadata.datadrivendiscovery.org/types/Attribute')
        rff_df.metadata = rff_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

        return CallResult(rff_df)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Perform supervised recursive feature elimination using random forests to generate an ordered
        list of features
        Parameters
        ----------
        inputs : Input pandas frame, NOTE: Target column MUST be the last column
        Returns
        -------
        Outputs : pandas frame with ordered list of original features in first column
        """

        inputs_target = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Target')

        features = [inputs.columns.get_loc(row) for row in self.bestFeatures] # get integer location for each label
        # add suggested target
        features = [*features, *inputs_target]

        # drop all values below threshold value
        result = inputs.select_columns(features)

        return CallResult(result)