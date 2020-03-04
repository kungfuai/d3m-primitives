import sys
import os.path
import numpy as np
import pandas as pd
import functools
from typing import List, Union, Optional, Tuple
import logging

from ..utils.cluster import KMeans
from tslearn import utils as ts_utils

from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataframe_utils

__author__ = 'Distil'
__version__ = '2.0.5'
__contact__ = 'mailto:jeffrey.gleason@yonder.co'


logger = logging.getLogger(__name__)
#logger.setLevel(logging.DEBUG)

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(default = 'TimeSeriesKMeans',
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        values = ['GlobalAlignmentKernelKMeans', 'TimeSeriesKMeans'],
        description = 'type of clustering algorithm to use')
    nclusters = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default=3, semantic_types=
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description = 'number of clusters \
        to user in kernel kmeans algorithm')
    n_init = hyperparams.UniformInt(lower=1, upper=sys.maxsize, default=10, semantic_types=
        ['https://metadata.datadrivendiscovery.org/types/TuningParameter'], description = 'Number of times the k-means algorithm \
        will be run with different centroid seeds. Final result will be the best output on n_init consecutive runs in terms of inertia')
    time_col_index = hyperparams.Hyperparameter[Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column in input dataframe containing timestamps.'
    )
    value_col_index = hyperparams.Hyperparameter[Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column in input dataframe containing the values associated with the timestamps.'
    )
    grouping_col_index = hyperparams.Hyperparameter[Union[int, None]](
        default=None,
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Index of column in input dataframe containing the values used to mark timeseries groups'
    )
    output_col_name = hyperparams.Hyperparameter[str](
        default='__cluster',
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        description='Name to assign to cluster column that is appended to the input dataset'
    )

class StorcPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
        Primitive that applies kmeans clustering to time series data. Algorithm options are 'GlobalAlignmentKernelKMeans'
        or 'TimeSeriesKMeans,' both of which are bootstrapped from the base library tslearn.clustering. This is an unsupervised,
        clustering primitive, but has been represented as a supervised classification problem to produce a compliant primitive.

        Training inputs: D3M dataset with features and labels, and D3M indices
        Outputs: D3M dataset with predicted labels and D3M indices
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "77bf4b92-2faa-3e38-bb7e-804131243a7f",
        'version': __version__,
        'name': "Sloth",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series','Clustering'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            "uris": [
                # Unstructured URIs.
                "https://github.com/Yonder-OSS/D3M-Primitives",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        "installation": [
            {"type": "PIP", "package": "cython", "version": "0.29.14"},
            {
                "type": metadata_base.PrimitiveInstallationType.PIP,
                "package_uri": "git+https://github.com/Yonder-OSS/D3M-Primitives.git@{git_commit}#egg=yonder-primitives".format(
                    git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                ),
            },
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.clustering.k_means.Sloth',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.K_MEANS_CLUSTERING,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.CLUSTERING,
    })


    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._kmeans = KMeans(self.hyperparams['nclusters'], self.hyperparams['algorithm'])
        self._grouping_key_col: Optional[int] = None
        self._timestamp_col: Optional[int] = None
        self._value_col: Optional[int] = None

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[container.pandas.DataFrame]:
        """
        Parameters
        ----------
        inputs : D3M dataframe with associated metadata.

        Returns
        -------
        Outputs
            For unsupervised problems: The output is a dataframe containing a single column where each entry is the associated series' cluster number.
            For semi-supervised problems: The output is the input df containing an additional feature - cluster_label
        """

        # generate the clusters
        self._get_columns(inputs)
        clusters = self._get_clusters(inputs)

        # append the cluster column
        cluster_df = pd.DataFrame(clusters, columns=('key', self.hyperparams['output_col_name']))
        outputs = inputs.join(cluster_df.set_index('key'), on=inputs.columns[self._grouping_key_col])
        outputs.metadata = outputs.metadata.generate(outputs)

        # update the new column metadata
        outputs.metadata = outputs.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, len(outputs.columns)-1), 'https://metadata.datadrivendiscovery.org/types/Attribute')
        outputs.metadata = outputs.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, len(outputs.columns)-1), 'https://metadata.datadrivendiscovery.org/types/ConstructedAttribute')
        outputs.metadata = outputs.metadata.add_semantic_type((metadata_base.ALL_ELEMENTS, len(outputs.columns)-1), 'http://schema.org/Integer')

        return CallResult(outputs)


    def produce_clusters(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[container.pandas.DataFrame]:
        # generate the clusters
        self._get_columns(inputs)
        clusters = self._get_clusters(inputs)

        # generate the response df
        cluster_df = container.DataFrame(clusters, columns=('key', self.hyperparams['output_col_name']), generate_metadata=True)
        return CallResult(cluster_df)


    def _get_columns(self, inputs: Inputs) -> None:
        # if the grouping col isn't set infer based on presence of grouping key
        grouping_key_cols = self.hyperparams.get('grouping_col_index', None)
        if grouping_key_cols is None:
            grouping_key_cols = inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/GroupingKey',))
            if grouping_key_cols:
                self._grouping_key_col = grouping_key_cols[0]
            else:
                # if no grouping key is specified we can't split, and therefore we can't cluster.
                return None
        else:
            self._grouping_key_col = grouping_key_cols[0]

        # if the timestamp col isn't set infer based on presence of the Time role
        timestamp_cols = self.hyperparams.get('timestamp_col_index', None)
        if timestamp_cols is None:
            self._timestamp_col = inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Time',))[0]
        else:
            self._timestamp_col = timestamp_cols[0]

        # if the value col isn't set, take the first integer/float attribute we come across that isn't the grouping or timestamp col
        value_cols = self.hyperparams.get('value_col_index', None)
        if value_cols is None:
            attribute_cols = inputs.metadata.list_columns_with_semantic_types(('https://metadata.datadrivendiscovery.org/types/Attribute',))
            numerical_cols = inputs.metadata.list_columns_with_semantic_types(('http://schema.org/Integer', 'http://schema.org/Float'))

            for idx in numerical_cols:
                if idx != self._grouping_key_col and idx != self._timestamp_col and idx in attribute_cols:
                    self._value_col = idx
                    break
                self._value_col = -1
        else:
            self._value_col = value_cols[0]


    def _get_clusters(self, inputs: Inputs) -> List[Tuple[str, int]]:
        # split the long form series out into individual series
        groups = inputs.groupby(inputs.columns[self._grouping_key_col])

        # we need the lengths to be the same to keep tslearn happy
        max_length = max([len(group) for _, group in groups])

        timeseries = []
        for group_name, group in groups:
            # Ensure timeseries are list of floating point values of the same size.  Pad the ends out with NaNs and then
            # interpolate all missing data.
            timeseries_values = (group.iloc[:, self._value_col]).astype(float)
            timeseries_padded = (timeseries_values.append(pd.Series([np.nan] * (max_length - timeseries_values.shape[0]))))
            timeseries_padded.interpolate(inplace=True)
            timeseries.append(timeseries_padded)

        keys = [group_name for group_name, _ in groups]

        # cluster the data
        timeseries_dataset = ts_utils.to_time_series_dataset(timeseries) # needed to get rid of tslearn dimension warning
        self._kmeans.fit(timeseries_dataset)
        clusters = self._kmeans.predict(timeseries_dataset)

        return list(zip(keys, clusters))
