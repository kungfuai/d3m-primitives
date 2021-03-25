"""
   Copyright © 2019 Uncharted Software Inc.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

import unittest
from os import path
import sys

from d3m import container

from d3m.metadata import base as metadata_base
from d3m.primitives.clustering.k_means import Sloth
from distil.primitives import utils

import numpy as np
import pandas as pd

import utils as test_utils


class SlothTestCase(unittest.TestCase):

    _dataset_path = path.abspath(
        path.join(path.dirname(__file__), "timeseries_dataset")
    )

    def test_defaults_produce(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # mark grouping key
        dataframe.metadata = dataframe.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/GroupingKey",
        )

        # create the clustering primitive and cluster
        hyperparams_class = Sloth.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        sloth = Sloth(hyperparams=hyperparams_class.defaults())
        result = sloth.produce(inputs=dataframe).value

        # check if the first four columns match the original
        pd.testing.assert_frame_equal(result.iloc[:, :-1], dataframe)

        # check that the first two keys are in their own cluster, and the the last two are in
        # the same cluster
        clusters_by_key = result[["key", "__cluster"]].drop_duplicates()
        self.assertNotEquals(clusters_by_key.iloc[0, 1], clusters_by_key.iloc[1, 1])
        self.assertNotEquals(clusters_by_key.iloc[1, 1], clusters_by_key.iloc[2, 1])
        self.assertEquals(clusters_by_key.iloc[2, 1], clusters_by_key.iloc[3, 1])

        # check metadata is correct for new column
        column_metadata = result.metadata.query_column(4)
        self.assertListEqual(
            list(column_metadata["semantic_types"]),
            [
                "https://metadata.datadrivendiscovery.org/types/Attribute",
                "https://metadata.datadrivendiscovery.org/types/ConstructedAttribute",
                "http://schema.org/Integer",
            ],
        ),
        self.assertEqual(column_metadata["structural_type"], np.int64)

    def test_defaults_produce_clusters(self) -> None:
        # load test data into a dataframe
        dataset = test_utils.load_dataset(self._dataset_path)
        dataframe = test_utils.get_dataframe(dataset, "learningData")

        # mark grouping key
        dataframe.metadata = dataframe.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/GroupingKey",
        )

        # create the clustering primitive and cluster
        hyperparams_class = Sloth.metadata.query()["primitive_code"][
            "class_type_arguments"
        ]["Hyperparams"]
        sloth = Sloth(hyperparams=hyperparams_class.defaults())
        result = sloth.produce_clusters(inputs=dataframe).value

        # check that the grouping key columns match
        self.assertListEqual(
            result.iloc[:, 0].tolist(), ["alpha", "bravo", "charlie", "delta"]
        )

        # check that the first two keys are in their own cluster, and the the last two are in
        # the same cluster
        clusters_by_key = result[["key", "__cluster"]].drop_duplicates()
        self.assertNotEquals(clusters_by_key.iloc[0, 1], clusters_by_key.iloc[1, 1])
        self.assertNotEquals(clusters_by_key.iloc[1, 1], clusters_by_key.iloc[2, 1])
        self.assertEquals(clusters_by_key.iloc[2, 1], clusters_by_key.iloc[3, 1])

        # check metadata is correct
        column_metadata = result.metadata.query_column(0)
        self.assertEqual(column_metadata["structural_type"], str)
        column_metadata = result.metadata.query_column(1)
        self.assertEqual(column_metadata["structural_type"], np.int64)


if __name__ == "__main__":
    unittest.main()