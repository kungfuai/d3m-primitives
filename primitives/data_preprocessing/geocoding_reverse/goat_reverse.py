import os
import sys
import subprocess
import collections
import time
import typing
from json import JSONDecoder
from typing import List

import pandas as pd
import requests
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base, params
from d3m.container import DataFrame as d3m_DataFrame

from ..utils.geocoding import check_geocoding_server

__author__ = "Distil"
__version__ = "1.0.8"
__contact__ = "mailto:jeffrey.gleason@kungfu.ai"


Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

# LRU Cache helper class
class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = collections.OrderedDict()

    def get(self, key):
        key = "".join(str(e) for e in key)
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def set(self, key, value):
        key = "".join(str(e) for e in key)
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value


class Hyperparams(hyperparams.Hyperparams):
    geocoding_resolution = hyperparams.Enumeration(
        default="city",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        values=["city", "country", "state", "postcode"],
        description="type of clustering algorithm to use",
    )
    rampup_timeout = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=100,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="timeout, how much time to give elastic search database to startup, may vary based on infrastructure",
    )
    cache_size = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=2000,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="LRU cache size",
    )


class GoatReversePrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Accept a set of lat/long pair, processes it and returns a set corresponding geographic location names
    
    Parameters
    ----------
    inputs : pandas dataframe containing 2 coordinate float values, i.e., [longitude,latitude] 
                representing each geographic location of interest - a pair of values
                per location/row in the specified target column

    Returns
    -------
    Outputs
        Pandas dataframe containing one location per longitude/latitude pair (if reverse
        geocoding possible, otherwise NaNs) appended as new columns
    """

    # Make sure to populate this with JSON annotations...
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "f6e4880b-98c7-32f0-b687-a4b1d74c8f99",
            "version": __version__,
            "name": "Goat_reverse",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": ["Reverse Geocoder"],
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
                    "type": "UBUNTU",
                    "package": "default-jre",
                    "version": "2:1.8-56ubuntu2",
                },
                {
                    "type": "TGZ",
                    "key": "photon-db-latest",
                    "file_uri": "http://public.datadrivendiscovery.org/photon.tar.gz",
                    "file_digest": "d7e3d5c6ae795b5f53d31faa3a9af63a9691070782fa962dfcd0edf13e8f1eab",
                },
            ],
            # The same path the primitive is registered with entry points in setup.py.
            "python_path": "d3m.primitives.data_cleaning.geocoding.Goat_reverse",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [metadata_base.PrimitiveAlgorithmType.NUMERICAL_METHOD],
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
            hyperparams=hyperparams,
            random_seed=random_seed,
            volumes=volumes,
        )

        self._decoder = JSONDecoder()
        self.volumes = volumes
        self.goat_cache = LRUCache(self.hyperparams["cache_size"])

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Accept a set of lat/long pair, processes it and returns a set corresponding geographic location names
        
        Parameters
        ----------
        inputs : pandas dataframe containing 2 coordinate float values, i.e., [longitude,latitude] 
                 representing each geographic location of interest - a pair of values
                 per location/row in the specified target column

        Returns
        -------
        Outputs
            Pandas dataframe containing one location per longitude/latitude pair (if reverse
            geocoding possible, otherwise NaNs)
        """

        # confirm that server is responding before proceeding
        address = "http://localhost:2322/"
        PopenObj = check_geocoding_server(
            address, self.volumes, self.hyperparams["rampup_timeout"]
        )

        # find location columns, real columns, and real-vector columns
        targets = inputs.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/Location"
        )
        real_values = inputs.metadata.get_columns_with_semantic_type(
            "http://schema.org/Float"
        )
        real_values += inputs.metadata.get_columns_with_semantic_type(
            "http://schema.org/Integer"
        )
        real_values = list(set(real_values))
        real_vectors = inputs.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/FloatVector"
        )
        target_column_idxs = []
        target_columns = []

        # convert target columns to list if they have single value and are adjacent in the df
        for target, target_col in zip(targets, [list(inputs)[idx] for idx in targets]):
            if target in real_vectors:
                target_column_idxs.append(target)
                target_columns.append(target_col)
            # pair of individual lat / lon columns already in list
            elif list(inputs)[target - 1] in target_columns:
                continue
            elif target in real_values:
                if target + 1 in real_values:
                    # convert to single column with list of [lat, lon]
                    col_name = "new_col_" + target_col
                    inputs[col_name] = inputs.iloc[
                        :, target : target + 2
                    ].values.tolist()
                    target_columns.append(col_name)
                    target_column_idxs.append(target)
                    target_column_idxs.append(target + 1)
                    target_column_idxs.append(inputs.shape[1] - 1)

        # make sure columns are structured as 1) lat , 2) lon pairs
        for col in target_columns:
            if inputs[col].apply(lambda x: x[0]).max() > 90:
                inputs[col] = inputs[col].apply(lambda x: x[::-1])

        # delete columns with path names of nested media files
        outputs = inputs.remove_columns(target_column_idxs)

        # reverse-geocode each requested location
        output_data = []
        for i, ith_column in enumerate(target_columns):
            j = 0
            for longlat in inputs[ith_column]:
                cache_ret = self.goat_cache.get(longlat)
                row_data = []
                if cache_ret == -1:
                    r = requests.get(
                        address
                        + "reverse?lat="
                        + str(longlat[0])
                        + "&lon="
                        + str(longlat[1])
                    )
                    tmp = self._decoder.decode(r.text)
                    if len(tmp["features"]) == 0:
                        if self.hyperparams["geocoding_resolution"] == "postcode":
                            row_data = float("nan")
                        else:
                            row_data = ""
                    elif (
                        self.hyperparams["geocoding_resolution"]
                        not in tmp["features"][0]["properties"].keys()
                    ):
                        if self.hyperparams["geocoding_resolution"] == "postcode":
                            row_data = float("nan")
                        else:
                            row_data = ""
                    else:
                        row_data = tmp["features"][0]["properties"][
                            self.hyperparams["geocoding_resolution"]
                        ]
                    self.goat_cache.set(longlat, row_data)
                else:
                    row_data = cache_ret

                if len(output_data) <= j:
                    output_data.append(row_data)
                else:
                    output_data[j] = output_data[j] + row_data
                j = j + 1

        # need to cleanup by closing the server when done...
        PopenObj.kill()

        # Build d3m-type dataframe
        out_df = pd.DataFrame(index=range(inputs.shape[0]),columns=target_columns)
        d3m_df = d3m_DataFrame(out_df)
        for i, ith_column in enumerate(target_columns):
            # for every column
            col_dict = dict(d3m_df.metadata.query((metadata_base.ALL_ELEMENTS, i)))
            if self.hyperparams["geocoding_resolution"] == "postcode":
                col_dict["structural_type"] = type(1)
                col_dict["semantic_types"] = (
                    "http://schema.org/Integer",
                    "https://metadata.datadrivendiscovery.org/types/Attribute",
                )
            else:
                col_dict["structural_type"] = type("it is a string")
                col_dict["semantic_types"] = (
                    "http://schema.org/Text",
                    "https://metadata.datadrivendiscovery.org/types/Attribute",
                )
            col_dict["name"] = target_columns[i]
            d3m_df.metadata = d3m_df.metadata.update(
                (metadata_base.ALL_ELEMENTS, i), col_dict
            )
        df_dict = dict(d3m_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict_1 = dict(d3m_df.metadata.query((metadata_base.ALL_ELEMENTS,)))
        df_dict["dimension"] = df_dict_1
        df_dict_1["name"] = "columns"
        df_dict_1["semantic_types"] = (
            "https://metadata.datadrivendiscovery.org/types/TabularColumn",
        )
        df_dict_1["length"] = d3m_df.shape[1]
        d3m_df.metadata = d3m_df.metadata.update((metadata_base.ALL_ELEMENTS,), df_dict)
        return CallResult(outputs.append_columns(d3m_df))
