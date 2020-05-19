import os
import sys
import collections
import typing
import re
from json import JSONDecoder
from typing import List

import pandas as pd
import requests
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base
from d3m.container import DataFrame as d3m_DataFrame
from d3m.container import List as d3m_List

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
        try:
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        except KeyError:
            return -1

    def set(self, key, value):
        try:
            self.cache.pop(key)
        except KeyError:
            if len(self.cache) >= self.capacity:
                self.cache.popitem(last=False)
        self.cache[key] = value

class Hyperparams(hyperparams.Hyperparams):
    rampup_timeout = hyperparams.UniformInt(
        lower=1,
        upper=sys.maxsize,
        default=100,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="timeout, how much time to give elastic search database to startup, may vary based on infrastructure",
    )
    target_columns = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="indices of column with geolocation formatted as text that should be converted to lat,lon pairs",
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


class GoatForwardPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    Geocode all names of locations in specified columns into lat/long pairs.

    Parameters
    ----------
    inputs : pandas dataframe containing strings representing some geographic locations -
                 (name, address, etc) - one location per row in columns marked as locationIndicator

    Returns
    -------
    Outputs
        Pandas dataframe, with a pair of 2 float columns -- [longitude, latitude] -- per original row/location column
        appended as new columns
    """

    # Make sure to populate this with JSON annotations...
    # This should contain only metadata which cannot be automatically determined from the code.
    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "c7c61da3-cf57-354e-8841-664853370106",
            "version": __version__,
            "name": "Goat_forward",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": ["Geocoder"],
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
                    "package": "default-jre-headless",
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
            "python_path": "d3m.primitives.data_cleaning.geocoding.Goat_forward",
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

    def _is_geocoded(self, geocode_result) -> bool:
        # check if geocoding was successful or not
        if (
            geocode_result["features"] and len(geocode_result["features"]) > 0
        ):  # make sure (sub-)dictionaries are non-empty
            if geocode_result["features"][0]["geometry"]:
                if geocode_result["features"][0]["geometry"]["coordinates"]:
                    return True
        return False

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Accept a set of location strings, processes it and returns a set of long/lat coordinates.

        Parameters
        ----------
        inputs : pandas dataframe containing strings representing some geographic locations -
                 (name, address, etc) - one location per row in the specified target column

        timeout : float
            A maximum time this primitive should take to produce outputs during this method call, in seconds. N/A
        iterations : int
            How many of internal iterations should the primitive do. N/A for now...

        Returns
        -------
        Outputs
            Pandas dataframe, with a pair of 2 float columns -- [longitude, latitude] -- per original row/location column
        """

        # confirm that server is responding before proceeding
        address = "http://localhost:2322/"
        PopenObj = check_geocoding_server(
            address, self.volumes, self.hyperparams["rampup_timeout"]
        )

        # target columns are columns with location tag
        target_column_idxs = self.hyperparams["target_columns"]
        target_columns = [list(inputs)[idx] for idx in target_column_idxs]
        target_columns_long_lat = [
            target_columns[i // 2] for i in range(len(target_columns) * 2)
        ]
        outputs = inputs.remove_columns(target_column_idxs)

        # geocode each requested location
        output_data = []
        for i, ith_column in enumerate(target_columns):
            j = 0
            target_columns_long_lat[2 * i] = (
                target_columns_long_lat[2 * i] + "_longitude"
            )
            target_columns_long_lat[2 * i + 1] = (
                target_columns_long_lat[2 * i + 1] + "_latitude"
            )

            # remove ampersand from strings
            inputs_cleaned = inputs[ith_column].apply(
                lambda val: re.sub(r"\s*&\s*", r" and ", val)
            )
            for location in inputs_cleaned:
                cache_ret = self.goat_cache.get(location)
                row_data = []
                if cache_ret == -1:
                    r = requests.get(address + "api?q=" + location)
                    tmp = self._decoder.decode(r.text)
                    if self._is_geocoded(tmp):
                        row_data = tmp["features"][0]["geometry"]["coordinates"]
                        self.goat_cache.set(location, str(row_data))
                    else:
                        self.goat_cache.set(location, "[float('nan'), float('nan')]")
                else:
                    # cache_ret is [longitude, latitude]
                    row_data = [eval(cache_ret)[0], eval(cache_ret)[1]]

                if len(output_data) <= j:
                    output_data.append(row_data)
                else:
                    output_data[j] = output_data[j] + row_data
                j = j + 1
        # need to cleanup by closing the server when done...
        PopenObj.kill()

        # Build d3m-type dataframe
        out_df = pd.DataFrame(
            output_data, index=range(inputs.shape[0]), columns=target_columns_long_lat
        )
        d3m_df = d3m_DataFrame(out_df)
        for i, ith_column in enumerate(target_columns_long_lat):
            # for every column
            col_dict = dict(d3m_df.metadata.query((metadata_base.ALL_ELEMENTS, i)))
            col_dict["structural_type"] = type(0.0)
            col_dict["semantic_types"] = (
                "http://schema.org/Float",
                "https://metadata.datadrivendiscovery.org/types/Attribute",
            )
            col_dict["name"] = target_columns_long_lat[i]
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
