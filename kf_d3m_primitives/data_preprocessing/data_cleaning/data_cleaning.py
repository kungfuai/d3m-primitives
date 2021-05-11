import os.path

import pandas
from punk.preppy import CleanStrings, CleanDates, CleanNumbers
from d3m.primitive_interfaces.transformer import TransformerPrimitiveBase
from d3m.primitive_interfaces.base import CallResult
from d3m import container, utils
from d3m.metadata import hyperparams, base as metadata_base

__author__ = "Distil"
__version__ = "3.0.2"
__contact__ = "mailto:cbethune@uncharted.software"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame


class Hyperparams(hyperparams.Hyperparams):
    pass


class DataCleaningPrimitive(TransformerPrimitiveBase[Inputs, Outputs, Hyperparams]):
    """
    This primitive standardizes columns that represent dates or numbers, including missing values.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "fc6bf33a-f3e0-3496-aa47-9a40289661bc",
            "version": __version__,
            "name": "Data cleaning",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": ["Clean data values in data frame"],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": [
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
            "python_path": "d3m.primitives.data_cleaning.data_cleaning.Datacleaning",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.DATA_STRUCTURE_ALIGNMENT,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.DATA_CLEANING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """
        Parameters
        ----------
        inputs: D3M dataframe

        Returns
        ----------
        Outputs: A frame structurally identical to the input frame, with each feature
            standardized according to its type (e.g. all date objects will be modified to be
            of a common structure)
        """

        string_cleaner = CleanStrings()
        number_cleaner = CleanNumbers()
        date_cleaner = CleanDates()

        def dtype_apply(series):
            if series.dtype in ["int64", "float64"]:
                return number_cleaner.clean_numbers(series)
            elif series.dtype in ["object"]:
                return string_cleaner.clean_strings(series)
            elif "datetime" in series.dtype:
                return date_cleaner.clean_dates(series)
            else:
                return series

        try:
            return CallResult(inputs.apply(dtype_apply))
        except:
            return CallResult(inputs)
