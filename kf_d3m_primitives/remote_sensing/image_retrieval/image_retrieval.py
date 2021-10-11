import os.path
import logging
import sys
from typing import List
from time import time
import pickle
import math
from threading import Thread

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.decomposition import TruncatedSVD
from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

from .gem import linear_gem, segmented_gem, thread_helper_gem

__author__ = "Distil"
__version__ = "1.0.0"
__contact__ = "mailto:cbethune@uncharted.software"

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)


class Params(params.Params):
    pos_idxs: List[int]
    neg_idxs: List[int]
    mis_idxs: np.ndarray
    d3m_idxs: np.ndarray
    pos_scores: List[np.ndarray]
    neg_scores: List[np.ndarray]
    features: np.ndarray
    idx_name: str


class Hyperparams(hyperparams.Hyperparams):
    reduce_method = hyperparams.Enumeration(
        default="pca",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["pca", "svd"],
        description="dimensionality reduction method that is applied to feature vectors",
    )
    reduce_dimension = hyperparams.UniformInt(
        lower=0,
        upper=1024,
        default=128,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="number of dimensions in reduced feature vectors",
    )
    gem_p = hyperparams.Uniform(
        lower=0,
        upper=sys.maxsize,
        default=1,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="parameter p in generalized mean pooling; p > 1 increases the constrast of the \
                    pooled feature map; p = 1 equivalent to average pooling; p = +inf equivalent to \
                    max pooling.",
    )
    denominator_min = hyperparams.UniformInt(
        lower=0,
        upper=sys.maxsize,
        default=5,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="only ranks according to positive annotations until this many negative \
                    annotations are obtained",
    )
    dot_products_cache = hyperparams.Hyperparameter[str](
        default="dot_product_cache",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="already computed dot products will be cached in this location",
    )


class ImageRetrievalPrimitive(
    SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]
):
    """This primitive retrieves semantically similar images from an index of
        un-annotated images using heuristics.

    Training inputs: 1) Feature dataframe, 2) Label dataframe
    Outputs: D3M dataset with similarity ranking of images
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "6dd2032c-5558-4621-9bea-ea42403682da",
            "version": __version__,
            "name": "ImageRetrieval",
            "keywords": [
                "remote sensing",
                "image retrieval",
                "active search",
                "active learning",
                "euclidean distance",
                "iterative labeling",
                "similarity modeling",
            ],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": [
                    "https://github.com/kungfuai/d3m-primitives",
                ],
            },
            "installation": [
                {"type": "PIP", "package": "cython", "version": "0.29.16"},
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/kungfuai/d3m-primitives.git@{git_commit}#egg=kf-d3m-primitives".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            "python_path": "d3m.primitives.similarity_modeling.iterative_labeling.ImageRetrieval",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.ITERATIVE_LABELING,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.SIMILARITY_MODELING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self.pos_idxs = []
        self.neg_idxs = []
        self.pos_scores = []
        self.neg_scores = []
        self.random_seed = random_seed
        self.linear_threshold=10000 #TODO expose as a param
        np.random.seed(random_seed)
        cache_dir = self.hyperparams["dot_products_cache"]
        os.makedirs(cache_dir, exist_ok=True)

    def get_params(self) -> Params:
        return Params(
            pos_idxs=self.pos_idxs,
            neg_idxs=self.neg_idxs,
            mis_idxs=self.mis_idxs,
            d3m_idxs=self.d3m_idxs,
            pos_scores=self.pos_scores,
            neg_scores=self.neg_scores,
            features=self.features,
            idx_name=self.idx_name,
        )

    def set_params(self, *, params: Params) -> None:
        self.pos_idxs = params["pos_idxs"]
        self.neg_idxs = params["neg_idxs"]
        self.mis_idxs = params["mis_idxs"]
        self.d3m_idxs = params["d3m_idxs"]
        self.pos_scores = params["pos_scores"]
        self.neg_scores = params["neg_scores"]
        self.features = params["features"]
        self.idx_name = params["idx_name"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """set primitive's training data. Outputs are a series of labels, where
        1 = positive, 0 = negative, -1 = missing. Inputs should only contain
        image feature columns and, optionally, a d3mIndex.

        Arguments:
            inputs {Inputs} -- D3M dataframe containing features
            outputs {Outputs} -- D3M dataframe containing labels

        """

        index_cols = inputs.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey"
        )
        if len(index_cols):
            self.idx_name = inputs.columns[index_cols[0]]
            inputs = inputs.sort_values(by=self.idx_name)
            self.d3m_idxs = inputs[self.idx_name].values
            input_features = inputs.drop(self.idx_name, axis=1).values
        else:
            self.idx_name = "d3mIndex"
            self.d3m_idxs = np.arange(inputs.shape[0])
            input_features = inputs.values

        cache_dir = self.hyperparams["dot_products_cache"]
        features_f = os.path.join(cache_dir, "features.pkl")
        if os.path.isfile(features_f):
            self.features = pickle.load(open(features_f, "rb"))
        else:
            self.features = self._postprocess(input_features)
            pickle.dump(self.features, open(features_f, "wb"))
        self.features = np.ascontiguousarray(self.features)

        ann_index_cols = inputs.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/PrimaryMultiKey"
        )
        if len(ann_index_cols):
            outputs = outputs.sort_values(by=outputs.columns[ann_index_cols[0]])

        self.annotations = outputs["annotations"].values.astype(int)

        ann_values = np.sort(np.unique(self.annotations))
        if not np.isin(ann_values, np.array([-1, 0, 1])).all():
            raise ValueError(
                "Outputs (i.e., the annotations) must contain only values in [-1,0,1]"
            )

        self.mis_idxs = np.where(self.annotations == -1)[0]

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:

        self._attempt_load(self.hyperparams["dot_products_cache"])

        for idx in np.where(self.annotations == 1)[0]:
            if idx not in self.pos_idxs:
                self.pos_idxs.append(int(idx))
                self.pos_scores.append(self.features @ self.features[idx])
        for idx in np.where(self.annotations == 0)[0]:
            if idx not in self.neg_idxs:
                self.neg_idxs.append(int(idx))
                self.neg_scores.append(self.features @ self.features[idx])

        self._cache(self.hyperparams["dot_products_cache"])

        return CallResult(None)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """return ranking of unlabeled instances based on similarity to positively and negatively
        labeled instances

        Ex.
            d3mIndex       score
                1130    0.586983
                11      0.469862
                1077    0.394225
                1125    0.355335
                21      0.353363

        Arguments:
            inputs {Inputs} -- ignores these `inputs`, uses `inputs` from `set_training_data()`

        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, not considered (default: {None})
        """

        pos_scores = np.row_stack(self.pos_scores)
        pos_scores = self._conditional_gem(pos_scores)

        if len(self.neg_scores) >= self.hyperparams["denominator_min"]:
            neg_scores = np.row_stack(self.neg_scores)
            neg_scores = self._conditional_gem(neg_scores)
            scores = pos_scores / (neg_scores + 1e-12)
        else:
            scores = pos_scores

        mis_scores = scores[self.mis_idxs]
        mis_ranks = self.mis_idxs[np.argsort(-mis_scores)]
        mis_ranks = self.d3m_idxs[mis_ranks]

        ranking_df = pd.DataFrame(
            {
                self.idx_name: mis_ranks,
                "score": np.flip(np.sort(mis_scores)),
            }
        )
        ranking_df = d3m_DataFrame(ranking_df, generate_metadata=True)

        ranking_df.metadata = ranking_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0), "http://schema.org/Integer"
        )
        ranking_df.metadata = ranking_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
        )
        ranking_df.metadata = ranking_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1), "http://schema.org/Float"
        )
        return CallResult(ranking_df)

    def _postprocess(self, features: np.ndarray) -> np.ndarray:
        """ postprocess feature vectors """
        features = self._normalize(features)
        features = self._reduce(features)
        features = self._normalize(features)
        return features
    # runs gem in linear or in parallel depending on dimension
    def _conditional_gem(self, scores: np.ndarray, num_of_threads:int = 4) -> np.ndarray:
        _, num_of_columns = scores.shape
        p=self.hyperparams["gem_p"]
        if num_of_columns < self.linear_threshold:
            return linear_gem(scores, p)
        if num_of_columns < num_of_threads:
            num_of_threads = num_of_columns
        bucket_size = math.ceil(num_of_columns / num_of_threads)
        thread_pool = [Thread] * num_of_threads
        result = [None] * num_of_threads
        for i in range(num_of_threads):
            start = i * bucket_size
            end = min(start + bucket_size, num_of_columns)
            thread_pool[i] = Thread(target=thread_helper_gem, args=(result, i, scores, start, end, p))
            thread_pool[i].start()
        # join all threads
        for thread in thread_pool:
            thread.join()
        output=[]
        for i in range(len(result)):
            output=np.concatenate([output,result[i]], axis=0)
        return output
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """ L2 normalize features """
        return features / np.sqrt((features ** 2).sum(axis=-1, keepdims=True))

    def _reduce(self, features: np.ndarray) -> np.ndarray:
        """ reduce dimensions of feature vectors """
        n_components = min(
            self.hyperparams["reduce_dimension"], features.shape[0], features.shape[1]
        )
        if self.hyperparams["reduce_method"] == "pca":
            reduce_method = PCA(
                n_components=n_components, whiten=True, random_state=self.random_seed
            )

        elif self.hyperparams["reduce_method"] == "svd":
            reduce_method = TruncatedSVD(n_components=n_components)

        return reduce_method.fit_transform(features)

    def _attempt_load(self, cache_dir: str) -> None:
        """ attempt to load cached dot products """

        if os.path.isdir(cache_dir):

            pos_idxs_f = os.path.join(cache_dir, "pos_idxs.pkl")
            pos_scores_f = os.path.join(cache_dir, "pos_scores.pkl")
            if os.path.isfile(pos_idxs_f):
                self.pos_idxs = pickle.load(open(pos_idxs_f, "rb"))
                self.pos_scores = pickle.load(open(pos_scores_f, "rb"))

            neg_idxs_f = os.path.join(cache_dir, "neg_idxs.pkl")
            neg_scores_f = os.path.join(cache_dir, "neg_scores.pkl")
            if os.path.isfile(neg_idxs_f):
                self.neg_idxs = pickle.load(open(neg_idxs_f, "rb"))
                self.neg_scores = pickle.load(open(neg_scores_f, "rb"))

    def _cache(self, cache_dir: str) -> None:
        """ cache computed dot products """

        if len(self.pos_idxs):
            pos_idxs_f = os.path.join(cache_dir, "pos_idxs.pkl")
            pos_scores_f = os.path.join(cache_dir, "pos_scores.pkl")
            pickle.dump(self.pos_idxs, open(pos_idxs_f, "wb"))
            pickle.dump(self.pos_scores, open(pos_scores_f, "wb"))

        if len(self.neg_idxs):
            neg_idxs_f = os.path.join(cache_dir, "neg_idxs.pkl")
            neg_scores_f = os.path.join(cache_dir, "neg_scores.pkl")
            pickle.dump(self.neg_idxs, open(neg_idxs_f, "wb"))
            pickle.dump(self.neg_scores, open(neg_scores_f, "wb"))
