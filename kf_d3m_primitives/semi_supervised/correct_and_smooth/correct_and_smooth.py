import os.path
import logging
import typing

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import faiss
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.svm import LinearSVC
from sklearn.preprocessing import LabelEncoder

__author__ = "Distil"
__version__ = "1.0.0"
__contact__ = "mailto:cbethune@uncharted.software"

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)


class Params(params.Params):
    is_fit: bool
    X_train: np.ndarray
    idx_train: np.ndarray
    y_train: np.ndarray
    output_column: str
    clf: LinearSVC
    label_encoder: LabelEncoder


class Hyperparams(hyperparams.Hyperparams):
    k = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=10,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="number of neighbors to use when constructing k-NN adjacency matrix",
    )
    alpha = hyperparams.Uniform(
        lower=0,
        upper=1,
        default=0.85,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="controls step size during label propagations",
    )
    n_iterations = hyperparams.UniformInt(
        lower=10,
        upper=100,
        default=50,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="number of iterations during label propagations",
    )
    all_scores = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to return scores for all classes from produce method",
    )
    normalize_features = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to L2 normalize feature vectors",
    )


class CorrectAndSmoothPrimitive(
    SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]
):
    """This primitive applies the "Correct and Smooth" procedure for semi-supervised learning
    (https://arxiv.org/pdf/2010.13993.pdf). It combines a simple classification model with
    two label propagation post-processing steps - one that spreads residual errors and one
    that smooths predictions.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "8372bb87-5894-4dcc-bf9f-dcc60387b7bf",
            "version": __version__,
            "name": "CorrectAndSmooth",
            "keywords": [
                "semi-supervised",
                "label propagation",
                "graph structure",
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
            "python_path": "d3m.primitives.semisupervised_classification.iterative_labeling.CorrectAndSmooth",
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.ITERATIVE_LABELING,
            ],
            "primitive_family": metadata_base.PrimitiveFamily.SEMISUPERVISED_CLASSIFICATION,
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

        self._is_fit = False

    def get_params(self) -> Params:
        return Params(
            is_fit=self._is_fit,
            X_train=self.X_train,
            idx_train=self.idx_train,
            y_train=self.y_train,
            output_column=self.output_column,
            clf=self.clf,
            label_encoder=self.label_encoder,
        )

    def set_params(self, *, params: Params) -> None:
        self._is_fit = params["is_fit"]
        self.X_train = params["X_train"]
        self.idx_train = params["idx_train"]
        self.y_train = params["y_train"]
        self.output_column = params["output_column"]
        self.clf = params["clf"]
        self.label_encoder = params["label_encoder"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """set primitive's training data.

        Arguments:
            inputs {Inputs} -- D3M dataframe containing features
            outputs {Outputs} -- D3M dataframe containing labels

        """

        X = inputs.astype(np.float32).values
        if self.hyperparams["normalize_features"]:
            X = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))

        self.idx_train = np.where(outputs.values != "")[0]
        self.X_train = X[self.idx_train]

        y_train = outputs.values[self.idx_train].flatten()
        self.label_encoder = LabelEncoder()
        self.y_train = self.label_encoder.fit_transform(y_train)

        self.output_column = outputs.columns[0]
        self._is_fit = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """Fits Linear SVC, smooths and corrects predictions with label propagation

        Keyword Arguments:
            timeout {float} -- timeout, considered (default: {None})
            iterations {int} -- iterations, considered (default: {None})

        Returns:
            CallResult[None]
        """

        self.clf = LinearSVC().fit(self.X_train, self.y_train)
        self._is_fit = True
        return CallResult(None)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """produce corrected and smoothed predictions

        Arguments:
            inputs {Inputs} -- D3M dataframe containing images

        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, not considered (default: {None})
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        X = inputs.astype(np.float32).values
        if self.hyperparams["normalize_features"]:
            X = X / np.sqrt((X ** 2).sum(axis=-1, keepdims=True))
        X, idx_train = self._compare_train_rows(X)
        X = np.ascontiguousarray(X)

        S, AD = self._make_adj_matrix(X)
        n_class = len(self.label_encoder.classes_)

        Z_orig = self._get_initial_predictions(X, n_class)
        Y_resid = self._get_residuals(Z_orig, idx_train, n_class)
        Z_corrected = self._spread_residuals(Z_orig, Y_resid, AD, idx_train)
        Z_smoothed = self._smooth_predictions(Z_corrected, S, idx_train)

        preds_df = self._prepare_d3m_df(Z_smoothed, n_class)
        return CallResult(preds_df)

    def _compare_train_rows(self, X):
        """ compare train rows against test set; add train rows if necessary"""

        if self.idx_train.max() > X.shape[0]:
            X = np.vstack((self.X_train, X))
            idx_train = np.arange(self.X_train.shape[0])
            self.test_dataset = True

        else:
            train_rows = X[self.idx_train]
            if not np.array_equal(train_rows, self.X_train):
                X = np.vstack((self.X_train, X))
                idx_train = np.arange(self.X_train.shape[0])
                self.test_dataset = True
            else:
                idx_train = self.idx_train
                self.test_dataset = False

        return X, idx_train

    def _make_adj_matrix(self, features):
        """ make normalized adjacency matrix from features """

        n_obs = features.shape[0]

        findex = faiss.IndexFlatL2(features.shape[1])
        findex.add(features)
        _, I = findex.search(features, k=self.hyperparams["k"])

        row = np.arange(n_obs).repeat(self.hyperparams["k"])
        col = I.ravel()
        val = np.ones(self.hyperparams["k"] * n_obs)

        adj = sp.csr_matrix((val, (row, col)), shape=(n_obs, n_obs))
        adj = (adj + adj.T).astype(np.float)

        # Compute normalization matrix
        D = np.asarray(adj.sum(axis=0)).squeeze()
        Dinv = sp.diags(1 / D)

        # Compute normalized adjacency matrices
        S = np.sqrt(Dinv) @ adj @ np.sqrt(Dinv)
        AD = adj @ Dinv

        return S, AD

    def _get_initial_predictions(self, X, n_class):
        """ get initial predictions from Linear SVC"""

        Z_orig = self.clf.decision_function(X)

        if n_class == 2:
            Z_orig = 1 / (1 + np.exp(Z_orig))
            Z_orig = np.column_stack([1 - Z_orig, Z_orig])
        else:
            Z_orig = np.exp(Z_orig) / np.exp(Z_orig).sum(axis=-1, keepdims=True)

        return Z_orig

    def _get_residuals(self, Z_orig, idx_train, n_class):
        """ get residuals from original classifier"""

        Y_resid = np.zeros((Z_orig.shape[0], n_class))
        Y_resid[(idx_train, self.y_train)] = 1
        Y_resid[idx_train] -= Z_orig[idx_train]
        return Y_resid

    def _label_propagation(self, adj, labels, clip=(0, 1)):
        """ propagate labels for n_iterations"""

        Z = labels.copy()
        for _ in range(self.hyperparams["n_iterations"]):
            Z = self.hyperparams["alpha"] * (adj @ Z)
            Z = Z + (1 - self.hyperparams["alpha"]) * labels
            Z = Z.clip(*clip)
        return Z

    def _spread_residuals(self, Z_orig, Y_resid, AD, idx_train):
        """ spread residuals with label propagation"""

        resid = self._label_propagation(AD, Y_resid, clip=(-1, 1))

        num = np.abs(Y_resid[idx_train]).sum() / idx_train.shape[0]
        denom = np.abs(resid).sum(axis=-1, keepdims=True)
        scale = num / denom
        scale[denom == 0] = 1
        scale[scale > 1000] = 1

        Z_corrected = Z_orig + scale * resid
        return Z_corrected

    def _smooth_predictions(self, Z_corrected, S, idx_train):
        """ smooth predictions with label propagation"""

        Y_corrected = Z_corrected.copy()
        Y_corrected[idx_train] = 0
        Y_corrected[(idx_train, self.y_train)] = 1

        Z_smoothed = self._label_propagation(S, Y_corrected, clip=(0, 1))
        return Z_smoothed

    def _prepare_d3m_df(self, Z_smoothed, n_class):
        """ prepare d3m dataframe with appropriate metadata """

        if self.test_dataset:
            Z_smoothed = Z_smoothed[len(self.idx_train) :]

        if self.hyperparams["all_scores"]:
            index = np.repeat(range(len(Z_smoothed)), n_class)
            labels = np.tile(range(n_class), len(Z_smoothed))
            scores = Z_smoothed.flatten()
        else:
            index = None
            labels = np.argmax(Z_smoothed, -1)
            scores = Z_smoothed[range(len(labels)), labels]

        labels = self.label_encoder.inverse_transform(labels)

        preds_df = d3m_DataFrame(
            pd.DataFrame(
                {self.output_column: labels, "confidence": scores},
                index=index,
            ),
            generate_metadata=True,
        )

        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 0),
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        )
        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/Score",
        )
        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1),
            "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
        )
        preds_df.metadata = preds_df.metadata.add_semantic_type(
            (metadata_base.ALL_ELEMENTS, 1), "http://schema.org/Float"
        )

        return preds_df