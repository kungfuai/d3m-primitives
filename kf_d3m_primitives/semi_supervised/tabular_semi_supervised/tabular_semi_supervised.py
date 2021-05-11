import os.path
import logging
import typing

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, params, base as metadata_base
from d3m.exceptions import PrimitiveNotFittedError
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

from .mlp import Mlp
from .tabular_ds import TabularDs
from .cross_entropy import CrossEntropy
from .ssl_algorithms import PseudoLabel, VAT, ICT

__author__ = "Distil"
__version__ = "1.0.0"
__contact__ = "mailto:cbethune@uncharted.software"

Inputs = container.DataFrame
Outputs = container.DataFrame

logger = logging.getLogger(__name__)


class Params(params.Params):
    is_fit: bool
    output_column: str
    label_encoder: LabelEncoder


class Hyperparams(hyperparams.Hyperparams):
    algorithm = hyperparams.Enumeration(
        default="PseudoLabel",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["PseudoLabel", "VAT", "ICT"],
        description="which semi-supervised algorithm to use",
    )
    batch_size = hyperparams.UniformInt(
        lower=1,
        upper=512,
        default=64,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="training and inference batch size",
    )
    batch_size_labeled_prop = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.05,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="training and inference batch size",
    )
    semi_supervised_loss_weight = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=1.0,
        upper_inclusive=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="training and inference batch size",
    )
    epochs = hyperparams.UniformInt(
        lower=0,
        upper=1000,
        default=100,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="how many epochs for which to finetune classification head (happens first)",
    )
    learning_rate = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.003,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/TuningParameter"
        ],
        description="learning rate",
    )
    augmentation_weak = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="learning rate",
    )
    augmentation_strong = hyperparams.Uniform(
        lower=0.0,
        upper=1.0,
        default=0.2,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="learning rate",
    )
    weights_filepath = hyperparams.Hyperparameter[str](
        default="model_weights.pth",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="weights of trained model will be saved to this filepath",
    )
    all_scores = hyperparams.UniformBool(
        default=False,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to return scores for all classes from produce method",
    )


class TabularSemiSupervisedPrimitive(
    SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]
):
    """This primitive applies one of three methods (configurable through hyperparameters) for
    semi-supervised learning on tabular data. The three methods are PseudoLabel, VAT, and ICT.

    PseudoLabel applies a threshold on the model outputs of unlabeled, augmented data to assign labels.
    It then uses these pseudo-labels to train a classifier using the "consistency" between the
    pseudo-labeled data and model outputs of augmented versions of the pseudo-labeled data.

    VAT uses gradient descent to find the augmentation of unlabeled data that improves the classifier
    the most. It then trains a classifier using the "consistency" between the model outputs on this
    augmentation and the model outputs on a random augmentation.

    ICT uses mixup (random combinations of model outputs) on unlabeled data and trains a classifier
    using the "consistency" between these mixed-up examples.
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            "id": "4c0f4d85-16ce-4b2d-964f-883d61a23ced",
            "version": __version__,
            "name": "TabularSemiSupervised",
            "keywords": [
                "semi-supervised",
                "pseudo-labels",
                "tabular",
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
            "python_path": "d3m.primitives.semisupervised_classification.iterative_labeling.TabularSemiSupervised",
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

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        np.random.seed(random_seed)
        torch.manual_seed(random_seed + 111)
        if self.device == "cuda:0":
            torch.cuda.manual_seed(random_seed + 222)

        if (
            self.hyperparams["augmentation_weak"]
            > self.hyperparams["augmentation_strong"]
        ):
            raise ValueError(
                "'augmentation_strong' HP must be greater than 'augmentation_weak' HP"
            )

        self._is_fit = False

    def get_params(self) -> Params:
        return Params(
            is_fit=self._is_fit,
            output_column=self.output_column,
            label_encoder=self.label_encoder,
        )

    def set_params(self, *, params: Params) -> None:
        self._is_fit = params["is_fit"]
        self.output_column = params["output_column"]
        self.label_encoder = params["label_encoder"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """set primitive's training data.

        Arguments:
            inputs {Inputs} -- D3M dataframe containing features
            outputs {Outputs} -- D3M dataframe containing labels

        """

        X = inputs.astype(np.float32).values
        X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)

        idx_labeled = np.where(outputs.values != "")[0]
        idx_unlabeled = np.where(outputs.values == "")[0]
        X_labeled = X[idx_labeled]
        X_unlabeled = X[idx_unlabeled]

        y_labeled = outputs.values[idx_labeled].flatten()
        self.label_encoder = LabelEncoder()
        y_labeled = self.label_encoder.fit_transform(y_labeled)
        n_class = len(self.label_encoder.classes_)

        self.mlp_model = Mlp(X.shape[1], n_class, 128)
        self.labeled_loader, self.unlabeled_loader = self._create_datasets(
            X_labeled, y_labeled, X_unlabeled
        )

        self.output_column = outputs.columns[0]
        self._is_fit = False

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """Fits semi-supervised learning algorithm

        Keyword Arguments:
            timeout {float} -- timeout, considered (default: {None})
            iterations {int} -- iterations, considered (default: {None})

        Returns:
            CallResult[None]
        """
        if self.hyperparams["algorithm"] == "PseudoLabel":
            self.ssl_algorithm = PseudoLabel()
        elif self.hyperparams["algorithm"] == "VAT":
            self.ssl_algorithm = VAT()
        else:
            self.ssl_algorithm = ICT()

        self.loss = CrossEntropy()
        f_aug_w = (
            lambda x: x + torch.randn_like(x) * self.hyperparams["augmentation_weak"]
        )
        f_aug_s = (
            lambda x: x + torch.randn_like(x) * self.hyperparams["augmentation_strong"]
        )

        self.mlp_model = self.mlp_model.to(self.device)
        opt = optim.Adam(
            self.mlp_model.parameters(), lr=self.hyperparams["learning_rate"]
        )

        self._train(f_aug_w, f_aug_s, opt)
        self._is_fit = True
        torch.save(self.mlp_model.state_dict(), self.hyperparams["weights_filepath"])

        return CallResult(None)

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """produce predictions

        Arguments:
            inputs {Inputs} -- D3M dataframe containing images

        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, not considered (default: {None})
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        X = inputs.astype(np.float32).values
        X = (X - X.mean(0, keepdims=True)) / X.std(0, keepdims=True)

        n_class = len(self.label_encoder.classes_)
        mlp_model = Mlp(X.shape[1], n_class, 128).to(self.device)
        mlp_model.load_state_dict(torch.load(self.hyperparams["weights_filepath"]))
        mlp_model.eval()

        dataset = TensorDataset(torch.Tensor(X))
        loader = DataLoader(dataset, 64, shuffle=False, num_workers=10)

        all_logits = []
        for data in loader:
            logits = mlp_model(data[0].to(self.device))
            all_logits.append(logits)
        all_logits = torch.cat(all_logits).detach().cpu().numpy()

        preds_df = self._prepare_d3m_df(all_logits, n_class)
        return CallResult(preds_df)

    def _create_datasets(self, X_labeled, y_labeled, X_unlabeled):
        bs_l = int(
            self.hyperparams["batch_size_labeled_prop"] * self.hyperparams["batch_size"]
        )
        bs_u = int(self.hyperparams["batch_size"] - bs_l)

        iters_l = X_labeled.shape[0] / bs_l
        iters_u = np.ceil(X_unlabeled.shape[0] / bs_u)
        rel_l = int(max(iters_u, iters_l) / min(iters_u, iters_l))

        labeled_dataset = TabularDs(
            data=X_labeled, labels=y_labeled, relative_len=rel_l
        )
        unlabeled_dataset = TabularDs(data=X_unlabeled, relative_len=1)
        labeled_loader = DataLoader(labeled_dataset, bs_l, shuffle=True, num_workers=10)
        unlabeled_loader = DataLoader(
            unlabeled_dataset, bs_u, shuffle=True, num_workers=10
        )

        return labeled_loader, unlabeled_loader

    def _train(self, f_aug_w, f_aug_s, opt):
        for epoch in tqdm(range(self.hyperparams["epochs"])):
            self.mlp_model.train()
            for i, (labeled, X_u) in enumerate(
                zip(self.labeled_loader, self.unlabeled_loader)
            ):
                X_l, y_l = labeled
                X_l, y_l = X_l.to(self.device), y_l.to(self.device)
                X_u = X_u.to(self.device)

                X_u_w, X_u_s = f_aug_w(X_u), f_aug_s(X_u)

                # forward pass on all data
                all_data = torch.cat([X_l, X_u_w, X_u_s])

                # labeled grad / loss
                outputs = self.mlp_model(all_data)
                logits_l = outputs[: X_l.shape[0]]
                loss_l = F.cross_entropy(logits_l, y_l)

                # unlabeled grad / loss
                logits_u_w, logits_u_s = torch.chunk(outputs[X_l.shape[0] :], 2, dim=0)

                y_hat_u, y_u, mask = self.ssl_algorithm(
                    y_hat_s=logits_u_s,
                    y_hat_w=logits_u_w.detach(),
                    w_data=X_u_w,
                    s_data=X_u_s,
                    student_f=self.mlp_model.forward,
                )

                loss_u = self.loss(y_hat_u, y_u, mask)
                loss = loss_l + (
                    self.hyperparams["semi_supervised_loss_weight"] * loss_u
                )
                opt.zero_grad()
                loss.backward()
                opt.step()

    def _prepare_d3m_df(self, logits, n_class):
        """ prepare d3m dataframe with appropriate metadata """

        if self.hyperparams["all_scores"]:
            index = np.repeat(range(len(logits)), n_class)
            labels = np.tile(range(n_class), len(logits))
            scores = logits.flatten()
        else:
            index = None
            labels = np.argmax(logits, -1)
            scores = logits[range(len(labels)), labels]

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