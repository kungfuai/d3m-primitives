import sys
import os.path
import logging

import numpy as np
import pandas as pd
import shap
from sklearn.cluster import KMeans

logger = logging.getLogger(__name__)


class Tree:
    """
    Primitive that takes a dataset and a fitted tree-based model and returns explanations.
    For global explanations returns a tuple containing the datatset that explanations were made on, and the global explanations
    For sample explnations, returns a list of the explanations for the samples provided

    Parameters
    ----------
    model : model object
        The tree based machine learning model that we want to explain. XGBoost, LightGBM, CatBoost,
        and most tree-based scikit-learn models are supported.

    X :numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
        A matrix of samples (# samples x # features) on which to explain the model’s output.
        (catboost not yet supported)

    number_of_features : int, (default = 5)
        Function will return explanations for the top K number of features

    model_type : None (default), or 'Random_Forest'
        If 'Random_Forest' then global explanation is calculated on a sub-sample of the full dataset

    task_type : 'regression' (default) or 'classification'
        Is the tree model a regressor or a classifier

    max_dataset_size : int, (default = 1500)
        The maximum dataset size on which to apply SHAP interpretation to each sample individually. Otherwise, this number of samples will be
        drawn from the data distribution after clustering (to approximate the distribution) and interpretation will only be applied to these
        samples
    """

    def __init__(
        self,
        model,
        X,
        number_of_features=None,
        model_type=None,
        task_type="regression",
        max_dataset_size=1500,
    ) -> None:

        self.X = X
        self.model = model
        self.number_of_features = number_of_features
        self.model_type = model_type
        self.task_type = task_type
        self.explainer = shap.TreeExplainer(self.model)
        self.max_dataset_size = max_dataset_size

    def _get_data_sample(self):

        """
        Sub-samples the dataframe to provide a smaller, balanced dataframe to make global explainer


        """

        df = self.X.copy()
        df["cluster_assignment"] = KMeans().fit_predict(self.X).astype(int)

        n_classes = df["cluster_assignment"].unique()

        # deal with cases in which the predictions are all one class
        if len(n_classes) == 1:
            return df.sample(self.max_dataset_size).drop(columns=["cluster_assignment"])

        else:
            proportion = round(self.max_dataset_size / len(n_classes))

            dfs = []

            for i in n_classes:
                # dealing with classes that have less than or equal to their proportional representation
                if df[df["cluster_assignment"] == i].shape[0] <= proportion:
                    dfs.append(df[df["cluster_assignment"] == i])
                else:
                    dfs.append(df[df["cluster_assignment"] == i].sample(proportion))

            sub_sample_df = pd.concat(dfs)

            return sub_sample_df.drop(columns=["cluster_assignment"])

    def _get_top_features(self, shap_values, number_of_features):

        df = pd.DataFrame(shap_values, columns=self.X.columns)
        cols = abs(df.mean()).sort_values(ascending=False)[:number_of_features].index
        sorter = np.argsort(self.X.columns)
        imp_features = sorter[np.searchsorted(self.X.columns, cols, sorter=sorter)]

        keep = []
        for row in shap_values:
            keep.append(row[imp_features])
        keep_list = [k.tolist() for k in keep]

        return imp_features, keep_list

    def produce_sample(self, samples):

        """
        Returns a dataframe of the shapley values for the given samples

        """

        ##restrict features to most important

        shap_values = self.explainer.shap_values(self.X.iloc[samples])

        if (self.task_type == "classification") & (self.model_type == "Random_Forest"):

            probs = self.explainer.expected_value
            idx = probs.index(max(probs))

            return pd.DataFrame(
                shap_values[idx],
                columns=self.X.columns,
                index=self.X.iloc[samples].index,
            )

        else:
            return pd.DataFrame(
                shap_values, columns=self.X.columns, index=self.X.iloc[samples].index
            )

    def produce_global(self, approximate=False):
        """
        Returns a dataframe of the shap values for each feature
        This will be a downsampled dataframe for random forest models on datasets with more samples than self.max_dataset_size

        If the task_type is classification and the model_type is random forest, the returned interpretability values
            will be offset from the most frequent class in the dataset

        approximation : boolean, (default = False)
            Whether to calculate SHAP interpretability values using the Saabas approximation. This approximation samples over only one
            permuation of feature values for each sample - that defined by the path along the decision tree. Thus, this approximation suffers
            from inconsistency, which means that 'a model can change such that it relies more on a given feature, yet the importance estimate
            assigned to that feature decreases' (Lundberg et al. 2019). Specifically, it will place too much weight on lower splits in the tree.
        """

        if approximate:
            logger.warning(
                f"SHAP interpretability values are being calculated using the Saabas approximation. This approximation samples over only one "
                + "permuation of feature values for each sample - that defined by the path along the decision tree. Thus, this approximation suffers "
                + "from inconsistency, which means that 'a model can change such that it relies more on a given feature, yet the importance estimate "
                + "assigned to that feature decreases' (Lundberg et al. 2019). Specifically, it will place too much weight on lower splits in the tree."
            )

        if (self.model_type == "Random_Forest") & (
            self.X.shape[0] > self.max_dataset_size
        ):
            logger.warning(
                f"More than {self.max_dataset_size} rows in dataset, sub-sampling {self.max_dataset_size} approximately representative rows"
                + "(approximation from clustering with KMeans) on which to produce interpretations"
            )
            df = self._get_data_sample()
            shap_values = self.explainer.shap_values(df, approximate=approximate)
            df.sort_index(inplace=True)

        else:
            df = self.X
            shap_values = self.explainer.shap_values(df, approximate=approximate)

        if (self.task_type == "classification") & (self.model_type == "Random_Forest"):
            logger.info(
                f"Returning interpretability values offset from most frequent class in dataset"
            )
            shap_values = shap_values[np.argmax(self.explainer.expected_value)]

        # if self.number_of_features:
        #   features, shap_values = self._get_top_features(shap_values, self.number_of_features)
        #  return (sub_sample.iloc[:,features], shap_values)

        return pd.DataFrame(shap_values, columns=df.columns, index=df.index)
