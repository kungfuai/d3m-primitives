from tslearn.clustering import TimeSeriesKMeans, GlobalAlignmentKernelKMeans
from tslearn.metrics import sigma_gak, cdist_gak


class KMeans:
    def __init__(
        self, n_clusters, algorithm="GlobalAlignmentKernelKMeans", random_seed=0
    ):
        """
        initialize KMeans clustering model with specific kernel

        hyperparameters:
            n_clusters:         number of clusters in Kmeans model
            algorithm:          which kernel to use for model, options
                                are 'GlobalAlignmentKernelKMeans' and 'TimeSeriesKMeans'
            random_seed:        random seed with which to initialize Kmeans
        """
        try:
            assert (
                algorithm == "GlobalAlignmentKernelKMeans"
                or algorithm == "TimeSeriesKMeans"
            )
        except:
            raise ValueError(
                "algorithm must be one of 'GlobalAlignmentKernelKMeans' or 'TimeSeriesKMeans'"
            )
        self.n_clusters = n_clusters
        self.random_seed = random_seed
        self.algorithm = algorithm
        self.km = None

    def fit(self, train):
        """
        fit KMeans clustering model on training data

        parameters:
            train                : training time series
        """

        if self.algorithm == "TimeSeriesKMeans":
            self.km = TimeSeriesKMeans(
                n_clusters=self.n_clusters,
                n_init=20,
                verbose=True,
                random_state=self.random_seed,
            )
        else:
            self.km = GlobalAlignmentKernelKMeans(
                n_clusters=self.n_clusters,
                sigma=sigma_gak(train),
                n_init=20,
                verbose=True,
                random_state=self.random_seed,
            )
        return self.km.fit_predict(train)

    def predict(self, test):
        """
        clusters for time series in test data set

        parameters:
            test:     test time series on which to predict clusters

        returns: clusters for test data set
        """
        return self.km.predict(test)