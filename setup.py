from setuptools import setup, find_packages

setup(
    name="kf-d3m-primitives",
    version="0.6.0",
    description="All Kung Fu D3M primitives as a single library",
    license="Apache-2.0",
    packages=find_packages(exclude="scripts"),
    setkeywords=["d3m_primitive"],
    install_requires=[
        "d3m",
        "torch>=1.4.0",
        "pillow==9.0.0",
        "tslearn==0.4.1",
        "statsmodels==0.11.1",
        "pmdarima==1.6.1",
        "hdbscan==0.8.26",
        "requests==2.23.0",
        "shap==0.37.0",
        "torchvision>=0.5.0",
        "opencv-python-headless==4.1.1.26",
        "gluonts==0.5.2",
        "albumentations==0.4.6",
        "tifffile==2020.8.13",
        "tqdm==4.48.2",
        "segmentation-models-pytorch==0.1.3",
        "lz4==3.1.3",
        "faiss-cpu==1.7.0",
        "punk @ git+https://github.com/uncharted-distil/punk@8b101eca26b5f9a3df2a65aab2733bd404965578#egg=punk",
        "object_detection_retinanet @ git+https://github.com/uncharted-distil/object-detection-retinanet@485db6681ac98bf56f02cf681efbdcb004a5cfb5#egg=object_detection_retinanet",
        "Simon @ git+https://github.com/uncharted-distil/simon@ff2fa7e963653b9c42ed6d7ecb53d7e37b191670#egg=Simon",
        "nk_sent2vec @ git+https://github.com/uncharted-distil/nk-sent2vec@8face221f30c523261f60585f2344252d981c822#egg=nk_sent2vec",
        "duke @ git+https://github.com/uncharted-distil/duke@627912e23685d058c6becc4e4615a7d3e8c93b93#egg=duke",
        "rsp @ git+https://github.com/cfld/rs_pretrained@92d832efe1961d6a06011f689dad7ef2481a64b1#egg=rsp",
    ],
    extras_require={
        "cpu": ["tensorflow==2.2.0", "mxnet==1.6.0"],
        "gpu-cuda-10.1": ["tensorflow-gpu==2.2.0", "mxnet-cu101mkl==1.6.0.post0"],
        "gpu-cuda-9.2": ["tensorflow-gpu==2.2.0", "mxnet-cu92mkl==1.6.0.post0"],
    },
    entry_points={
        "d3m.primitives": [
            "data_cleaning.column_type_profiler.Simon = kf_d3m_primitives.data_preprocessing.data_typing.simon:SimonPrimitive",
            "data_cleaning.geocoding.Goat_forward = kf_d3m_primitives.data_preprocessing.geocoding_forward.goat_forward:GoatForwardPrimitive",
            "data_cleaning.geocoding.Goat_reverse = kf_d3m_primitives.data_preprocessing.geocoding_reverse.goat_reverse:GoatReversePrimitive",
            "feature_extraction.nk_sent2vec.Sent2Vec = kf_d3m_primitives.natural_language_processing.sent2vec.sent2vec:Sent2VecPrimitive",
            "clustering.k_means.Sloth = kf_d3m_primitives.clustering.k_means.Storc:StorcPrimitive",
            "clustering.hdbscan.Hdbscan = kf_d3m_primitives.clustering.hdbscan.Hdbscan:HdbscanPrimitive",
            "clustering.spectral_graph.SpectralClustering = kf_d3m_primitives.clustering.spectral_clustering.spectral_clustering:SpectralClusteringPrimitive",
            "dimensionality_reduction.t_distributed_stochastic_neighbor_embedding.Tsne = kf_d3m_primitives.dimensionality_reduction.tsne.Tsne:TsnePrimitive",
            "time_series_classification.k_neighbors.Kanine = kf_d3m_primitives.ts_classification.knn.kanine:KaninePrimitive",
            "time_series_forecasting.vector_autoregression.VAR = kf_d3m_primitives.ts_forecasting.vector_autoregression.var:VarPrimitive",
            "time_series_classification.convolutional_neural_net.LSTM_FCN = kf_d3m_primitives.ts_classification.lstm_fcn.lstm_fcn:LstmFcnPrimitive",
            "time_series_forecasting.lstm.DeepAR = kf_d3m_primitives.ts_forecasting.deep_ar.deepar:DeepArPrimitive",
            "time_series_forecasting.feed_forward_neural_net.NBEATS = kf_d3m_primitives.ts_forecasting.nbeats.nbeats:NBEATSPrimitive",
            "object_detection.retina_net.ObjectDetectionRN = kf_d3m_primitives.object_detection.retinanet.object_detection_retinanet:ObjectDetectionRNPrimitive",
            "data_cleaning.data_cleaning.Datacleaning = kf_d3m_primitives.data_preprocessing.data_cleaning.data_cleaning:DataCleaningPrimitive",
            "data_cleaning.text_summarization.Duke = kf_d3m_primitives.data_preprocessing.text_summarization.duke:DukePrimitive",
            "feature_selection.pca_features.Pcafeatures = kf_d3m_primitives.feature_selection.pca_features.pca_features:PcaFeaturesPrimitive",
            "feature_selection.rffeatures.Rffeatures = kf_d3m_primitives.feature_selection.rf_features.rf_features:RfFeaturesPrimitive",
            "remote_sensing.remote_sensing_pretrained.RemoteSensingPretrained = kf_d3m_primitives.remote_sensing.featurizer.remote_sensing_pretrained:RemoteSensingPretrainedPrimitive",
            "remote_sensing.mlp.MlpClassifier = kf_d3m_primitives.remote_sensing.classifier.mlp_classifier:MlpClassifierPrimitive",
            "similarity_modeling.iterative_labeling.ImageRetrieval = kf_d3m_primitives.remote_sensing.image_retrieval.image_retrieval:ImageRetrievalPrimitive",
            "remote_sensing.convolutional_neural_net.ImageSegmentation = kf_d3m_primitives.remote_sensing.segmentation.image_segmentation:ImageSegmentationPrimitive",
            "semisupervised_classification.iterative_labeling.CorrectAndSmooth = kf_d3m_primitives.semi_supervised.correct_and_smooth.correct_and_smooth:CorrectAndSmoothPrimitive",
            "semisupervised_classification.iterative_labeling.TabularSemiSupervised = kf_d3m_primitives.semi_supervised.tabular_semi_supervised.tabular_semi_supervised:TabularSemiSupervisedPrimitive",
        ],
    },
)
