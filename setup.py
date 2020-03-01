from setuptools import setup, find_packages

setup(
    name="yonder-primitives",
    version="2.0.0",
    description="All Yonder primitives as a single library",
    packages=find_packages(),
    keywords=['d3m_primitive'],
    install_requires=[
        "numpy>=1.15.4,<=1.17.3",
        "scipy>=1.2.1,<=1.3.1",
        "scikit-learn[alldeps]>=0.20.3,<=0.21.3",
        "pandas>=0.23.4,<=0.25.2",
        "tensorflow-gpu == 2.0.0",
        "tslearn == 0.2.5",
        "statsmodels==0.10.2",
        "pmdarima==1.0.0",
        "punk==3.0.0",
        "deepar @ git+https://github.com/NewKnowledge/deepar@c801332d26742c17c4265d2155372ce7f1192bc4#egg=deepar-0.0.2",
        "object_detection_retinanet @ git+https://github.com/NewKnowledge/object-detection-retinanet@beca7ff86faa2295408e46fe221a3c7437cfdc81#egg=object_detection_retinanet",
    ],
    entry_points={
        "d3m.primitives": [
            "time_series_classification.k_neighbors.Kanine = primitives.ts_classification.knn.classification_knn:KaninePrimitive",
            "time_series_forecasting.vector_autoregression.VAR = primitives.ts_forecasting.vector_autoregression.forecasting_var:VarPrimitive",
            "time_series_classification.convolutional_neural_net.LSTM_FCN = primitives.ts_classification.lstm_fcn.classification_lstm:LstmFcnPrimitive",
            "time_series_forecasting.lstm.DeepAR = primitives.ts_forecasting.deep_ar.forecasting_deepar:DeepArPrimitive",
            "object_detection.retinanet = primitives.object_detection.retinanet.object_detection_retinanet:ObjectDetectionRNPrimitive",
            "data_cleaning.data_cleaning.Datacleaning = primitives.data_preprocessing.data_cleaning.data_cleaning:DataCleaningPrimitive",
            "data_cleaning.text_summarization.Duke = primitives.data_preprocessing.duke.duke:DukePrimitive",
            "feature_selection.pca_features.Pcafeatures = primitives.feature_selection.pca_features.pca_features:PcaFeaturesPrimitive",
            "feature_selection.rffeatures.Rffeatures = primitives.feature_selection.rf_features.rf_features:RfFeaturesPrimitive"
        ],
    },
)
