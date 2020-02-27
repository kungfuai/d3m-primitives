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
        "deepar @ git+https://github.com/NewKnowledge/deepar@d06db4f6324ab8006c9b4703408ce3b6ae955cf4#egg=deepar-0.0.2",
    ],
    entry_points={
        "d3m.primitives": [
            "time_series_classification.k_neighbors.Kanine = primitives.ts_classification.knn.classification_knn:KaninePrimitive",
            "time_series_forecasting.vector_autoregression.VAR = primitives.ts_forecasting.vector_autoregression.forecasting_var:VarPrimitive",
            "time_series_classification.convolutional_neural_net.LSTM_FCN = primitives.ts_classification.lstm_fcn.classification_lstm:LstmFcnPrimitive",
            "time_series_forecasting.lstm.DeepAR = primitives.ts_forecasting.deep_ar.forecasting_deepar:DeepArPrimitive",
        ],
    },
)
