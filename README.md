# KUNGFU.AI TA1 D3M Primitives

This repository contains all of the primitives developed by the teams at KUNGFU.AI, Yonder, and New Knowledge for the [D3M](https://www.darpa.mil/program/data-driven-discovery-of-models) program. 

## Installation

kf-d3m-primitives requires Python 3.6, and the easiest
way to install it is via `pip`:

```bash
pip install kf-d3m-primitives
```

## Development

The latest versions of D3M datasets can be downloaded by running the following script from inside the cloned directory. [D3M Gitlab](https://gitlab.com/datadrivendiscovery/d3m) credentials are required. 

```bash
python download_datasets.py
```

To make a docker image with kf-d3m-primitives installed on top of the D3M program image run:

```bash
make build
```

To download the large static volumes that are necessary to run and test some of the primitives run:

```bash
make volumes
```

To run the image with the downloaded datasets and static volumes mounted run:

```bash
make run
```

## Tests

To test that each primitive's `produce` method, and, where applicable, its `set_training_data`, `fit`, `get_params`, and `set_params` methods can be called sucessfully within D3M pipelines, run the following command. This will also test that the predictions produced on test sets by each pipeline that can be scored by the D3M `runtime`. 

```bash
make test
```

## Submission

To generate `json` annotations for all primitives with the required directory structure for D3M submission run:

```bash
make annotations
```

To generate `yml.gz` pipeline run documents for all CPU-dependent pipelines with the required directory structure for D3M submission run:

```bash
make pipelines-cpu
```

To generate `yml.gz` pipeline run documents for all GPU-dependent pipelines with the required directory structure for D3M submission run:

```bash
make pipelines-gpu
```

## Primitives

### Data Preprocessing

1. **DataCleaningPrimitive**: wrapper of the data cleaning primitive based on the [punk](https://github.com/NewKnowledge/punk) library.

2. **DukePrimitive**: wrapper of the [Duke library](https://github.com/uncharted-distil/duke) in the D3M infrastructure.

3. **SimonPrimitive**: LSTM-FCN neural network trained on 18 different semantic types, which infers the semantic type of each column. Base library [here](https://github.com/uncharted-distil/simon)

4. **GoatForwardPrimitive**: geocodes names of locations into lat/long pairs with requests to [photon](https://github.com/komoot/photon) geocoding server (based on OpenStreetMap)

5. **GoatReversePrimitive**: geocodes lat/long pairs into geographic names of varying granularity with requests to [photon](https://github.com/komoot/photon) geocoding server (based on OpenStreetMap)

### Clustering

1. **HdbscanPrimitive**: wrapper of [HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html) and [DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)

2. **StorcPrimitive**: wrapper of [tslearn](https://tslearn.readthedocs.io/en/stable/index.html) 's kmeans implementations

3. **SpectralClustering**: wrapper of [Spectral Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html)

### Feature Selection

1. **PcaFeaturesPrimitive**: wrapper of the [Punk](https://github.com/NewKnowledge/punk) feature ranker into D3M infrastructure.

2. **RfFeaturesPrimitive** wrapper of the [Punk](https://github.com/NewKnowledge/punk) punk rrfeatures library into D3M infrastructure

### Dimensionality Reduction

1. **TsnePrimitive**: wrapper of [TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)

### Natural Language Processing

1. **Sent2VecPrimitive**: converts sentences into numerical feature representations. Base library [here](https://github.com/uncharted-distil/nk-sent2vec).

### Image Classification

1. **GatorPrimitive**: Inception V3 model pretrained on ImageNet finetuned for classification

### Object Detection

1. **ObjectDetectionRNPrimitive**: wrapper of the Keras implementation of Retinanet from [this repo](https://github.com/fizyr/keras-retinanet). The original Retinanet paper can be found [here](https://arxiv.org/abs/1708.02002).

### Time Series Classification

1. **KaninePrimitive**: wrapper of [KNeighborsTimeSeriesClassifier](https://tslearn.readthedocs.io/en/latest/gen_modules/neighbors/tslearn.neighbors.KNeighborsTimeSeriesClassifier.html#tslearn.neighbors.KNeighborsTimeSeriesClassifier)

2. **LstmFcnPrimitive**: wrapper of [LSTM Fully Convolutional Networks for Time Series Classification](https://github.com/houshd/LSTM-FCN)

### Time Series Forecasting

1. **DeepArPrimitive**: wrapper of [DeepAR](https://arxiv.org/abs/1704.04110) recurrent, autoregressive, probabilistic Time Series Forecasting algorithm based on [GluonTS](https://github.com/awslabs/gluon-ts) implementation.

2. **VarPrimitive**: wrapper of [VAR](https://www.statsmodels.org/dev/vector_ar.html) for multivariate time series and [auto_arima](http://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) for univariate time series

### Interpretability

**shap_explainers**: wrapper of Lundberg's shapley values implementation for tree models. Currently integrated into *d3m.primitives.learner.random_forest.DistilEnsembleForest* as *produce_shap_values()*

### Remote Sensing

1. **RemoteSensingPretrainedPrimitive**: featurizes remote sensing imagery using pre-trained models that were optimized with a self-supervised objective. There are two inference models that correspond to two pretext tasks: [Augmented Multiscale Deep InfoMax](https://arxiv.org/abs/1906.00910) and [Momentum Contrast](https://arxiv.org/abs/1911.05722). The implementation of the inference models comes from [this repo](git+https://github.com/cfld/rs_pretrained#egg=rsp).

2. **MlpClassifierPrimitive**: trains a two-layer neural network classifier on featurized remote sensing imagery. Produces heatmap visualizations for predictions using gradient-based [GradCam](https://arxiv.org/pdf/1610.02391v1.pdf) technique. 

