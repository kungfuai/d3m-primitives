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

1. **DataCleaningPrimitive**: This primitive standardizes columns that represent dates or numbers, including missing values. It is based on the [punk](https://github.com/NewKnowledge/punk) library.

2. **DukePrimitive**: This primitive produces abstractive summarization tags based on a word2vec model trained on Wikipedia data and a corresponding Wikipedia ontology. It is based on the [Duke](https://github.com/uncharted-distil/duke) library.

3. **SimonPrimitive**: This primitive infers the semantic type of each column using a pre-trained LSTM-CNN model. The model was trained on simulated data of different semantic types using the python Faker library. Base library [here](https://github.com/uncharted-distil/simon).

4. **GoatForwardPrimitive**: This primitive geocodes location names in specified columns into longitude/latitude coordinates. It uses a large downloaded search index: the [photon](https://github.com/komoot/photon) geocoding server (based on OpenStreetMap).

5. **GoatReversePrimitive**: This primitive converts longitude/latitude coordinates into geographic location names. It uses a large downloaded search index: the [photon](https://github.com/komoot/photon) geocoding server (based on OpenStreetMap).

### Clustering

1. **HdbscanPrimitive**: This primitive applies hierarchical density-based ([HDBSCAN](https://hdbscan.readthedocs.io/en/latest/index.html)) and density-based ([DBSCAN](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html)) clustering algorithms.

2. **StorcPrimitive**: This primitive applies [tslearn](https://tslearn.readthedocs.io/en/stable/index.html)'s kmeans clustering implementations to time series data.

3. **SpectralClustering**: This primitive applies scikit-learn's [Spectral Clustering](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html) algorithm to unsupervised, supervised or semi-supervised datasets. 

### Feature Selection

1. **PcaFeaturesPrimitive**: This primitive implements a two-step feature selection process. First, it performs prinicipal component analysis on all numeric data in the dataset. Second, it uses each original feature's contribution to the first principal component as a proxy for the 'score' of that feature. Base library [here](https://github.com/NewKnowledge/punk).

2. **RfFeaturesPrimitive** This primitive performs supervised recursive feature elimination using random forests to generate an ordered list of features. Base library [here](https://github.com/NewKnowledge/punk).

### Dimensionality Reduction

1. **TsnePrimitive**: This primitive applies scikit-learn's T-distributed stochastic neighbour embedding ([TSNE](https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html)) algorithm.

### Natural Language Processing

1. **Sent2VecPrimitive**: This primitive produces numerical representations of text data using a model that was pre-trained on English Twitter bi-grams. Base library [here](https://github.com/uncharted-distil/nk-sent2vec).

### Object Detection

1. **ObjectDetectionRNPrimitive**: This primitive utilizes RetinaNet, a convolutional neural network (CNN), for object detection. It wraps the Keras implementation of Retinanet from [this repo](https://github.com/fizyr/keras-retinanet). The original Retinanet paper can be found [here](https://arxiv.org/abs/1708.02002).  

### Time Series Classification

1. **KaninePrimitive**: This primitive applies the k-nearest neighbors ([KNeighborsTimeSeriesClassifier](https://tslearn.readthedocs.io/en/latest/gen_modules/neighbors/tslearn.neighbors.KNeighborsTimeSeriesClassifier.html#tslearn.neighbors.KNeighborsTimeSeriesClassifier)) classification algorithm to time series data.

2. **LstmFcnPrimitive**: This primitive applies a LSTM fully convolutional neural network for time series classification. It wraps the implementation here [LSTM Fully Convolutional Networks for Time Series Classification](https://github.com/houshd/LSTM-FCN).

### Time Series Forecasting

1. **DeepArPrimitive**: This primitive applies the [DeepAR](https://arxiv.org/abs/1704.04110) (deep, autoregressive) forecasting methodology for time series prediction. It trains a global model on related time series to produce probabilistic forecasts. It wraps the implementation from [GluonTS](https://github.com/awslabs/gluon-ts).

2. **NBEATSPrimitive**: This primitive applies the Neural basis expansion analysis for interpretable time
series forecasting ([N-BEATS](https://arxiv.org/abs/1905.10437)) method for time series forecasting. It wraps the implementation from [GluonTS](https://github.com/awslabs/gluon-ts).

3. **VarPrimitive**: This primitive applies a vector autoregression ([VAR](https://www.statsmodels.org/dev/vector_ar.html)) multivariate forecasting model to time series data. It defaults to an [ARIMA](http://alkaline-ml.com/pmdarima/0.9.0/modules/generated/pyramid.arima.auto_arima.html) model if the time series is univariate. 

### Interpretability

**shap_explainers**: wrapper of Lundberg's shapley values implementation for tree models. Currently integrated into *d3m.primitives.learner.random_forest.DistilEnsembleForest* as *produce_shap_values()*.

### Remote Sensing

1. **RemoteSensingPretrainedPrimitive**: This primitive featurizes remote sensing imagery using a pre-trained model. The pre-trained model was learned on a sample of Sentinel-2 imagery and optimized using a self-supervised objective. There are two inference models that correspond to two pretext tasks: [Augmented Multiscale Deep InfoMax](https://arxiv.org/abs/1906.00910) and [Momentum Contrast](https://arxiv.org/abs/1911.05722). The implementation of the inference models comes from [this repo](git+https://github.com/cfld/rs_pretrained#egg=rsp).

2. **MlpClassifierPrimitive**: This primitive trains a two-layer neural network classifier on featurized remote sensing imagery. It also produces heatmap visualizations for predictions using the gradient-based [GradCam](https://arxiv.org/pdf/1610.02391v1.pdf) technique. 

3. **ImageRetrievalPrimitive**: This primitive retrieves semantically similar images from an index of un-annotated images using heuristics. It is useful for supporting an iterative, human-in-the-loop, retrieval pipeline

4. **ImageSegmentationPrimitive**: This primitive trains a binary image segmentation model using image-level weak supervision (see [1](https://www.mdpi.com/2072-4292/12/2/207/htm)). Furthermore, the pre-trained featurizer is used the initialize the parameters of segmentation encoder. Thus, the training process is also an instance of transfer learning (see [2](https://arxiv.org/pdf/2003.02899.pdf)). 

### Semi-Supervised

1. **CorrectAndSmoothPrimitive**: This primitive applies the [Correct and Smooth](https://arxiv.org/pdf/2010.13993.pdf) procedure for semi-supervised learning. It combines a simple classification model with two label propagation post-processing steps - one that spreads residual errors and one that smooths predictions. 

2. **TabularSemiSupervisedPrimitive**: This primitive applies one of three methods (PseudoLabel, VAT, and ICT) for semi-supervised learning on tabular data. 



