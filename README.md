# Repository of Yonder Open Source D3M Primitives

## Data Preprocessing

1. **DataCleaningPrimitive**: wrapper for the data cleaning primitive based on the Yonder [punk](https://github.com/NewKnowledge/punk) library.

2. **DukePrimitive**: wrapper of the [Duke library](https://github.com/NewKnowledge/duke) in the D3M infrastructure.

3. **ImageAugmentationPrimitive**: An image augmentation primitive based on the [Albumentations](https://github.com/albumentations-team/albumentations) library.

## Clustering

## Feature Selection

1. **PcaFeaturesPrimitive**: wrapper of the [Punk](https://github.com/NewKnowledge/punk) feature ranker into D3M infrastructure.

2. **RfFeaturesPrimitive** wrapper of the [Punk](https://github.com/NewKnowledge/punk) punk rrfeatures library into D3M infrastructure

## Dimensionality Reduction

## Natural Language Processing

## Image Classification

## Object Detection

1. **ObjectDetectionRNPrimitive**: wrapper for the Keras implementation of Retinanet from [this repo](https://github.com/fizyr/keras-retinanet). The original Retinanet paper can be found [here](https://arxiv.org/abs/1708.02002).

## Time Series Classification

1. **KaninePrimitive**: wrapper for tslearn's KNeighborsTimeSeriesClassifier algorithm

2. **LstmFcnPrimitive**: wrapper for LSTM Fully Convolutional Networks for Time Series Classification paper, original repo (https://github.com/titu1994/MLSTM-FCN), paper (https://arxiv.org/abs/1801.04503)

**layer_utils.py**: implementation of AttentionLSTM in tensorflow (compatible with 2), originally from https://github.com/houshd/LSTM-FCN

**lstm_model_utils.py**: functions to generate LSTM_FCN model architecture and data generators

**var_model_utils.py**: wrapper of the **auto_arima** method from **pmdarima.arima** with some specific parameters fixed

## Time Series Forecasting

1. **DeepArPrimitive**: wrapper for DeepAR recurrent, autoregressive Time Series Forecasting algorithm (https://arxiv.org/abs/1704.04110). Custom implementation repo (https://github.com/NewKnowledge/deepar)

2. **VarPrimitive**: wrapper for **statsmodels**' implementation of vector autoregression for multivariate time series

**var_model_utils.py**: wrapper of the **auto_arima** method from **pmdarima.arima** with some specific parameters fixed


