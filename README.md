# Repository of Yonder Open Source D3M Primitives

## Data Preprocessing

## Clustering

## Feature Selection

## Dimensionality Reduction

## Natural Language Processing

## Image Classification

## Object Detection

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


