# D3M wrapper for Keras implemenetation of Retinanet.

This is D3M wrapper of the Keras Retinanet implementation for D3M. It installs most of the files from our [object_detection_retinanet](https://github.com/NewKnowledge/object-detection-retinanet/) repo.

It is based on the Keras RetinaNet base library found [here](https://github.com/fizyr/keras-retinanet). See section below for major changes and modifications.

## Pipeline input

File containing the D3M file index, the image name, and a string of bounding coordinates in the following format: `x_min, y_min, x_min, y_max, x_max, y_max, x_max, y_min`.

## Pipeline output

D3M data frame contained the D3M file index, string of bounding coordinates in the 8-coordinate format (see pipeline input above), and a confidence score for the prediction. There may be more than one prediction per image or dummy predictions where images did not have any predictions.

## Pipeline

The `object_detection_pipeline.py` file is to be run in d3m runtime with a weights file. During testing, average precision (AP) is used to score the prediction results.

## Changes from Fizyr Keras RetinaNet implemetation

The D3M primitive is essentially a wrapper on the entire Keras-RetinaNet codebase to fit into D3M specifications. Since the Keras-RetinaNet codebase is a command-line tool, these details had to be stripped out and the arguments exposed as primitive hyperparameters. Most of the `train.py` script was inserted into the `fit()` and the other methods it calls were inserted into the primitive class. The only major modifications were to the `Generator` class which has to be modified slightly to parse the datasets as they are input in D3M format.

`convert_model.py` and `evaluate.py` were inserted into the `produce()` method. `evaluate.py` has a `--convert-model` CLI argument which is mostly the same as the content in `convert_model.py`. Therefore, `convert_model.py` was removed and the contents of `evaluate.py` that convert the model were retained to be inserted into `produce()`. The modifications to `evaluate.py()` were to output a data frame that contains the list of bounding boxes in the expected format for the `metric.py` D3M evaluation using average precision.