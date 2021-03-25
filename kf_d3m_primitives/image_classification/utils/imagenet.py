""" 
    Bootstrapped from https://github.com/NewKnowledge/imagenet and refined for D3M purposes
    Original implementation from Craig Corcoran
"""

import os
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import inception_v3, mobilenet_v2, xception
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, GlobalMaxPooling2D
from tensorflow.keras.utils import to_categorical, Sequence

import logging

logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


class ImagenetModel:

    """A class for featurizing images using pre-trained neural nets on ImageNet
    and finetuning those nets for downstream classification
    """

    def __init__(
        self,
        model="inception_v3",
        weights="imagenet",
        include_top=False,
        pooling=None,
        n_channels=None,
        clf_head_dense_dim=1024,
    ):
        """Creates ImageNet base model for featurization or classification and corresponding image
        preprocessing function
            :param model: options are xception, inception_v3, and mobilenet_v2
            :param weights: 'imagenet' or filepath
            :param include_top: whether to include original ImageNet classification head with 1000 classes
            :param pooling: 'avg', 'max', or None
            :param n_channels: number of channels to keep if performing featurization
            :param clf_head_dense_dim: dimension of dense layer before softmax classification (only applies
                if `include_top` is false)
        """

        self.include_top = (
            include_top  # determines if used for classification or featurization
        )
        self.n_channels = n_channels
        self.pooling = pooling
        self.clf_head_dense_dim = clf_head_dense_dim

        if model == "xception":
            self.model = xception.Xception(
                weights=weights, include_top=include_top, pooling=pooling
            )
            self.preprocess = xception.preprocess_input
            self.target_size = (299, 299)
            if include_top:
                self.decode = xception.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 2048) * (
                    1 if pooling else 10 ** 2
                )
        elif model == "inception_v3":
            self.model = inception_v3.InceptionV3(
                weights=weights, include_top=include_top, pooling=pooling
            )
            self.preprocess = inception_v3.preprocess_input
            self.target_size = (299, 299)
            if include_top:
                self.decode = inception_v3.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 2048) * (
                    1 if pooling else 8 ** 2
                )
        elif model == "mobilenet_v2":
            self.model = mobilenetv2.MobileNetV2(
                weights=weights, include_top=include_top, pooling=pooling
            )
            self.preprocess = mobilenetv2.preprocess_input
            self.target_size = (244, 244)
            if include_top:
                self.decode = mobilenetv2.decode_predictions
            else:
                self.output_dim = (n_channels if n_channels else 1280) * (
                    1 if pooling else 7 ** 2
                )
        else:
            raise Exception("model option not implemented")

    def _load_finetune_model(
        self,
        nclasses=2,
        weights_path=None,
    ):
        """Constructs finetuning model architecture and optionally loads weights
        :param nclasses: number of classes on which to softmax over
        :param weights_path: optional filepath from which to try to load weights
        """

        out = self.model.output
        if self.pooling is None:
            out = GlobalAveragePooling2D()(
                out
            )  # if self.pooling == 'avg' else GlobalMaxPooling2D()(out)
        dense = Dense(self.clf_head_dense_dim, activation="relu")(out)
        preds = Dense(nclasses, activation="softmax")(dense)
        finetune_model = Model(inputs=self.model.input, outputs=preds)

        # try to load weights
        if weights_path is not None:
            if os.path.isfile(weights_path):
                finetune_model.load_weights(weights_path)

        return finetune_model

    def get_features(self, images_array):
        """ takes a batch of images as a 4-d array and returns the (flattened) imagenet features for those images as a 2-d array """
        if self.include_top:
            raise Exception(
                "getting features from a classification model with include_top=True is currently not supported"
            )

        if images_array.ndim != 4:
            raise Exception("invalid input shape for images_array, expects a 4d array")

        # preprocess and compute image features
        logger.debug(f"preprocessing {images_array.shape[0]} images")
        images_array = self.preprocess(images_array)
        logger.debug(f"computing image features")
        image_features = self.model.predict(images_array)

        # if n_channels is specified, only keep that number of channels
        if self.n_channels:
            logger.debug(f"truncating to first {self.n_channels} channels")
            image_features = image_features.T[: self.n_channels].T

        # reshape output array by flattening each image into a vector of features
        shape = image_features.shape
        return image_features.reshape(shape[0], np.prod(shape[1:]))

    def predict(self, images_array):
        """ alias for get_features to more closely match scikit-learn interface """
        return self.get_features(images_array)

    def finetune(
        self,
        train_dataset,
        val_dataset=None,
        nclasses=2,
        top_layer_epochs=1,
        unfreeze_proportions=[0.5],
        all_layer_epochs=5,
        class_weight=None,
        optimizer_top="rmsprop",
        optimizer_full="sgd",
        callbacks=None,
        load_weights_path=None,
        save_weights_path=None,
    ):
        """Finetunes the Imagenet model iteratively on a smaller set of images with (potentially) a smaller set of classes.
        First finetunes last layer then freezes bottom N layers and retrains the rest
            :param train_dataset: (X, y) pair of tf.constant tensors for training
            :param val_dataset: (X, y) pair of tf.constant tensors for validation, optional
            :param nclasses: number of classes
            :param top_layer_epochs: how many epochs for which to finetune classification head (happens first)
            :param unfreeze_proportions: list of proportions representing how much of the base ImageNet model one wants to
                unfreeze (later layers unfrozen) for another round of finetuning
            :param all_layer_epochs: how many epochs for which to finetune entire model (happens second)
            :param class_weight: class weights (used for both training steps)
            :param optimizer_top: optimizer to use for training of classification head
            :param optimizer_full: optimizer to use for training full classification model
                * suggest to use lower learning rate / more conservative optimizer for this step to
                  prevent catastrophic forgetting
            :param callbacks: optional list of callbacks to use for each round of finetuning
            :param load_weights_path: optional filepath from which to try to load weights
            :param save_weights_path: optional filepath to which to store weights
        """

        finetune_model = self._load_finetune_model(
            nclasses=nclasses, weights_path=load_weights_path
        )

        fitting_histories = []

        # freeze all convolutional InceptionV3 layers, retrain top layer
        for layer in self.model.layers:
            layer.trainable = False
        finetune_model.compile(optimizer=optimizer_top, loss="categorical_crossentropy")

        fitting_histories.append(
            finetune_model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=top_layer_epochs,
                class_weight=class_weight,
                shuffle=False,
                callbacks=callbacks,
            )
        )

        # iteratively unfreeze specified proportion of later ImageNet base layers and finetune
        finetune_model.compile(
            optimizer=optimizer_full, loss="categorical_crossentropy"
        )

        for p in unfreeze_proportions:
            freeze_count = int(len(self.model.layers) * p)
            for layer in finetune_model.layers[:freeze_count]:
                layer.trainable = False
            for layer in finetune_model.layers[freeze_count:]:
                layer.trainable = True

            fitting_histories.append(
                finetune_model.fit(
                    train_dataset,
                    validation_data=val_dataset,
                    epochs=all_layer_epochs,
                    class_weight=class_weight,
                    shuffle=False,
                    callbacks=callbacks,
                )
            )

        # save weights
        if save_weights_path is not None:
            finetune_model.save_weights(save_weights_path)

        return fitting_histories

    def finetune_classify(
        self,
        test_dataset,
        nclasses=2,
        load_weights_path=None,
    ):

        """Uses the finetuned model to predict on a test dataset.
        :param test_dataset: X, tf.constant tensor for inference
        :param nclasses: number of classes
        :return: array of softmaxed prediction probabilities
        :param load_weights_path: optional filepath from which to try to load weights
        """

        finetune_model = self._load_finetune_model(
            nclasses=nclasses, weights_path=load_weights_path
        )

        return finetune_model.predict(test_dataset)


class ImageNetGen(Sequence):
    """ Tf.Keras Sequence for ImageNet input data """

    def __init__(self, X, y=None, batch_size=32):
        self.X = X
        self.y = y
        self.batch_size = batch_size

    def __len__(self):
        return math.ceil(self.X.shape[0] / self.batch_size)

    def __getitem__(self, idx):
        batch_x = self.X[idx * self.batch_size : (idx + 1) * self.batch_size]
        if self.y is None:
            return tf.constant(batch_x)
        else:
            batch_y = self.y[idx * self.batch_size : (idx + 1) * self.batch_size]
            return tf.constant(batch_x), tf.constant(batch_y)
