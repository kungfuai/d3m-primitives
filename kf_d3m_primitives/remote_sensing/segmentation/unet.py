from typing import Optional, Union, List

from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead
from segmentation_models_pytorch.base.modules import Activation

import torch

from .moco_r50 import moco_r50


class Unet(SegmentationModel):
    def __init__(
        self,
        encoder_weights: str = None,
        encoder_freeze: bool = True,
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        classes: int = 1,
        activation: Optional[Union[str, callable]] = "sigmoid",
        device=torch.device("cpu"),
    ):
        super().__init__()

        self.encoder = moco_r50(
            encoder_weights, encoder_freeze=encoder_freeze, map_location=device
        )

        self.decoder = UnetDecoder(
            encoder_channels=(12, 64, 256, 512, 1024, 2048),
            decoder_channels=decoder_channels,
            n_blocks=5,
            use_batchnorm=decoder_use_batchnorm,
            center=False,
            attention_type=None,
        )

        self.segmentation_head = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            torch.nn.Flatten(),
            torch.nn.Linear(16, classes),
            Activation(activation),
        )

        self.classification_head = None
        self.initialize()


class SegmentationHeadImageLabelEval(torch.nn.Module):
    def __init__(self, segmentation_head_train):
        super().__init__()

        for layer in segmentation_head_train.modules():
            if isinstance(layer, torch.nn.Linear):
                self.linear_layer = layer

        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, features):

        features = features.permute(0, 2, 3, 1)
        logits = self.linear_layer(features)
        activations = self.sigmoid(logits)
        return activations
