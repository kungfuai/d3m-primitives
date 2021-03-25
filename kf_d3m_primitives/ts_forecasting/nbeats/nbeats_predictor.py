from typing import List
from itertools import product
import copy

import numpy as np
from mxnet.gluon import HybridBlock
import mxnet as mx
from gluonts.model.n_beats._network import NBEATSPredictionNetwork
from gluonts.model.n_beats._estimator import NBEATSEstimator
from gluonts.model.n_beats._ensemble import NBEATSEnsembleEstimator
from gluonts.core.component import validated
from gluonts.model.common import Tensor
from gluonts.transform import Transformation
from gluonts.model.predictor import Predictor, RepresentableBlockPredictor

""" This module overwrites the NBEATSPredictionNetwork, NBEATSEstimator, and
    NBEATSEnsembleEstimator classes to make it possible to 
    extract the additive decomposition of the seasonal and trend forecast
    from an interpretable NBEATS architecture. 
"""


class NBEATSPredictionNetworkHook(NBEATSPredictionNetwork):
    @validated()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.all_trend_forecasts = []

    # noinspection PyMethodOverriding,PyPep8Naming
    def hybrid_forward(
        self, F, past_target: Tensor, future_target: Tensor = None
    ) -> Tensor:
        """
        Parameters
        ----------
        F
        past_target
            Tensor with past observations.
            Shape: (batch_size, context_length, target_dim).
        future_target
            Not used.
        Returns
        -------
        Tensor
            Prediction sample. Shape: (batch_size, 1, prediction_length).
        """

        backcast, forecast = self.net_blocks[0](past_target)
        self.all_trend_forecasts.append(forecast.asnumpy())
        backcast = past_target - backcast
        forecast = forecast + self.net_blocks[1](backcast)
        forecast = F.expand_dims(forecast, axis=1)
        return forecast

    def get_trend_forecast(self):
        trend_forecasts = np.concatenate(self.all_trend_forecasts, axis=0)
        return trend_forecasts

    def clear_trend_forecast(self):
        self.all_trend_forecasts = []


class NBEATSEstimatorHook(NBEATSEstimator):
    @validated()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    # def _validate_nbeats_argument(self, *args):
    #     return super()._validate_nbeats_argument(*args)

    # def create_transformation(self) -> Transformation:
    #     return super().create_transformation()

    # def create_training_network(self) -> HybridBlock:
    #     return super().create_training_network()

    def create_predictor(
        self, transformation: Transformation, trained_network: HybridBlock
    ) -> Predictor:
        prediction_network = NBEATSPredictionNetworkHook(
            prediction_length=self.prediction_length,
            context_length=self.context_length,
            num_stacks=self.num_stacks,
            widths=self.widths,
            num_blocks=self.num_blocks,
            num_block_layers=self.num_block_layers,
            expansion_coefficient_lengths=self.expansion_coefficient_lengths,
            sharing=self.sharing,
            stack_types=self.stack_types,
            params=trained_network.collect_params(),
        )

        return RepresentableBlockPredictor(
            input_transform=transformation,
            prediction_net=prediction_network,
            batch_size=self.trainer.batch_size,
            freq=self.freq,
            prediction_length=self.prediction_length,
            ctx=self.trainer.ctx,
        )


class NBEATSEnsembleEstimatorHook(NBEATSEnsembleEstimator):
    @validated()
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def _estimator_factory(self, **kwargs):
        estimators = []
        for context_length, loss_function, init_id in product(
            self.meta_context_length,
            self.meta_loss_function,
            list(range(self.meta_bagging_size)),
        ):
            # So far no use for the init_id, models are by default always randomly initialized
            estimators.append(
                NBEATSEstimatorHook(
                    freq=self.freq,
                    prediction_length=self.prediction_length,
                    context_length=context_length,
                    trainer=copy.deepcopy(self.trainer),
                    num_stacks=self.num_stacks,
                    widths=self.widths,
                    num_blocks=self.num_blocks,
                    num_block_layers=self.num_block_layers,
                    expansion_coefficient_lengths=self.expansion_coefficient_lengths,
                    sharing=self.sharing,
                    stack_types=self.stack_types,
                    loss_function=loss_function,
                    **kwargs,
                )
            )
        return estimators
