"""Temporal Convolutional Network (TCN) forecaster implementation.

Based on "An Empirical Evaluation of Generic Convolutional and Recurrent
Networks for Sequence Modeling" by Bai et al. (2018).

Reference: https://arxiv.org/abs/1803.01271
"""

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from config.constants import (
    FORECAST_HORIZON,
    FORECASTER_LOSS,
    HP_DROPOUT_RATES,
    HP_LEARNING_RATES,
    HP_MODEL_DIMS,
    OBSERVATION_WINDOW,
)
from src.forecaster.base_forecaster import BaseForecasterHyperModel
from src.forecaster.model_architectures import ResidualBlock


def build_tcn_model(
    observation_window,
    n_variables,
    forecast_horizon,
    num_channels,
    kernel_size,
    dropout_rate,
    learning_rate,
):
    """Build a Temporal Convolutional Network (TCN) for forecasting.

    This implementation follows the architecture described in Bai et al. (2018):
    "An Empirical Evaluation of Generic Convolutional and Recurrent Networks
    for Sequence Modeling"

    Architecture:
    - Stack of residual blocks with exponentially increasing dilation rates
    - Each residual block contains two dilated causal convolutions
    - Residual connections allow training of very deep networks
    - Final global pooling and dense layers for prediction

    Key differences from original paper:
    - Original uses weight normalization, we use batch normalization
    - Original tested on various tasks, we apply to time series forecasting
    - We add final dense layers for multi-step forecasting

    Args:
        observation_window (int): Number of input time steps (e.g., 14).
        n_variables (int): Number of features per time step (e.g., 6).
        forecast_horizon (int): Number of time steps to forecast (e.g., 7).
        num_channels (list): Number of filters in each TCN block.
                            Length determines network depth.
                            Example: [64, 64, 64, 64] = 4 blocks with 64 filters each.
        kernel_size (int): Size of convolutional kernel (typically 3, 5, or 7).
        dropout_rate (float): Dropout probability for regularization.
        learning_rate (float): Learning rate for Adam optimizer.

    Returns:
        Model: Compiled Keras model ready for training.
    """
    inputs = layers.Input(shape=(observation_window, n_variables))

    x = layers.Conv1D(
        filters=num_channels[0],
        kernel_size=1,
        padding="same",
        kernel_initializer="he_normal",
    )(inputs)

    for i, num_filters in enumerate(num_channels):
        dilation_rate = 2**i
        x = ResidualBlock(
            filters=num_filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            dropout_rate=dropout_rate,
        )(x)

    x = layers.GlobalAveragePooling1D()(x)

    x = layers.Dense(num_channels[-1] // 2, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n_variables * forecast_horizon)(x)

    outputs = layers.Reshape((forecast_horizon, n_variables))(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=FORECASTER_LOSS,
    )

    return model


class TCNForecasterHyperModel(BaseForecasterHyperModel):
    """HyperModel for building and training TCN-based forecasting models.

    This HyperModel constructs a Temporal Convolutional Network following
    Bai et al. (2018). The architecture uses:
    - Causal dilated convolutions for large receptive fields
    - Residual connections for training deep networks
    - Exponentially increasing dilation rates

    The model performs Neural Architecture Search (NAS) and Hyperparameter
    Optimization (HPO) over:
    - Number of channels (filters) per layer
    - Kernel size
    - Number of stacked blocks (depth)
    - Dropout rate
    - Learning rate

    Attributes:
        n_variables (int): Number of variables in the time series data.
    """

    def build(self, hp: Any) -> Model:
        """Build and compile a TCN model based on provided hyperparameters.

        The hyperparameter space follows recommendations from the original
        TCN paper (Bai et al. 2018):
        - num_channels: Typically constant across layers (e.g., [64,64,64,64])
        - kernel_size: Usually 3, 5, or 7
        - num_blocks: 3-8 blocks (controls receptive field)
        - dropout_rate: 0.0-0.3 for regularization

        Receptive field calculation:
        RF = 1 + 2 * (kernel_size - 1) * sum(dilation_rates)
        RF = 1 + 2 * (k - 1) * (2^0 + 2^1 + ... + 2^(n-1))
        RF = 1 + 2 * (k - 1) * (2^n - 1)

        Example with kernel_size=3, num_blocks=4:
        RF = 1 + 2 * 2 * (2^4 - 1) = 1 + 4 * 15 = 61 timesteps

        Args:
            hp (Any): Keras Tuner hyperparameters object.

        Returns:
            Model: Compiled Keras model ready for training.
        """
        num_filters = hp.Choice("num_filters", HP_MODEL_DIMS)
        kernel_size = hp.Choice("kernel_size", [3, 5, 7])
        num_blocks = hp.Int("num_blocks", 3, 6)
        num_channels = [num_filters] * num_blocks
        dropout_rate = hp.Choice("dropout_rate", HP_DROPOUT_RATES)
        learning_rate = hp.Choice("learning_rate", HP_LEARNING_RATES)

        model = build_tcn_model(
            observation_window=OBSERVATION_WINDOW,
            n_variables=self.n_variables,
            forecast_horizon=FORECAST_HORIZON,
            num_channels=num_channels,
            kernel_size=kernel_size,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )

        return model
