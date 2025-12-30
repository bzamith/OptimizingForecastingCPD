"""GRU-based forecaster implementation."""

from typing import Any

from tensorflow.keras.layers import BatchNormalization, Dense, GRU, Input, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config.constants import (
    FORECAST_HORIZON,
    FORECASTER_LOSS,
    HP_DROPOUT_RATES,
    HP_L2_REG,
    HP_LEARNING_RATES,
    HP_MODEL_DIMS,
    OBSERVATION_WINDOW,
)
from src.forecaster.base_forecaster import BaseForecasterHyperModel


def build_gru_model(
    observation_window,
    n_variables,
    forecast_horizon,
    num_layers,
    units,
    dropout_rate,
    recurrent_dropout_rate,
    l2_reg,
    learning_rate,
):
    """Build a GRU-based forecasting model.

    GRU (Gated Recurrent Unit) is a simplified variant of LSTM with fewer parameters
    and faster training, often achieving comparable performance. This implementation
    uses stacked GRU layers with dropout, recurrent dropout, L2 regularization,
    and batch normalization for improved training stability.

    Args:
        observation_window (int): Number of time steps in input sequence.
        n_variables (int): Number of variables/features per time step.
        forecast_horizon (int): Number of time steps to forecast.
        num_layers (int): Number of GRU layers (1-3).
        units (int): Number of units in each GRU layer.
        dropout_rate (float): Dropout rate for GRU layers.
        recurrent_dropout_rate (float): Recurrent dropout rate for GRU layers.
        l2_reg (float): L2 regularization strength.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        Sequential: Compiled Keras Sequential model.
    """
    model = Sequential()
    model.add(Input(shape=(observation_window, n_variables)))

    for i in range(num_layers):
        return_seq = True if i < num_layers - 1 else False

        model.add(
            GRU(
                units=units,
                return_sequences=return_seq,
                dropout=dropout_rate,
                recurrent_dropout=recurrent_dropout_rate,
                kernel_regularizer=l2(l2_reg),
            )
        )

    # Always use BatchNormalization for better training stability
    model.add(BatchNormalization())

    model.add(Dense(n_variables * forecast_horizon))
    model.add(Reshape((forecast_horizon, n_variables)))

    model.compile(optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss=FORECASTER_LOSS)

    return model


class GRUForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel for building and training GRU-based time series forecasting models.

    This HyperModel constructs a Keras Sequential model with a configurable number of
    GRU layers and a Dense output layer. The model architecture and training parameters
    are optimized using Keras Tuner.

    GRU offers similar capabilities to LSTM but with:
    - Fewer parameters (faster training and less memory)
    - Simpler architecture (no separate cell state)
    - Often comparable performance on many tasks

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def build(self, hp: Any) -> Sequential:
        """Build and compile a Keras Sequential model based on provided hyperparameters.

        The model architecture is determined by the following hyperparameters:
          - 'num_layers': Number of GRU layers (1-3).
          - 'units': Number of units in each GRU layer (64, 128, 256).
          - 'dropout_rate': Dropout rate for GRU layers (0.1, 0.2).
          - 'recurrent_dropout_rate': Recurrent dropout rate (0.1, 0.2).
          - 'l2_reg': L2 regularization strength (1e-4, 1e-3).
          - 'learning_rate': Learning rate for the Adam optimizer (1e-2, 1e-3, 1e-4).

        Note: BatchNormalization is always enabled for training stability.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Sequential: A compiled Keras Sequential model.
        """
        num_layers = hp.Int("num_layers", 1, 3)
        units = hp.Choice("units", HP_MODEL_DIMS)
        dropout_rate = hp.Choice("dropout_rate", HP_DROPOUT_RATES)
        recurrent_dropout_rate = hp.Choice("recurrent_dropout_rate", HP_DROPOUT_RATES)
        l2_reg = hp.Choice("l2_reg", HP_L2_REG)
        learning_rate = hp.Choice("learning_rate", HP_LEARNING_RATES)

        model = build_gru_model(
            observation_window=OBSERVATION_WINDOW,
            n_variables=self.n_variables,
            forecast_horizon=FORECAST_HORIZON,
            num_layers=num_layers,
            units=units,
            dropout_rate=dropout_rate,
            recurrent_dropout_rate=recurrent_dropout_rate,
            l2_reg=l2_reg,
            learning_rate=learning_rate,
        )

        return model
