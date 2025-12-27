"""LSTM-based forecaster implementation."""

from typing import Any

from tensorflow.keras.layers import BatchNormalization, Dense, Input, LSTM, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config.constants import FORECAST_HORIZON, FORECASTER_LOSS, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel


class LSTMForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel for building and training LSTM-based time series forecasting models.

    This HyperModel constructs a Keras Sequential model with a configurable number of
    LSTM layers and a Dense output layer. The model architecture and training parameters
    are optimized using Keras Tuner.

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def build(self, hp: Any) -> Sequential:
        """Build and compile a Keras Sequential model based on provided hyperparameters.

        The model architecture is determined by the following hyperparameters:
          - 'num_layers': Number of LSTM layers (1-5).
          - 'units_<i>': Number of units in the i-th LSTM layer (32, 64, 96, 128).
          - 'dropout_rate': Dropout rate for LSTM layers (0.1-0.3).
          - 'recurrent_dropout_rate': Recurrent dropout rate (0.1-0.3).
          - 'l2_reg': L2 regularization strength (1e-5, 1e-4, 1e-3).
          - 'use_batch_norm': Whether to use batch normalization.
          - 'learning_rate': Learning rate for the Adam optimizer.

        The input shape is determined by OBSERVATION_WINDOW and n_variables, and the output
        is reshaped to match the forecast horizon.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Sequential: A compiled Keras Sequential model.
        """
        model = Sequential()
        model.add(Input(shape=(OBSERVATION_WINDOW, self.n_variables)))

        num_layers = hp.Int("num_layers", 1, 5)
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.3, step=0.1)
        recurrent_dropout_rate = hp.Float("recurrent_dropout_rate", 0.1, 0.3, step=0.1)
        l2_reg = hp.Choice("l2_reg", [1e-5, 1e-4, 1e-3])
        use_batch_norm = hp.Boolean("use_batch_norm")
        learning_rate = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

        for i in range(num_layers):
            units = hp.Choice(f"units_{i}", [32, 64, 96, 128])
            return_seq = True if i < num_layers - 1 else False

            model.add(
                LSTM(
                    units=units,
                    return_sequences=return_seq,
                    dropout=dropout_rate,
                    recurrent_dropout=recurrent_dropout_rate,
                    kernel_regularizer=l2(l2_reg),
                )
            )

            if use_batch_norm:
                model.add(BatchNormalization())

        model.add(Dense(self.n_variables * FORECAST_HORIZON))
        model.add(Reshape((FORECAST_HORIZON, self.n_variables)))

        model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss=FORECASTER_LOSS
        )

        return model
