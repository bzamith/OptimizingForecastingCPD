"""TS2VEC forecaster implementation.

TS2VEC (Time Series to Vector) is a self-supervised representation learning
framework for time series. It learns representations that can be used for
downstream forecasting tasks.

Reference: https://github.com/yuezhihan/ts2vec
"""

from typing import Any

import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, LayerNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel


class TS2VECForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel for TS2VEC-based time series forecasting.

    TS2VEC learns temporal representations through contrastive learning,
    then uses these representations for forecasting.

    This is a simplified implementation that uses a similar architecture
    to TS2VEC but adapted for direct forecasting.

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def build(self, hp: Any) -> Model:
        """Build TS2VEC-inspired model.

        The model parameters are:
          - 'repr_dims': Dimensionality of learned representations (32-256).
          - 'depth': Number of  encoding layers (1-4).
          - 'hidden_dims': Hidden layer size (64-256).
          - 'dropout_rate': Dropout rate (0.1-0.4).
          - 'learning_rate': Learning rate.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Model: Compiled Keras model.
        """
        from tensorflow.keras.layers import Bidirectional, Concatenate, GRU, Input

        repr_dims = hp.Choice("repr_dims", [32, 64, 128, 256])
        depth = hp.Int("depth", 1, 4)
        hidden_dims = hp.Choice("hidden_dims", [64, 128, 256])
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.4, step=0.1)
        learning_rate = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

        # Input
        inputs = Input(shape=(OBSERVATION_WINDOW, self.n_variables))

        # Temporal encoding with bidirectional GRU (similar to TS2VEC dilated CNN idea)
        x = inputs
        for i in range(depth):
            x = Bidirectional(
                GRU(
                    repr_dims // 2,
                    return_sequences=True if i < depth - 1 else False,
                    dropout=dropout_rate,
                )
            )(x)
            x = LayerNormalization()(x)

        # Representation layer
        x = Dense(repr_dims, activation="relu")(x)
        x = Dropout(dropout_rate)(x)

        # Forecasting head
        x = Dense(hidden_dims, activation="relu")(x)
        x = Dropout(dropout_rate)(x)
        x = Dense(hidden_dims // 2, activation="relu")(x)
        x = Dropout(dropout_rate)(x)

        # Output
        outputs = Dense(self.n_variables * FORECAST_HORIZON)(x)
        outputs = tf.keras.layers.Reshape((FORECAST_HORIZON, self.n_variables))(outputs)

        model = Model(inputs=inputs, outputs=outputs)

        model.compile(
            optimizer=Adam(learning_rate=learning_rate, clipnorm=1.0), loss="mean_squared_error"
        )

        return model


class TS2VECInternalForecaster:
    """Wrapper for TS2VEC-based forecaster.

    This class provides a unified interface for TS2VEC forecasting.

    Attributes:
        model: Trained Keras model.
        n_variables (int): Number of variables.
        batch_size (int): Batch size for training.
        epochs (int): Number of training epochs.
    """

    def __init__(self, model: Model, n_variables: int, batch_size: int, epochs: int):
        """Initialize the TS2VEC forecaster.

        Args:
            model (Model): Compiled Keras model.
            n_variables (int): Number of variables.
            batch_size (int): Batch size.
            epochs (int): Number of epochs.
        """
        self.model = model
        self.n_variables = n_variables
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X_train: np.array, y_train: np.array, **kwargs):
        """Fit the TS2VEC model.

        Args:
            X_train (np.array): Training input sequences.
            y_train (np.array): Training target sequences.
            **kwargs: Additional arguments for training.

        Returns:
            dict: Training history.
        """
        from src.forecaster.base_forecaster import get_early_stopping

        num_train = len(X_train)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(self.batch_size).repeat()

        steps_per_epoch = num_train // self.batch_size

        kwargs["callbacks"] = kwargs.get("callbacks", []) + [get_early_stopping(False)]

        history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

        return history.history

    def forecast(self, X: np.array) -> np.array:
        """Generate forecasts.

        Args:
            X (np.array): Input sequences.

        Returns:
            np.array: Forecasted values.
        """
        return self.model.predict(X).reshape(-1, FORECAST_HORIZON, self.n_variables)

    def summary(self) -> str:
        """Generate model summary.

        Returns:
            str: Model summary string.
        """
        import io

        string_io = io.StringIO()
        self.model.summary(print_fn=lambda x: string_io.write(x + "\n"))
        return string_io.getvalue()
