import io
from typing import Any

from keras_tuner import HyperModel

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GRU, Input, LSTM, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config.constants import (
    FORECASTER_LOSS, FORECASTER_OBJECTIVE,
    FORECAST_HORIZON, OBSERVATION_WINDOW
)


class TimeSeriesHyperModel(HyperModel):
    """A HyperModel for building and training time series forecasting models.

    This HyperModel constructs a Keras Sequential model with a configurable number of recurrent
    layers (LSTM or GRU) and a Dense output layer. The model architecture and training parameters
    are optimized using Keras Tuner.

    Attributes:
        n_variables (int): The number of variables in the time series data.
        model_type (str): The type of recurrent layer to use ('LSTM' or 'GRU'). Defaults to 'LSTM'.
    """

    def __init__(self, n_variables: int, model_type: str = 'LSTM'):
        """Initialize the TimeSeriesHyperModel.

        Args:
            n_variables (int): Number of variables in the time series data.
            model_type (str, optional): Type of recurrent layer to use ('LSTM' or 'GRU'). Defaults to 'LSTM'.
        """
        super().__init__()
        self.model_type = model_type
        self.n_variables = n_variables

    def build(self, hp: Any) -> Sequential:
        """Build and compile a Keras Sequential model based on provided hyperparameters.

        The model architecture is determined by the following hyperparameters:
          - 'num_layers': Number of recurrent layers (from 1 to 5).
          - 'units_<i>': Number of units in the i-th recurrent layer (from 32 to 128, step 32).
          - 'learning_rate': Learning rate for the Adam optimizer (choices: 1e-2, 1e-3, 1e-4).

        The type of recurrent layer (LSTM or GRU) is chosen based on the `model_type` attribute.
        The input shape is determined by `OBSERVATION_WINDOW` and `n_variables`, and the output
        is reshaped to match the forecast horizon.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Sequential: A compiled Keras Sequential model.
        """
        model = Sequential()
        model.add(Input(shape=(OBSERVATION_WINDOW, self.n_variables)))

        # Determine the number of layers only once to ensure consistency.
        num_layers = hp.Int('num_layers', 1, 5)
        for i in range(num_layers):
            units = hp.Int(f'units_{i}', 32, 128, step=32)
            return_seq = True if i < num_layers - 1 else False
            if self.model_type == 'LSTM':
                model.add(LSTM(units=units, return_sequences=return_seq))
            elif self.model_type == 'GRU':
                model.add(GRU(units=units, return_sequences=return_seq))
        model.add(Dense(self.n_variables * FORECAST_HORIZON))
        model.add(Reshape((FORECAST_HORIZON, self.n_variables)))
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice('learning_rate', [1e-2, 1e-3, 1e-4])
            ),
            loss=FORECASTER_LOSS
        )
        return model

    def fit(self, hp: Any, model: Any, X_train: np.array, y_train: np.array, **kwargs) -> dict:
        """Train the model on the provided training data with hyperparameter tuning.

        This method sets up early stopping based on the forecasting objective and splits the
        training data into training and validation sets based on a provided validation split.
        The data is then converted into TensorFlow datasets with batching and repeated for
        training.

        Args:
            hp (Any): Hyperparameters for tuning the model.
            model (Any): The Keras model to be trained.
            X_train (np.array): Training data features.
            y_train (np.array): Training data labels.
            **kwargs: Additional keyword arguments for model training. Must include a 'validation_split'
                      key that specifies the proportion of data to use for validation.

        Returns:
            dict: A dictionary containing the history of training metrics.

        Raises:
            ValueError: If 'validation_split' is not provided in **kwargs.
        """
        early_stopping = EarlyStopping(
            monitor=FORECASTER_OBJECTIVE,
            patience=5,
            restore_best_weights=True
        )
        kwargs['callbacks'] = kwargs.get('callbacks', []) + [early_stopping]

        batch_size = hp.Choice('batch_size', [16, 32, 64, 128])

        validation_split = kwargs.pop("validation_split", None)
        if validation_split is None:
            raise ValueError("validation_split must be provided.")

        num_total = len(X_train)
        num_train = int(num_total * (1 - validation_split))
        X_val = X_train[num_train:]
        y_val = y_train[num_train:]
        X_train = X_train[:num_train]
        y_train = y_train[:num_train]

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(batch_size, drop_remainder=True).repeat()
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True).repeat()

        steps_per_epoch = num_train // batch_size
        validation_steps = len(X_val) // batch_size

        history = model.fit(
            train_dataset,
            epochs=hp.Int('epochs', 25, 500),
            steps_per_epoch=steps_per_epoch,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            **kwargs,
        )

        return history.history


class InternalForecaster:
    """Encapsulate a forecasting model and provide utility methods for prediction and summary.

    This class wraps a Keras Sequential model to facilitate forecasting and obtaining a
    summary of the model's architecture.

    Attributes:
        model (Sequential): A Keras Sequential model used for forecasting.
        n_variables (int): The number of variables in the time series data.
    """

    def __init__(self, model: Sequential, n_variables: int):
        """Initialize the InternalForecaster.

        Args:
            model (Sequential): A trained Keras Sequential model.
            n_variables (int): The number of variables in the time series data.
        """
        self.model = model
        self.n_variables = n_variables

    def forecast(self, X: np.array) -> np.array:
        """Generate forecasts using the trained model.

        Args:
            X (np.array): Input data for which forecasts are to be generated.

        Returns:
            np.array: Forecasted values reshaped to (number of samples, FORECAST_HORIZON, n_variables).
        """
        return self.model.predict(X).reshape(-1, FORECAST_HORIZON, self.n_variables)

    def summary(self) -> str:
        """Generate a string summary of the model architecture.

        Uses an in-memory string buffer to capture the model summary output.

        Returns:
            str: A string containing the summary of the model.
        """
        string_io = io.StringIO()
        self.model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        return string_io.getvalue()
