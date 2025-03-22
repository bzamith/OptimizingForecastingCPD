import io
from typing import Any

from keras_tuner import HyperModel

import numpy as np

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import BatchNormalization, Dense, Input, LSTM, Reshape
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from config.constants import FORECASTER_LOSS, FORECAST_HORIZON, OBSERVATION_WINDOW


def get_early_stopping(is_validation: bool = True) -> EarlyStopping:
    """
    Creates and returns an EarlyStopping callback for training models.

    Args:
        is_validation (bool): Whether it should consider validation loss or not.
            Default is True.

    Returns:
        EarlyStopping: An instance of the EarlyStopping callback configured with
        the specified parameters.
    """
    return EarlyStopping(
        monitor='val_loss' if is_validation else 'loss',
        patience=10,
        min_delta=1e-2,
        restore_best_weights=True
    )


def get_reduce_lr(is_validation: bool = True) -> EarlyStopping:
    return ReduceLROnPlateau(
        monitor='val_loss' if is_validation else 'loss',
        factor=0.5,
        patience=5,
        restore_best_weights=True,
        min_lr=1e-5
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

    def __init__(self, n_variables: int):
        """Initialize the TimeSeriesHyperModel.

        Args:
            n_variables (int): Number of variables in the time series data.
        """
        super().__init__()
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

        num_layers = hp.Int('num_layers', 1, 5)
        for i in range(num_layers):
            units = hp.Int(f'units_{i}', 32, 128, step=32)
            return_seq = True if i < num_layers - 1 else False
            model.add(LSTM(units=units, return_sequences=return_seq, dropout=0.2, recurrent_dropout=0.2, kernel_regularizer=l2(1e-4)))
            model.add(BatchNormalization())
        model.add(Dense(self.n_variables * FORECAST_HORIZON))
        model.add(Reshape((FORECAST_HORIZON, self.n_variables)))
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice('learning_rate', [1e-2, 5e-2, 1e-3, 5e-3, 1e-4, 5e-4]),
                clipnorm=1.0
            ),
            loss=FORECASTER_LOSS
        )
        return model

    def fit(self, hp: Any, model: Any, X_train: np.array, y_train: np.array, validation_data: tuple, **kwargs) -> dict:
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
            validation_data (tuple): A tuple containing validation features and labels.
            **kwargs: Additional keyword arguments for model training. Must include a 'validation_split'
                      key that specifies the proportion of data to use for validation.

        Returns:
            dict: A dictionary containing the history of training metrics.

        Raises:
            ValueError: If 'validation_split' is not provided in **kwargs.
        """
        X_val, y_val = validation_data

        len_X_train = len(X_train)
        len_X_val = len(X_val)
        val_min_batch = len_X_val - (len_X_val % 4)
        if val_min_batch <= 0:
            raise Exception("Validation batch size must be greater than 0.")

        batch_size = hp.Int('batch_size', min_value=min(32, val_min_batch), max_value=max(32, len_X_val // 4), step=16)

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(batch_size).repeat()
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = val_dataset.batch(batch_size, drop_remainder=True).repeat()

        steps_per_epoch = len_X_train // batch_size
        validation_steps = len_X_val // batch_size

        if validation_steps <= 0:
            raise Exception("Validation steps must be greater than 0.")

        kwargs['callbacks'] = kwargs.get('callbacks', []) + [get_early_stopping(), get_reduce_lr()]

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=hp.Int('epochs', min_value=25, max_value=150, step=25),
            steps_per_epoch=steps_per_epoch,
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

    def __init__(self, model: Sequential, n_variables: int, batch_size: int, epochs: int):
        """Initialize the InternalForecaster.

        Args:
            model (Sequential): A trained Keras Sequential model.
            n_variables (int): The number of variables in the time series data.
        """
        self.model = model
        self.n_variables = n_variables
        self.batch_size = batch_size
        self.epochs = epochs

    def fit(self, X_train: np.array, y_train: np.array, **kwargs) -> dict:
        """Fits the model to the training data.

        Args:
            X_train (np.array): Training input data.
            y_train (np.array): Training target data.
            **kwargs: Additional arguments to pass to the model's fit method.

        Returns:
            dict: A dictionary containing the history of training metrics.
        """
        num_train = len(X_train)
        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(self.batch_size).repeat()

        steps_per_epoch = num_train // self.batch_size

        kwargs['callbacks'] = kwargs.get('callbacks', []) + [get_early_stopping(False), get_reduce_lr(False)]

        history = self.model.fit(
            train_dataset,
            epochs=self.epochs,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

        return history.history

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
