"""Base forecaster classes with common functionality.

This module contains the base classes that all forecaster implementations inherit from.
"""

from abc import ABC, abstractmethod
import io
from typing import Any
import warnings

from keras_tuner import HyperModel
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model

from config.constants import (
    EARLY_STOPPING_PATIENCE,
    FORECAST_HORIZON,
    NB_EPOCHS,
)


def get_adaptive_batch_size(
    n_samples: int,
    min_batch: int = 32,
    max_batch: int = 128,
) -> int:
    """Compute an adaptive batch size based on dataset size.

    Strategy:
    - Use sqrt scaling: batch ~ sqrt(n_samples)
    - Clamp between min_batch and max_batch
    - Round to nearest power of 2 for better performance

    Args:
        n_samples (int): Number of training samples.
        min_batch (int): Minimum batch size.
        max_batch (int): Maximum batch size.

    Returns:
        int: Adaptive batch size.
    """
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    # Base heuristic
    batch = int(np.sqrt(n_samples))

    # Clamp
    batch = max(min_batch, min(batch, max_batch))

    # Round to nearest power of 2
    batch = 2 ** int(np.round(np.log2(batch)))

    return int(batch)


def get_early_stopping(is_validation: bool = True) -> EarlyStopping:
    """Creates and returns an EarlyStopping callback for training models.

    Args:
        is_validation (bool): Whether it should consider validation loss or not.
            Default is True.

    Returns:
        EarlyStopping: An instance of the EarlyStopping callback configured with
        the specified parameters.
    """
    return EarlyStopping(
        monitor="val_loss" if is_validation else "loss",
        patience=EARLY_STOPPING_PATIENCE,
        min_delta=1e-2,
        restore_best_weights=True,
    )


class BaseForecasterHyperModel(HyperModel, ABC):
    """Abstract base class for all forecaster hypermodels.

    This class provides common functionality for all forecaster implementations
    including the fit method with proper data handling.

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def __init__(self, n_variables: int):
        """Initialize the BaseForecasterHyperModel.

        Args:
            n_variables (int): Number of variables in the time series data.
        """
        super().__init__()
        self.n_variables = n_variables

    @abstractmethod
    def build(self, hp: Any) -> Model:
        """Build and compile a model based on provided hyperparameters.

        This method must be implemented by subclasses.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Model: A compiled Keras model.
        """
        pass

    def fit(
        self,
        hp: Any,
        model: Any,
        X_train: np.array,
        y_train: np.array,
        validation_data: tuple,
        **kwargs,
    ) -> dict:
        """Train the model on the provided training data with hyperparameter tuning.

        This method implements common training logic for all forecaster types.

        Args:
            hp (Any): Hyperparameters for tuning the model.
            model (Any): The Keras model to be trained.
            X_train (np.array): Training data features.
            y_train (np.array): Training data labels.
            validation_data (tuple): A tuple containing validation features and labels.
            **kwargs: Additional keyword arguments for model training.

        Returns:
            dict: A dictionary containing the history of training metrics.

        Raises:
            Exception: If validation batch size or steps are invalid.
        """
        X_val, y_val = validation_data

        len_X_train = len(X_train)
        len_X_val = len(X_val)

        batch_size = get_adaptive_batch_size(len_X_train)

        X_train = tf.convert_to_tensor(X_train, dtype=tf.float32)
        y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
        X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
        y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
        val_dataset = (
            val_dataset.batch(batch_size, drop_remainder=True).repeat().prefetch(tf.data.AUTOTUNE)
        )

        steps_per_epoch = len_X_train // batch_size
        validation_steps = len_X_val // batch_size

        if validation_steps <= 0:
            warnings.warn("Validation steps must be greater than 0.", UserWarning)
            validation_steps = 1

        kwargs["callbacks"] = kwargs.get("callbacks", []) + [get_early_stopping()]

        # For ARIMA models, also pass raw validation data to avoid issues with empty batches
        # when validation set is smaller than batch size
        # Only pass this if the model's fit method accepts it (check the signature)
        import inspect

        fit_signature = inspect.signature(model.fit)
        if "raw_validation_data" in fit_signature.parameters:
            kwargs["raw_validation_data"] = (X_val, y_val)

        history = model.fit(
            train_dataset,
            validation_data=val_dataset,
            validation_steps=validation_steps,
            epochs=NB_EPOCHS,
            steps_per_epoch=steps_per_epoch,
            **kwargs,
        )

        return history.history


class InternalForecaster:
    """Encapsulate a forecasting model and provide utility methods for prediction and summary.

    This class wraps a Keras model to facilitate forecasting and obtaining a
    summary of the model's architecture.

    Attributes:
        model (Model): A Keras model used for forecasting.
        n_variables (int): The number of variables in the time series data.
    """

    def __init__(self, model: Model, n_variables: int):
        """Initialize the InternalForecaster.

        Args:
            model (Model): A trained Keras model.
            n_variables (int): The number of variables in the time series data.
        """
        self.model = model
        self.n_variables = n_variables

    def fit(self, X_train: np.array, y_train: np.array, **kwargs) -> dict:
        """Fits the model to the training data.

        Args:
            X_train (np.array): Training input data.
            y_train (np.array): Training target data.
            **kwargs: Additional arguments to pass to the model's fit method.

        Returns:
            dict: A dictionary containing the history of training metrics.
        """
        len_X_train = len(X_train)

        batch_size = get_adaptive_batch_size(len_X_train)

        train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
        train_dataset = train_dataset.batch(batch_size).repeat().prefetch(tf.data.AUTOTUNE)

        steps_per_epoch = len_X_train // batch_size

        kwargs["callbacks"] = kwargs.get("callbacks", []) + [get_early_stopping(False)]

        history = self.model.fit(
            train_dataset,
            epochs=NB_EPOCHS,
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
        self.model.summary(print_fn=lambda x: string_io.write(x + "\n"))
        return string_io.getvalue()
