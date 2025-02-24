import io
from typing import Any

from keras_tuner import HyperModel

import numpy as np

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import Dense, GRU, Input, LSTM
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from config.constants import FORECASTER_OBJECTIVE, OBSERVATION_WINDOW


class TimeSeriesHyperModel(HyperModel):
    """
    A HyperModel for building and training time series models using Keras Tuner.

    Attributes:
    ----------
        n_variables (int): The number of variables in the time series data.
        model_type (str): The type of model to build ('LSTM' or 'GRU'). Default is 'LSTM'.

    Methods
    -------
        build(hp: Any) -> Sequential:
            Builds and compiles a Keras Sequential model based on the hyperparameters.

        fit(hp: Any, model: Any, X_train: np.array, y_train: np.array, **kwargs) -> None:
            Trains the model on the provided training data with hyperparameter tuning.
    """
    def __init__(self, n_variables: int, model_type: str = 'LSTM'):
        super().__init__()
        self.model_type = model_type
        self.n_variables = n_variables

    def build(self, hp: Any) -> Sequential:
        """
        Build a Sequential model based on hyperparameters.

        Parameters:
        hp (Any): Hyperparameters for tuning the model.

        Returns:
        Sequential: A compiled Keras Sequential model.

        The model architecture is determined by the following hyperparameters:
        - 'num_layers': Number of LSTM/GRU layers (1 to 5).
        - 'units_<i>': Number of units in the i-th LSTM/GRU layer (32 to 128, step 32).
        - 'learning_rate': Learning rate for the Adam optimizer (1e-2, 1e-3, 1e-4).

        The model type (LSTM or GRU) is determined by the `self.model_type` attribute.
        The input shape is determined by `OBSERVATION_WINDOW` and `self.n_variables`.
        The output layer is a Dense layer with `self.n_variables` units.
        """
        model = Sequential()
        model.add(Input(shape=(OBSERVATION_WINDOW, self.n_variables)))

        for i in range(hp.Int('num_layers', 1, 5)):
            if self.model_type == 'LSTM':
                model.add(
                    LSTM(
                        units=hp.Int('units_' + str(i), 32, 128, step=32),
                        return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False,
                    )
                )
            elif self.model_type == 'GRU':
                model.add(
                    GRU(
                        units=hp.Int('units_' + str(i), 32, 128, step=32),
                        return_sequences=True if i < hp.Int('num_layers', 1, 3) - 1 else False,
                    )
                )
        model.add(Dense(self.n_variables))
        model.compile(
            optimizer=Adam(
                learning_rate=hp.Choice(
                    'learning_rate', [1e-2, 1e-3, 1e-4]
                )
            ),
            loss='mean_squared_error'
        )
        return model

    def fit(self, hp: Any, model: Any, X_train: np.array, y_train: np.array, **kwargs) -> None:
        """
        Fits the provided model to the training data.

        Parameters:
        hp (Any): Hyperparameters for the model.
        model (Any): The model to be trained.
        X_train (np.array): Training data features.
        y_train (np.array): Training data labels.
        **kwargs: Additional arguments to be passed to the model's fit method.

        Returns:
        dict: A dictionary containing the history of training metrics.
        """
        early_stopping = EarlyStopping(
            monitor=FORECASTER_OBJECTIVE,
            patience=5,
            restore_best_weights=True
        )
        kwargs['callbacks'] = kwargs['callbacks'] + [early_stopping]
        history = model.fit(
            X_train,
            y_train,
            epochs=hp.Int('epochs', 25, 500),
            batch_size=hp.Choice('batch_size', [16, 32, 64, 128]),
            **kwargs,
        )

        return history.history


class InternalForecaster:
    """
    A class used to encapsulate a forecasting model and provide utility methods for forecasting and summarizing the model.

    Attributes
    ----------
    model : Sequential
        A Keras Sequential model used for forecasting.

    Methods
    -------
    forecast(X: np.array) -> np.array
        Generates forecasts for the given input data.
    
    summary() -> str
        Returns a string summary of the model architecture.
    """
    def __init__(self, model: Sequential):
        self.model = model

    def forecast(self, X: np.array) -> np.array:
        """
        Generate forecasts using the trained model.

        Parameters:
        X (np.array): Input data for which forecasts are to be generated.

        Returns:
        np.array: Forecasted values.
        """
        return self.model.predict(X)

    def summary(self) -> str:
        """
        Generates a summary of the model.

        This method captures the summary of the model and returns it as a string.
        It uses an in-memory string buffer to capture the output of the model's
        summary method.

        Returns:
            str: A string containing the summary of the model.
        """
        string_io = io.StringIO()
        self.model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        return string_io.getvalue()
