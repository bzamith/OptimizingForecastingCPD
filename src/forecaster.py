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
    def __init__(self, n_variables: int, model_type: str = 'LSTM'):
        super().__init__()
        self.model_type = model_type
        self.n_variables = n_variables

    def build(self, hp: Any) -> Sequential:
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
        early_stopping = EarlyStopping(
            monitor=FORECASTER_OBJECTIVE,
            patience=hp.Int('patience', 3, 15),
            restore_best_weights=True
        )
        kwargs['callbacks'] = kwargs['callbacks'] + [early_stopping]
        history = model.fit(
            X_train,
            y_train,
            epochs=hp.Int('epochs', 50, 500),
            batch_size=hp.Choice('batch_size', [16, 32, 64, 128]),
            **kwargs,
        )

        return history.history


class InternalForecaster:
    def __init__(self, model: Sequential):
        self.model = model

    def forecast(self, X: np.array) -> np.array:
        return self.model.predict(X)

    def summary(self) -> str:
        string_io = io.StringIO()
        self.model.summary(print_fn=lambda x: string_io.write(x + '\n'))
        return string_io.getvalue()
