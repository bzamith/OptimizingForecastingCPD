import time
from typing import Any, List, Tuple

import numpy as np

import pandas as pd

from tensorflow.keras import Sequential, callbacks, layers


OBSERVATION_WINDOW = 5
TRAIN_BATCH_SIZE = 16
PREDICT_BATCH_SIZE = 64


class CNNForecaster:
    """
    The CNNForecaster entity
    It extends the abstract Forecaster class
    """

    def __init__(self, variables: List[str]):
        """Initiate object"""
        self.variables = variables

    def split_sequence(self, sequence: np.array) -> Tuple[np.array, np.array]:
        """
        Extracts the input and target data for forecasting models
        :param sequence: The sequence that will be used by the forecaster
        :return: The input (x) and target (y) sequences for training
        """
        n_col = len(sequence[0])
        x, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + OBSERVATION_WINDOW
            if end_ix > len(sequence) - 1:
                break
            seq_x = sequence[i:end_ix, 0:n_col]
            x.append(seq_x)
            seq_y = sequence[end_ix, 0:n_col]
            y.append(seq_y)
        return np.array(x), np.array(y)

    def reshape_dataset(self, dataset: np.array, first_dim: int) -> Tuple[np.array, int]:
        """
        Reshapes the dataset accordingly, considering the forecasting window size
        :param dataset: The dataset that will be reshaped
        :param first_dim: The first dimension for reshaping
        :return: The reshaped dataset and the number of features value
        """
        try:
            n_features = dataset[0].shape[1]
        except IndexError:
            n_features = dataset.shape[1]
        reshaped_dataset = dataset.reshape((first_dim, OBSERVATION_WINDOW, n_features))
        return reshaped_dataset, n_features

    def get_assets(self, dataset: pd.DataFrame) -> Tuple[np.array, np.array, int]:
        """
        Extracts the temp_assets given the dataset, for the forecaster
        :param dataset: The dataset from which the temp_assets will be extracted
        :return: The input (x) and target (y), as well as the number of features
        """
        X, y = self.split_sequence(np.asarray(dataset))
        n_rows = X.shape[0]
        X, n_features = self.reshape_dataset(X, n_rows)
        return X, y, n_features

    def build_architecture(self, dataset: pd.DataFrame) -> Tuple[np.array, np.array, Any]:
        """
        Build the base architecture for the forecaster
        :param dataset: The dataset for training the forecaster
        :return: a list containing the temp_assets X and y from dataset, and the forecaster base architecture
        """
        architecture = Sequential(name="cnn-lstm")

        X, y, n_features = self.get_assets(dataset)

        architecture(layers.TimeDistributed(
            layers.Conv1D(filters=64, kernel_size=1, activation="sigmoid"),
            input_shape=(None, OBSERVATION_WINDOW, n_features)))
        architecture(layers.TimeDistributed(
            layers.MaxPooling1D(pool_size=2)))
        architecture(layers.TimeDistributed(layers.Flatten()))
        architecture.add(layers.LSTM(
            name="lstm_1",
            input_shape=(OBSERVATION_WINDOW, n_features), return_sequences=True,
            units=50))
        architecture.add(layers.LSTM(
            name="lstm_2",
            input_shape=(OBSERVATION_WINDOW, n_features),
            units=50))
        architecture.add(layers.Dense(
            name="dense",
            units=n_features,
            activation="sigmoid"))

        return X, y, architecture

    def fit(self, dataset: pd.DataFrame) -> Tuple[Any, float]:
        """
        Trains the forecasting model
        :param dataset: The dataset for training the forecaster
        :return: The history from forecaster
        """
        start_time = time.time()

        dataset_output = dataset.copy()
        dataset_shrink = dataset_output[self.variables]

        X, y, self.forecaster = self.build_architecture(dataset_shrink)
        self.forecaster.compile(loss="mean_absolute_percentage_error", optimizer="adam", metrics=["mse", "mae"])

        callback = callbacks.EarlyStopping(monitor="val_loss", patience=15)

        history = self.forecaster.fit(X, y,
                                      epochs=500,
                                      callbacks=[callback],
                                      verbose=False,
                                      validation_split=0.2,
                                      batch_size=TRAIN_BATCH_SIZE,
                                      shuffle=False)

        end_time = time.time()

        return history, (end_time - start_time)

    def evaluate(self, dataset: pd.DataFrame) -> List[float]:
        """
        Gets score for model in a test set
        :param dataset: The dataset for testing the model
        :return: The score
        """
        dataset_output = dataset.copy()
        dataset_shrink = dataset_output[self.variables]

        X, y, _ = self.get_assets(dataset_shrink)
        scores = self.forecaster.evaluate(X, y,
                                          batch_size=64,
                                          verbose=False)
        if not isinstance(scores, list):
            scores = [scores]
        return scores

    def predict(self, dataset: pd.DataFrame) -> Tuple[np.array, np.array]:
        """
        Extracts the forecasted data after training the model
        :param dataset: The dataset for which the value will be forecasted
        :return: The forecasted values
        """
        dataset_output = dataset.copy()
        dataset_shrink = dataset_output[self.variables]

        X, y, _ = self.get_assets(dataset_shrink)
        y_pred = self.forecaster.predict(X,
                                         batch_size=PREDICT_BATCH_SIZE,
                                         verbose=False)
        y_true = np.array(dataset[OBSERVATION_WINDOW:][self.variables])

        return y_pred, y_true
