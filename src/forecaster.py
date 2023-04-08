import math
import time
from typing import Any, List, Tuple

import numpy as np

import pandas as pd

from tensorflow.keras import Sequential, callbacks, layers

from src.dataset import TRAIN_PERC

OBSERVATION_WINDOW = 5
TRAIN_BATCH_SIZE = 16
NB_UNITS = 50
PREDICT_BATCH_SIZE = 64

OBSERVATION_WINDOW_GRID = [5, 10, 15]
TRAIN_BATCH_SIZE_GRID = [4, 8, 16, 32]
NB_UNITS_GRID = [25, 50, 100, 200]


class CNNForecaster:
    """
    The CNNForecaster entity
    It extends the abstract Forecaster class
    """

    def __init__(self, variables: List[str], observation_window: int = OBSERVATION_WINDOW,
                 train_batch_size: int = TRAIN_BATCH_SIZE, nb_units: int = NB_UNITS):
        """Initiate object"""
        self.variables = variables
        self.observation_window = observation_window
        self.train_batch_size = train_batch_size
        self.nb_units = nb_units

    def split_sequence(self, sequence: np.array) -> Tuple[np.array, np.array]:
        """
        Extracts the input and target data for forecasting models
        :param sequence: The sequence that will be used by the forecaster
        :return: The input (x) and target (y) sequences for training
        """
        n_col = len(sequence[0])
        x, y = list(), list()
        for i in range(len(sequence)):
            end_ix = i + self.observation_window
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
        reshaped_dataset = dataset.reshape((first_dim, self.observation_window, n_features))
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
            input_shape=(None, self.observation_window, n_features)))
        architecture(layers.TimeDistributed(
            layers.MaxPooling1D(pool_size=2)))
        architecture(layers.TimeDistributed(layers.Flatten()))
        architecture.add(layers.LSTM(
            name="lstm_1",
            input_shape=(self.observation_window, n_features), return_sequences=True,
            units=self.nb_units))
        architecture.add(layers.LSTM(
            name="lstm_2",
            input_shape=(self.observation_window, n_features),
            units=self.nb_units))
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
                                      batch_size=self.train_batch_size,
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
        y_true = np.array(dataset[self.observation_window:][self.variables])

        return y_pred, y_true


def hpo_fit(X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[np.array, np.array, List[float], float]:
    index_split = math.floor(X_train.shape[0] * TRAIN_PERC)
    hpo_X_train = X_train[:index_split].reset_index(drop=True)
    hpo_X_val = X_train[index_split:].reset_index(drop=True)

    best_dict = {}
    best_mape = 1000000000000000

    for observation_window in OBSERVATION_WINDOW_GRID:
        for train_batch_size in TRAIN_BATCH_SIZE_GRID:
            for nb_units in NB_UNITS_GRID:
                cnn = CNNForecaster(variables, observation_window, train_batch_size, nb_units)
                _, _ = cnn.fit(hpo_X_train)
                curr_mape = cnn.evaluate(hpo_X_val)[0]
                if curr_mape < best_mape:
                    best_mape = curr_mape
                    best_dict = {
                        "observation_window": observation_window,
                        "train_batch_size": train_batch_size,
                        "nb_units": nb_units,
                    }

    cnn = CNNForecaster(variables, best_dict["observation_window"], best_dict["train_batch_size"], best_dict["nb_units"])
    _, seconds = cnn.fit(X_train)
    y_pred, y_true = cnn.predict(X_test)
    metrics = cnn.evaluate(X_test)
    return y_true, y_pred, metrics, seconds
