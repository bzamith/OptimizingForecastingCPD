import time

import numpy as np
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense


class Forecaster:
    def __init__(self, observation_window, nb_units, train_batch_size):
        self.observation_window = observation_window
        self.nb_units = nb_units
        self.train_batch_size = train_batch_size
        self.model = None

    def _reshape_input(self, X):
        num_samples = len(X) - self.observation_window
        num_features = X.shape[1]  # Get the number of features
        X_reshaped = np.zeros((num_samples, self.observation_window, num_features))
        y_reshaped = np.zeros((num_samples, 1, num_features))
        for i in range(num_samples):
            X_reshaped[i] = X[i:i+self.observation_window]
            y_reshaped[i] = X.iloc[i+self.observation_window]
        X_reshaped = np.array(X_reshaped)
        y_reshaped = np.array(y_reshaped)
        return X_reshaped, y_reshaped

    def fit(self, X_train):
        start_time = time.time()

        X_train_reshaped, y_train_reshaped = self._reshape_input(X_train)
        n_features = X_train_reshaped.shape[2]
        self.model = Sequential()
        self.model.add(LSTM(self.nb_units, input_shape=(self.observation_window, n_features)))
        self.model.add(Dense(n_features))
        self.model.compile(loss='mse', optimizer='adam')
        early_stopping = EarlyStopping(patience=5, monitor='val_loss', restore_best_weights=True)
        self.model.fit(X_train_reshaped, y_train_reshaped,
                       epochs=250,
                       batch_size=self.train_batch_size,
                       callbacks=[early_stopping],
                       validation_split=0.2,
                       verbose=0)

        end_time = time.time()

        return end_time - start_time

    def predict(self, X_pred):
        X_pred_reshaped, y_true = self._reshape_input(X_pred)
        y_pred = self.model.predict(X_pred_reshaped, verbose=0)
        y_true = y_true.tolist()
        y_pred = y_pred.tolist()
        y_true = [sublist[0] for sublist in y_true]
        return y_true, y_pred

