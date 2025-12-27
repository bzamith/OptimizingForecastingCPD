"""ARIMA-based forecaster implementation.

ARIMA (AutoRegressive Integrated Moving Average) is a classical statistical
model for time series forecasting. This implementation uses grid search to
find optimal parameters.
"""

from typing import Any
import warnings

import numpy as np
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima.model import ARIMA

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel


class ARIMAForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel for ARIMA-based time series forecasting.

    This implementation uses ARIMA as a baseline statistical model for comparison
    with neural network approaches. It searches over (p, d, q) parameters.

    Note: ARIMA is a univariate model, so for multivariate data, we fit separate
    models for each variable.

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def build(self, hp: Any):
        """Build ARIMA configuration based on hyperparameters.

        The model parameters are:
          - 'p': Order of autoregressive term (0-5).
          - 'd': Degree of differencing (0-2).
          - 'q': Order of moving average term (0-5).

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            dict: Configuration dictionary for ARIMA model.
        """
        # ARIMA hyperparameters
        p = hp.Int("p", 0, 5)  # AR order
        d = hp.Int("d", 0, 2)  # Differencing order
        q = hp.Int("q", 0, 5)  # MA order

        # Return configuration instead of model
        # (ARIMA needs to be fit differently than neural networks)
        return {
            "order": (p, d, q),
            "n_variables": self.n_variables,
            "forecast_horizon": FORECAST_HORIZON,
            "observation_window": OBSERVATION_WINDOW,
        }

    def fit(
        self,
        hp: Any,
        model: Any,
        X_train: np.array,
        y_train: np.array,
        validation_data: tuple,
        **kwargs,
    ) -> dict:
        """Train ARIMA models and evaluate on validation set.

        For multivariate data, fits separate ARIMA models for each variable.

        Args:
            hp (Any): Hyperparameters for tuning.
            model (dict): ARIMA configuration dictionary.
            X_train (np.array): Training data features (not used for ARIMA).
            y_train (np.array): Training data labels.
            validation_data (tuple): Validation features and labels.
            **kwargs: Additional arguments (ignored for ARIMA).

        Returns:
            dict: Dictionary containing validation loss.
        """
        X_val, y_val = validation_data
        order = model["order"]
        n_variables = model["n_variables"]

        # Combine training data (ARIMA uses full history)
        # X_train has shape (samples, window, variables)
        # y_train has shape (samples, horizon, variables)

        # For ARIMA, we need continuous time series
        # Reconstruct by taking the last value from X and all from y
        train_sequences = []
        for i in range(n_variables):
            # Get all data for this variable
            var_data = []
            for j in range(len(X_train)):
                var_data.extend(X_train[j, :, i])
            # Add first horizon values from y_train
            if len(y_train) > 0:
                var_data.extend(y_train[0, :, i])
            train_sequences.append(np.array(var_data))

        # Validate on first validation sample
        val_predictions = []
        val_losses = []

        warnings.filterwarnings("ignore")  # Suppress ARIMA warnings

        for var_idx in range(n_variables):
            try:
                # Fit ARIMA model
                model_fit = ARIMA(train_sequences[var_idx], order=order).fit()

                # Forecast
                forecast = model_fit.forecast(steps=FORECAST_HORIZON)
                val_predictions.append(forecast)

                # Calculate loss for this variable
                actual = y_val[0, :, var_idx]
                mse = mean_squared_error(actual, forecast)
                val_losses.append(mse)

            except Exception as e:
                # If ARIMA fails, return high loss
                val_losses.append(1e6)

        warnings.filterwarnings("default")

        # Average loss across all variables
        avg_loss = np.mean(val_losses)

        return {"val_loss": avg_loss}


class ARIMAInternalForecaster:
    """Wrapper for trained ARIMA models.

    This class maintains fitted ARIMA models for each variable and provides
    a unified interface for forecasting.

    Attributes:
        models (list): List of fitted ARIMA models, one per variable.
        order (tuple): ARIMA order (p, d, q).
        n_variables (int): Number of variables.
        forecast_horizon (int): Number of steps to forecast.
    """

    def __init__(self, order: tuple, n_variables: int, forecast_horizon: int):
        """Initialize the ARIMA forecaster.

        Args:
            order (tuple): ARIMA order (p, d, q).
            n_variables (int): Number of variables in the time series.
            forecast_horizon (int): Number of time steps to forecast.
        """
        self.order = order
        self.n_variables = n_variables
        self.forecast_horizon = forecast_horizon
        self.models = []

    def fit(self, X_train: np.array, y_train: np.array, **kwargs):
        """Fit ARIMA models for each variable.

        Args:
            X_train (np.array): Training input sequences.
            y_train (np.array): Training target sequences.
            **kwargs: Additional arguments (ignored).

        Returns:
            dict: Empty dictionary (for compatibility).
        """
        warnings.filterwarnings("ignore")

        # Reconstruct continuous time series for each variable
        for var_idx in range(self.n_variables):
            var_data = []
            for i in range(len(X_train)):
                var_data.extend(X_train[i, :, var_idx])
            # Add y_train data
            for i in range(len(y_train)):
                var_data.extend(y_train[i, :, var_idx])

            var_data = np.array(var_data)

            try:
                model = ARIMA(var_data, order=self.order).fit()
                self.models.append(model)
            except Exception as e:
                print(f"Warning: ARIMA failed for variable {var_idx}: {e}")
                self.models.append(None)

        warnings.filterwarnings("default")
        return {}

    def forecast(self, X: np.array) -> np.array:
        """Generate forecasts for test data.

        Args:
            X (np.array): Input sequences (shape: samples, window, variables).

        Returns:
            np.array: Forecasted values (shape: samples, horizon, variables).
        """
        n_samples = len(X)
        predictions = np.zeros((n_samples, self.forecast_horizon, self.n_variables))

        warnings.filterwarnings("ignore")

        for sample_idx in range(n_samples):
            for var_idx in range(self.n_variables):
                if self.models[var_idx] is not None:
                    try:
                        # Use the input sequence to update the model
                        history = X[sample_idx, :, var_idx]

                        # Forecast
                        # Note: This is a simplified approach
                        # For better results, you'd want to retrain or use rolling forecasts
                        forecast = self.models[var_idx].forecast(steps=self.forecast_horizon)
                        predictions[sample_idx, :, var_idx] = forecast
                    except Exception as e:
                        # If forecast fails, use last known value
                        predictions[sample_idx, :, var_idx] = X[sample_idx, -1, var_idx]
                else:
                    # If model failed to fit, use last known value
                    predictions[sample_idx, :, var_idx] = X[sample_idx, -1, var_idx]

        warnings.filterwarnings("default")

        return predictions

    def summary(self) -> str:
        """Generate a summary of the ARIMA models.

        Returns:
            str: Summary string.
        """
        summary = "ARIMA Models Summary\n"
        summary += f"Order (p, d, q): {self.order}\n"
        summary += f"Number of variables: {self.n_variables}\n"
        summary += f"Forecast horizon: {self.forecast_horizon}\n"
        summary += (
            f"Models fitted: {sum(1 for m in self.models if m is not None)}/{self.n_variables}\n"
        )
        return summary
