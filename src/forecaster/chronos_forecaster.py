"""Chronos forecaster implementation.

Chronos is a family of pretrained time series forecasting models based on
language model architectures. This implementation provides a wrapper around
the Chronos model for time series forecasting.

Reference: https://github.com/amazon-science/chronos-forecasting
"""

from typing import Any

import numpy as np

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel


class ChronosForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel for Chronos pretrained forecasting models.

    Chronos is a pretrained foundation model for time series forecasting.
    It uses transformer architecture trained on a large corpus of time series data.

    Note: This requires the chronos-forecasting package:
    pip install git+https://github.com/amazon-science/chronos-forecasting.git

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def build(self, hp: Any):
        """Build Chronos model configuration.

        The model parameters are:
          - 'model_size': Size of the pretrained model ('tiny', 'mini', 'small', 'base', 'large').
          - 'num_samples': Number of forecast samples to generate.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            dict: Configuration dictionary for Chronos model.
        """
        try:
            from chronos import ChronosPipeline
        except ImportError:
            raise ImportError(
                "Chronos not installed. Install with: "
                "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )

        model_size = hp.Choice("model_size", ["tiny", "mini", "small", "base"])
        num_samples = hp.Choice("num_samples", [10, 20, 50])

        # Load pretrained model
        model_name = f"amazon/chronos-t5-{model_size}"

        return {
            "model_name": model_name,
            "num_samples": num_samples,
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
        """Evaluate Chronos model on validation set.

        Note: Chronos is pretrained and does not require fitting.
        This method just evaluates the model on validation data.

        Args:
            hp (Any): Hyperparameters for tuning.
            model (dict): Chronos configuration dictionary.
            X_train (np.array): Training data (not used).
            y_train (np.array): Training labels (not used).
            validation_data (tuple): Validation features and labels.
            **kwargs: Additional arguments.

        Returns:
            dict: Dictionary containing validation loss.
        """
        try:
            from chronos import ChronosPipeline
            import torch
        except ImportError:
            return {"val_loss": 1e6}

        X_val, y_val = validation_data
        model_name = model["model_name"]
        num_samples = model["num_samples"]

        try:
            # Load pipeline
            pipeline = ChronosPipeline.from_pretrained(
                model_name,
                device_map="cpu",
                torch_dtype=torch.bfloat16,
            )

            # Evaluate on first validation sample
            val_losses = []

            for var_idx in range(self.n_variables):
                # Get context for this variable
                context = torch.tensor(X_val[0, :, var_idx])

                # Generate forecast
                forecast = pipeline.predict(
                    context,
                    prediction_length=FORECAST_HORIZON,
                    num_samples=num_samples,
                )

                # Take median of samples
                forecast_median = np.median(forecast.numpy(), axis=0)

                # Calculate loss
                actual = y_val[0, :, var_idx]
                mse = np.mean((actual - forecast_median) ** 2)
                val_losses.append(mse)

            avg_loss = np.mean(val_losses)
            return {"val_loss": avg_loss}

        except Exception as e:
            print(f"Chronos evaluation failed: {e}")
            return {"val_loss": 1e6}


class ChronosInternalForecaster:
    """Wrapper for Chronos pretrained model.

    This class provides a unified interface for using Chronos for forecasting.

    Attributes:
        pipeline: Chronos forecasting pipeline.
        model_name (str): Name of the pretrained model.
        num_samples (int): Number of forecast samples.
        n_variables (int): Number of variables.
        forecast_horizon (int): Number of steps to forecast.
    """

    def __init__(self, model_name: str, num_samples: int, n_variables: int, forecast_horizon: int):
        """Initialize the Chronos forecaster.

        Args:
            model_name (str): Pretrained model identifier.
            num_samples (int): Number of samples to generate.
            n_variables (int): Number of variables.
            forecast_horizon (int): Forecast horizon.
        """
        try:
            from chronos import ChronosPipeline
            import torch
        except ImportError:
            raise ImportError(
                "Chronos not installed. Install with: "
                "pip install git+https://github.com/amazon-science/chronos-forecasting.git"
            )

        self.model_name = model_name
        self.num_samples = num_samples
        self.n_variables = n_variables
        self.forecast_horizon = forecast_horizon

        # Load pipeline
        self.pipeline = ChronosPipeline.from_pretrained(
            model_name,
            device_map="cpu",
            torch_dtype=torch.bfloat16,
        )

    def fit(self, X_train: np.array, y_train: np.array, **kwargs):
        """Chronos is pretrained and doesn't require fitting.

        Args:
            X_train (np.array): Training data (ignored).
            y_train (np.array): Training labels (ignored).
            **kwargs: Additional arguments (ignored).

        Returns:
            dict: Empty dictionary.
        """
        # Chronos is pretrained, no fitting needed
        return {}

    def forecast(self, X: np.array) -> np.array:
        """Generate forecasts using Chronos.

        Args:
            X (np.array): Input sequences (shape: samples, window, variables).

        Returns:
            np.array: Forecasted values (shape: samples, horizon, variables).
        """
        import torch

        n_samples = len(X)
        predictions = np.zeros((n_samples, self.forecast_horizon, self.n_variables))

        for sample_idx in range(n_samples):
            for var_idx in range(self.n_variables):
                try:
                    # Get context
                    context = torch.tensor(X[sample_idx, :, var_idx])

                    # Generate forecast
                    forecast = self.pipeline.predict(
                        context,
                        prediction_length=self.forecast_horizon,
                        num_samples=self.num_samples,
                    )

                    # Take median of samples
                    predictions[sample_idx, :, var_idx] = np.median(forecast.numpy(), axis=0)

                except Exception as e:
                    print(f"Forecast failed for sample {sample_idx}, var {var_idx}: {e}")
                    # Use last known value as fallback
                    predictions[sample_idx, :, var_idx] = X[sample_idx, -1, var_idx]

        return predictions

    def summary(self) -> str:
        """Generate a summary of the Chronos model.

        Returns:
            str: Summary string.
        """
        summary = "Chronos Forecaster Summary\n"
        summary += f"Model: {self.model_name}\n"
        summary += f"Number of samples: {self.num_samples}\n"
        summary += f"Number of variables: {self.n_variables}\n"
        summary += f"Forecast horizon: {self.forecast_horizon}\n"
        summary += "Type: Pretrained foundation model (no training required)\n"
        return summary
