"""ARIMA-based forecaster implementation using traditional rolling forecast approach.

This implementation fits one separate ARIMA (AutoRegressive Integrated Moving Average) model
per variable. Unlike VARIMA, this approach does not capture cross-variable dependencies,
but treats each variable independently.
"""

from typing import Any, List
import warnings

import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from tensorflow.data import Dataset
from tensorflow.keras.models import Model

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel, InternalForecaster


class ARIMAForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel for building and training ARIMA-based time series forecasting models.

    ARIMA (AutoRegressive Integrated Moving Average) is a univariate time series model.
    This implementation fits one separate ARIMA model per variable, treating each variable
    independently. Unlike neural network-based approaches, ARIMA uses traditional rolling
    forecasts where models are fit once on training data and then make predictions.

    The ARIMA model is defined by three parameters:
    - p (order of autoregression): Number of past time steps to use
    - d (degree of differencing): Number of times to difference the series
    - q (order of moving average): Number of past forecast errors to use

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def __init__(self, n_variables: int):
        """Initialize the ARIMAForecasterHyperModel.

        Args:
            n_variables (int): Number of variables in the time series data.
        """
        super().__init__(n_variables)

    def build(self, hp: Any) -> "ARIMAModelWrapper":
        """Build an ARIMA model configuration based on provided hyperparameters.

        Args:
            hp (Any): Hyperparameters used for model tuning. Expects:
                - 'p': AutoRegressive order (0-10)
                - 'd': Differencing order (0-2)
                - 'q': Moving Average order (0-5)

        Returns:
            ARIMAModelWrapper: A wrapper object that manages ARIMA model training and forecasting.
        """
        p = hp.Int("p", 0, 10)
        d = hp.Int("d", 0, 2)
        q = hp.Int("q", 0, 5)

        return ARIMAModelWrapper(
            p=p,
            d=d,
            q=q,
            n_variables=self.n_variables,
            forecast_horizon=FORECAST_HORIZON,
        )


class ARIMAModelWrapper(Model):
    """Wrapper class to make ARIMA models compatible with Keras training interface.

    This implementation fits one separate ARIMA model per variable:
    - Training: Reconstructs the original time series from sliding windows and fits
      one ARIMA model for each variable independently
    - Prediction: Uses the fitted models to make multi-step forecasts for each variable

    Attributes:
        p (int): AutoRegressive order.
        d (int): Differencing order.
        q (int): Moving Average order.
        n_variables (int): Number of variables in the time series.
        forecast_horizon (int): Number of steps ahead to forecast.
        models (List): List of fitted ARIMA models (one per variable).
        training_series (np.ndarray): Stored training time series.
    """

    def __init__(self, p: int, d: int, q: int, n_variables: int, forecast_horizon: int):
        """Initialize the ARIMA model wrapper.

        Args:
            p (int): AutoRegressive order.
            d (int): Differencing order.
            q (int): Moving Average order.
            n_variables (int): Number of variables in the time series.
            forecast_horizon (int): Number of steps ahead to forecast.
        """
        super().__init__()
        self.p = p
        self.d = d
        self.q = q
        self.n_variables = n_variables
        self.forecast_horizon = forecast_horizon
        self.models: List = [None] * n_variables
        self.training_series = None
        self._is_fitted = False

    def build(self, input_shape):
        """Build method required by Keras Model.

        Args:
            input_shape: Shape of the input data.
        """
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass through the model.

        For each sample in the batch, this uses the fitted ARIMA models to make
        multi-step forecasts. Each variable is forecasted independently using its own model.

        Args:
            inputs: Input tensor of shape (batch_size, observation_window, n_variables).
            training: Whether the model is in training mode.

        Returns:
            Output tensor of shape (batch_size, forecast_horizon, n_variables).
        """
        from tensorflow import Tensor

        if not self._is_fitted or any(model is None for model in self.models):
            raise RuntimeError("Models must be fitted before making predictions.")

        inputs_np = inputs.numpy() if isinstance(inputs, Tensor) else inputs
        batch_size = inputs_np.shape[0]
        predictions = np.zeros((batch_size, self.forecast_horizon, self.n_variables))

        for sample_idx in range(batch_size):
            # Get the observation window for all variables
            # Shape: (observation_window, n_variables)
            history = inputs_np[sample_idx, :, :]

            # Forecast each variable independently using pre-fitted models
            for var_idx in range(self.n_variables):
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings("ignore")

                        # Use the pre-fitted ARIMA model for this variable
                        # Do NOT refit - models were already fitted in the fit() method
                        if self.models[var_idx] is not None:
                            fitted = self.models[var_idx]
                            # Forecast for this variable
                            forecast = fitted.forecast(steps=self.forecast_horizon)
                            predictions[sample_idx, :, var_idx] = forecast
                        else:
                            # If no model available, use last observation as fallback
                            predictions[sample_idx, :, var_idx] = history[-1, var_idx]

                except Exception:
                    # If forecasting fails, use last observation as fallback
                    predictions[sample_idx, :, var_idx] = history[-1, var_idx]

        from tensorflow import convert_to_tensor

        return convert_to_tensor(predictions, dtype="float32")

    def fit(
        self,
        x=None,
        y=None,
        batch_size=None,
        epochs=1,
        verbose="auto",
        callbacks=None,
        validation_split=0.0,
        validation_data=None,
        shuffle=True,
        class_weight=None,
        sample_weight=None,
        initial_epoch=0,
        steps_per_epoch=None,
        validation_steps=None,
        validation_batch_size=None,
        validation_freq=1,
        max_queue_size=10,
        workers=1,
        use_multiprocessing=False,
        raw_validation_data=None,
    ):
        """Train separate ARIMA models for each variable independently.

        This reconstructs the original time series from the sliding windows and fits
        one ARIMA model per variable, treating each variable independently.

        Args:
            x: Input training data - can be a TensorFlow Dataset, Tensor, or numpy array.
                Shape: (n_samples, OBSERVATION_WINDOW, n_variables)
            y: Target training data (used to reconstruct full series).
                Shape: (n_samples, FORECAST_HORIZON, n_variables)
            raw_validation_data: Tuple of (X_val, y_val) tensors for validation.
            **kwargs: Additional keyword arguments (mostly ignored for ARIMA).

        Returns:
            MockHistory: Object with history attribute containing training history.
        """
        from tensorflow import Tensor

        # Convert to numpy if needed - handle TensorFlow Dataset
        if isinstance(x, Dataset):
            # Extract all data from the dataset
            all_x_batches = []
            all_y_batches = []
            for batch_x, batch_y in x.take(steps_per_epoch if steps_per_epoch else -1):
                batch_x_np = batch_x.numpy() if isinstance(batch_x, Tensor) else batch_x
                batch_y_np = batch_y.numpy() if isinstance(batch_y, Tensor) else batch_y
                all_x_batches.append(batch_x_np)
                all_y_batches.append(batch_y_np)
            x_np = np.concatenate(all_x_batches, axis=0) if all_x_batches else np.array([])
            y_np = np.concatenate(all_y_batches, axis=0) if all_y_batches else np.array([])
        elif isinstance(x, Tensor):
            x_np = x.numpy()
            y_np = y.numpy() if y is not None else None
        else:
            x_np = np.array(x)
            y_np = np.array(y) if y is not None else None

        # Validate shape
        if len(x_np.shape) != 3:
            raise ValueError(
                f"Expected input shape (batch_size, observation_window, n_variables), "
                f"but got shape {x_np.shape}"
            )

        # Reconstruct the original time series from sliding windows
        # The first window gives us the first OBSERVATION_WINDOW points
        # Each subsequent window shifts by FORECAST_HORIZON
        n_samples = x_np.shape[0]

        # Initialize with first observation window
        reconstructed_length = OBSERVATION_WINDOW + n_samples * FORECAST_HORIZON
        reconstructed_series = np.zeros((reconstructed_length, self.n_variables))

        # Start with the first observation window
        reconstructed_series[:OBSERVATION_WINDOW] = x_np[0]

        # Add subsequent forecast horizons (which become the next observations)
        if y_np is not None and len(y_np) > 0:
            for i in range(n_samples):
                start_idx = OBSERVATION_WINDOW + i * FORECAST_HORIZON
                end_idx = start_idx + FORECAST_HORIZON
                reconstructed_series[start_idx:end_idx] = y_np[i]

        # Store the reconstructed training series
        self.training_series = reconstructed_series

        # Fit one ARIMA model per variable
        print(
            f"Fitting ARIMA({self.p},{self.d},{self.q}) on training series of length {len(reconstructed_series)} with {self.n_variables} variables"
        )

        for var_idx in range(self.n_variables):
            try:
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")

                    # Get the time series for this variable
                    var_series = reconstructed_series[:, var_idx]

                    # Fit ARIMA model for this variable
                    arima_model = ARIMA(var_series, order=(self.p, self.d, self.q))
                    self.models[var_idx] = arima_model.fit()
                    print(
                        f"  Successfully fitted ARIMA({self.p},{self.d},{self.q}) for variable {var_idx}"
                    )

            except Exception as e:
                print(f"  Failed to fit ARIMA model for variable {var_idx}: {e}")
                import traceback

                traceback.print_exc()
                # Set model to None - will use fallback during prediction
                self.models[var_idx] = None

        self._is_fitted = True

        # Extract validation data if provided
        val_x_np = None
        val_y_np = None

        if (
            raw_validation_data is not None
            and isinstance(raw_validation_data, tuple)
            and len(raw_validation_data) >= 2
        ):
            # Use raw validation data directly (before it was wrapped in dataset)
            val_x, val_y = raw_validation_data[0], raw_validation_data[1]
            val_x_np = val_x.numpy() if isinstance(val_x, Tensor) else np.array(val_x)
            val_y_np = val_y.numpy() if isinstance(val_y, Tensor) else np.array(val_y)
        elif validation_data is not None:
            if isinstance(validation_data, Dataset):
                # Extract validation data from dataset
                val_batches_x = []
                val_batches_y = []
                # Use a reasonable number of steps to ensure we get data
                num_steps = max(validation_steps if validation_steps else 1, 1)
                for batch_x, batch_y in validation_data.take(num_steps):
                    batch_x_np = batch_x.numpy() if isinstance(batch_x, Tensor) else batch_x
                    batch_y_np = batch_y.numpy() if isinstance(batch_y, Tensor) else batch_y
                    val_batches_x.append(batch_x_np)
                    val_batches_y.append(batch_y_np)
                if val_batches_x:
                    val_x_np = np.concatenate(val_batches_x, axis=0)
                    val_y_np = np.concatenate(val_batches_y, axis=0)
            elif isinstance(validation_data, tuple) and len(validation_data) >= 2:
                val_x, val_y = validation_data[0], validation_data[1]
                val_x_np = val_x.numpy() if isinstance(val_x, Tensor) else np.array(val_x)
                val_y_np = val_y.numpy() if isinstance(val_y, Tensor) else np.array(val_y)

        # Compute validation loss if validation data is provided
        val_loss = None
        if (
            val_x_np is not None
            and val_y_np is not None
            and len(val_x_np) > 0
            and all(model is not None for model in self.models)
        ):
            try:
                # Use a subset of validation data to speed up evaluation
                num_val_samples = min(10, len(val_x_np))
                val_x_subset = val_x_np[:num_val_samples]
                val_y_subset = val_y_np[:num_val_samples]

                predictions = self.call(val_x_subset)
                # Compute MSE loss
                pred_np = predictions.numpy() if isinstance(predictions, Tensor) else predictions
                val_loss = float(np.mean((pred_np - val_y_subset) ** 2))

                # Ensure loss is valid (not nan or inf from bad predictions)
                if not np.isfinite(val_loss):
                    val_loss = float("inf")

                print(f"Validation MSE: {val_loss:.6f}")
            except Exception as e:
                print(f"Warning: Failed to compute validation loss: {e}")
                val_loss = float("inf")
        else:
            # If no validation data or any model failed to fit, use a large but finite loss
            val_loss = float("inf")

        # Return a mock history object for compatibility with Keras training interface
        return MockHistory(val_loss=val_loss)

    def predict(self, x, batch_size=None, verbose="auto", steps=None, callbacks=None, **kwargs):
        """Generate predictions using trained VARMAX model.

        Args:
            x: Input data of shape (batch_size, observation_window, n_variables).
            **kwargs: Additional keyword arguments (mostly ignored).

        Returns:
            Predictions of shape (batch_size, forecast_horizon, n_variables).
        """
        return self.call(x).numpy()


class MockHistory:
    """Mock history object for compatibility with Keras training interface."""

    def __init__(self, val_loss=None):
        """Initialize with validation loss if provided.

        Args:
            val_loss (float, optional): Validation loss value.
        """
        self.history = {
            "loss": [0.0],  # VARIMA doesn't track training loss
            "val_loss": [val_loss] if val_loss is not None else [],
        }
