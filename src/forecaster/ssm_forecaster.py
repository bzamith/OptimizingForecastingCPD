"""State Space Model (SSM) forecaster implementation."""

from typing import Any

from tensorflow.keras.models import Model

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel
from src.forecaster.model_architectures import build_ssm_model


class SSMForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel for building and training State Space Model (SSM) based forecasting models.

    This HyperModel constructs an SSM model inspired by S4/Mamba architectures for
    efficient long-range sequence modeling.

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def build(self, hp: Any) -> Model:
        """Build and compile an SSM model based on provided hyperparameters.

        The model architecture is determined by the following hyperparameters:
          - 'd_model': Model dimension (32, 64, 128).
          - 'd_state': State space dimension (32, 64, 128).
          - 'num_ssm_blocks': Number of S4 layers (1-3).
          - 'dropout_rate': Dropout rate for regularization (0.1-0.3).
          - 'learning_rate': Learning rate for the Adam optimizer.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Model: A compiled Keras Model.
        """
        d_model = hp.Choice("d_model", [32, 64, 128])
        d_state = hp.Choice("d_state", [32, 64, 128])
        num_blocks = hp.Int("num_ssm_blocks", 1, 3)
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.3, step=0.1)
        learning_rate = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

        model = build_ssm_model(
            observation_window=OBSERVATION_WINDOW,
            n_variables=self.n_variables,
            forecast_horizon=FORECAST_HORIZON,
            d_model=d_model,
            d_state=d_state,
            num_ssm_blocks=num_blocks,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )

        return model
