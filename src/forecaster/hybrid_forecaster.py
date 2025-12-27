"""Hybrid Transformer-SSM forecaster implementation."""

from typing import Any

from tensorflow.keras.models import Model

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel
from src.forecaster.model_architectures import build_hybrid_transformer_ssm_model


class HybridForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel combining Transformer and SSM architectures.

    This HyperModel constructs a hybrid model that leverages both Transformer attention
    mechanisms and State Space Models for robust time series forecasting.

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def build(self, hp: Any) -> Model:
        """Build and compile a hybrid Transformer-SSM model.

        The model architecture is determined by the following hyperparameters:
          - 'embed_dim': Embedding dimension (32, 64, 128).
          - 'num_heads': Number of attention heads (2, 4, 8).
          - 'd_state': State space dimension (32, 64, 128).
          - 'dropout_rate': Dropout rate for regularization (0.1-0.3).
          - 'learning_rate': Learning rate for the Adam optimizer.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Model: A compiled Keras Model.
        """
        embed_dim = hp.Choice("embed_dim", [32, 64, 128])
        num_heads = hp.Choice("num_heads", [2, 4, 8])
        d_state = hp.Choice("d_state", [32, 64, 128])
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.3, step=0.1)
        learning_rate = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

        model = build_hybrid_transformer_ssm_model(
            observation_window=OBSERVATION_WINDOW,
            n_variables=self.n_variables,
            forecast_horizon=FORECAST_HORIZON,
            embed_dim=embed_dim,
            num_heads=num_heads,
            d_state=d_state,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )

        return model
