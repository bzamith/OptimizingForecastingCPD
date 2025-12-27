"""Transformer-based forecaster implementation."""

from typing import Any

from tensorflow.keras.models import Model

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel
from src.forecaster.model_architectures import build_transformer_model


class TransformerForecasterHyperModel(BaseForecasterHyperModel):
    """A HyperModel for building and training Transformer-based time series forecasting models.

    This HyperModel constructs a Transformer model with multi-head attention mechanisms
    optimized for time series forecasting tasks.

    Attributes:
        n_variables (int): The number of variables in the time series data.
    """

    def build(self, hp: Any) -> Model:
        """Build and compile a Transformer model based on provided hyperparameters.

        The model architecture is determined by the following hyperparameters:
          - 'embed_dim': Embedding dimension (32, 64, 128).
          - 'num_heads': Number of attention heads (2, 4, 8).
          - 'num_transformer_blocks': Number of transformer encoder blocks (1-3).
          - 'ff_dim': Feed-forward network dimension (64, 128, 256).
          - 'dropout_rate': Dropout rate for regularization (0.1-0.3).
          - 'learning_rate': Learning rate for the Adam optimizer.

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Model: A compiled Keras Model.
        """
        embed_dim = hp.Choice("embed_dim", [32, 64, 128])
        num_heads = hp.Choice("num_heads", [2, 4, 8])
        num_blocks = hp.Int("num_transformer_blocks", 1, 3)
        ff_dim = hp.Choice("ff_dim", [64, 128, 256])
        dropout_rate = hp.Float("dropout_rate", 0.1, 0.3, step=0.1)
        learning_rate = hp.Choice("learning_rate", [1e-2, 5e-3, 1e-3, 5e-4, 1e-4])

        model = build_transformer_model(
            observation_window=OBSERVATION_WINDOW,
            n_variables=self.n_variables,
            forecast_horizon=FORECAST_HORIZON,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_transformer_blocks=num_blocks,
            ff_dim=ff_dim,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
        )

        return model
