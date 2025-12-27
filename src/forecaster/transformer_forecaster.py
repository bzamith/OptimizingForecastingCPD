"""Transformer-based forecaster implementation."""

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel
from src.forecaster.model_architectures import (
    PositionalEncoding,
    SeasonalTrendDecomposition,
    TransformerEncoderBlock,
)


def build_transformer_model(
    observation_window,
    n_variables,
    forecast_horizon,
    embed_dim=64,
    num_heads=4,
    num_transformer_blocks=2,
    ff_dim=128,
    dropout_rate=0.1,
    learning_rate=1e-3,
    use_decomposition=True,
    decomp_kernel_size=25,
    use_sparse_attention=True,
):
    """Build a Transformer-based forecasting model with seasonal-trend decomposition.

    This implementation follows SOTA practices from the transformer survey paper,
    including seasonal-trend decomposition which can improve performance by 50-80%
    (see paper Table 4) and optional sparse attention for O(N log N) complexity.

    Args:
        observation_window (int): Number of time steps in input sequence.
        n_variables (int): Number of variables/features per time step.
        forecast_horizon (int): Number of time steps to forecast.
        embed_dim (int): Dimension of embeddings and transformer layers.
        num_heads (int): Number of attention heads.
        num_transformer_blocks (int): Number of transformer encoder blocks.
        ff_dim (int): Dimension of feed-forward network.
        dropout_rate (float): Dropout rate.
        learning_rate (float): Learning rate for optimizer.
        use_decomposition (bool): Whether to use seasonal-trend decomposition.
        decomp_kernel_size (int): Kernel size for decomposition moving average.
        use_sparse_attention (bool): Whether to use sparse attention (O(N log N) vs O(N²)).

    Returns:
        Model: Compiled Keras model.
    """
    inputs = layers.Input(shape=(observation_window, n_variables))

    if use_decomposition:
        # Seasonal-Trend Decomposition (50-80% performance boost per survey paper)
        decomp = SeasonalTrendDecomposition(kernel_size=decomp_kernel_size)
        seasonal, trend = decomp(inputs)

        # Process seasonal component with Transformer (captures periodic patterns)
        x_seasonal = layers.Dense(embed_dim)(seasonal)
        x_seasonal = PositionalEncoding()(x_seasonal)

        for _ in range(num_transformer_blocks):
            x_seasonal = TransformerEncoderBlock(
                embed_dim, num_heads, ff_dim, dropout_rate, use_sparse_attention
            )(x_seasonal)

        x_seasonal = layers.GlobalAveragePooling1D()(x_seasonal)
        x_seasonal = layers.Dropout(dropout_rate)(x_seasonal)

        # Process trend component separately (simpler patterns)
        x_trend = layers.Dense(embed_dim // 2)(trend)
        x_trend = layers.GlobalAveragePooling1D()(x_trend)
        x_trend = layers.Dropout(dropout_rate)(x_trend)

        # Combine seasonal and trend representations
        x = layers.Concatenate()([x_seasonal, x_trend])
        combined_dim = embed_dim + embed_dim // 2

    else:
        # Standard transformer without decomposition (baseline)
        x = layers.Dense(embed_dim)(inputs)
        x = PositionalEncoding()(x)

        for _ in range(num_transformer_blocks):
            x = TransformerEncoderBlock(
                embed_dim, num_heads, ff_dim, dropout_rate, use_sparse_attention
            )(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout_rate)(x)
        combined_dim = embed_dim

    # Output projection
    x = layers.Dense(max(256, combined_dim), activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n_variables * forecast_horizon)(x)
    outputs = layers.Reshape((forecast_horizon, n_variables))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mean_squared_error",
    )

    return model


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
          - 'use_decomposition': Whether to use seasonal-trend decomposition (True/False).
          - 'decomp_kernel_size': Kernel size for decomposition (if enabled).

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

        # SOTA features from survey paper
        use_decomposition = hp.Boolean("use_decomposition", default=True)
        decomp_kernel_size = hp.Choice("decomp_kernel_size", [13, 25, 37])
        use_sparse_attention = hp.Boolean("use_sparse_attention", default=False)

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
            use_decomposition=use_decomposition,
            decomp_kernel_size=decomp_kernel_size,
            use_sparse_attention=use_sparse_attention,
        )

        return model
