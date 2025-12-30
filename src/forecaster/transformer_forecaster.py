"""Transformer-based forecaster implementation."""

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from config.constants import (
    FORECAST_HORIZON,
    FORECASTER_LOSS,
    HP_DROPOUT_RATES,
    HP_LEARNING_RATES,
    HP_MODEL_DIMS,
    OBSERVATION_WINDOW,
)
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
    embed_dim,
    num_heads,
    num_transformer_blocks,
    ff_dim,
    dropout_rate,
    learning_rate,
    use_decomposition,
    decomp_kernel_size,
    use_sparse_attention,
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
        decomp = SeasonalTrendDecomposition(kernel_size=decomp_kernel_size)
        seasonal, trend = decomp(inputs)

        x_seasonal = layers.Dense(embed_dim)(seasonal)
        x_seasonal = PositionalEncoding()(x_seasonal)

        for _ in range(num_transformer_blocks):
            x_seasonal = TransformerEncoderBlock(
                embed_dim, num_heads, ff_dim, dropout_rate, use_sparse_attention
            )(x_seasonal)

        x_seasonal = layers.GlobalAveragePooling1D()(x_seasonal)
        x_seasonal = layers.Dropout(dropout_rate)(x_seasonal)

        x_trend = layers.Dense(embed_dim // 2)(trend)
        x_trend = layers.GlobalAveragePooling1D()(x_trend)
        x_trend = layers.Dropout(dropout_rate)(x_trend)

        x = layers.Concatenate()([x_seasonal, x_trend])
        combined_dim = embed_dim + embed_dim // 2

    else:
        x = layers.Dense(embed_dim)(inputs)
        x = PositionalEncoding()(x)

        for _ in range(num_transformer_blocks):
            x = TransformerEncoderBlock(
                embed_dim, num_heads, ff_dim, dropout_rate, use_sparse_attention
            )(x)

        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dropout(dropout_rate)(x)
        combined_dim = embed_dim

    x = layers.Dense(max(256, combined_dim), activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n_variables * forecast_horizon)(x)
    outputs = layers.Reshape((forecast_horizon, n_variables))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss=FORECASTER_LOSS,
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
          - 'embed_dim': Embedding dimension.
          - 'num_heads': Number of attention heads.
          - 'num_transformer_blocks': Number of transformer encoder blocks.
          - 'ff_dim': Feed-forward network dimension.
          - 'dropout_rate': Dropout rate for regularization.
          - 'learning_rate': Learning rate for the Adam optimizer.
          - 'use_sparse_attention': Whether to use sparse attention for efficiency.

        Note: Seasonal-trend decomposition  is always enabled
        based on survey paper findings (50-80% performance boost).

        Args:
            hp (Any): Hyperparameters used for model tuning.

        Returns:
            Model: A compiled Keras Model.
        """
        embed_dim = hp.Choice("embed_dim", HP_MODEL_DIMS)
        num_heads = hp.Choice("num_heads", [4, 8])
        num_blocks = hp.Int("num_transformer_blocks", 1, 3)
        ff_dim = hp.Choice("ff_dim", [128, 256])
        dropout_rate = hp.Choice("dropout_rate", HP_DROPOUT_RATES)
        learning_rate = hp.Choice("learning_rate", HP_LEARNING_RATES)

        use_decomposition = True
        decomp_kernel_size = 25
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
