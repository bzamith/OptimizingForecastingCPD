"""State Space Model (SSM) forecaster implementation."""

from typing import Any

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model

from config.constants import FORECAST_HORIZON, OBSERVATION_WINDOW
from src.forecaster.base_forecaster import BaseForecasterHyperModel
from src.forecaster.model_architectures import BidirectionalS4Block, PatchEmbedding, S4Layer


def build_ssm_model(
    observation_window,
    n_variables,
    forecast_horizon,
    d_model=64,
    d_state=64,
    num_ssm_blocks=2,
    dropout_rate=0.1,
    learning_rate=1e-3,
    use_bidirectional=True,
    use_patching=True,
    patch_length=16,
    channel_independent=False,
):
    """Build a State Space Model (SSM) for forecasting.

    This implementation supports both basic S4 and enhanced bidirectional S4
    with patching, inspired by TSMamba architecture (without true Mamba's
    selective mechanism, which requires custom CUDA kernels).

    Args:
        observation_window (int): Number of time steps in input sequence.
        n_variables (int): Number of variables/features per time step.
        forecast_horizon (int): Number of time steps to forecast.
        d_model (int): Model dimension.
        d_state (int): State space dimension.
        num_ssm_blocks (int): Number of S4 layers.
        dropout_rate (float): Dropout rate.
        learning_rate (float): Learning rate for optimizer.
        use_bidirectional (bool): Whether to use bidirectional S4 blocks.
        use_patching (bool): Whether to use patch embedding.
        patch_length (int): Length of each patch (if patching enabled).
        channel_independent (bool): Process each channel independently.

    Returns:
        Model: Compiled Keras model.
    """
    inputs = layers.Input(shape=(observation_window, n_variables))

    if channel_independent:
        # Process each channel independently (TSMamba style)
        channel_outputs = []
        for i in range(n_variables):
            # Extract single channel
            channel_input = layers.Lambda(lambda x: x[:, :, i : i + 1])(inputs)

            # Apply patching if enabled
            if use_patching and observation_window >= patch_length:
                x = PatchEmbedding(d_model, patch_length)(channel_input)
            else:
                x = layers.Dense(d_model)(channel_input)

            # Stack bidirectional or unidirectional S4 blocks
            if use_bidirectional:
                for _ in range(num_ssm_blocks):
                    x = BidirectionalS4Block(d_model, d_state, dropout_rate)(x)
            else:
                for j in range(num_ssm_blocks):
                    residual = x
                    x = S4Layer(d_model, d_state)(x)
                    x = layers.LayerNormalization(epsilon=1e-6)(x)
                    x = layers.Dropout(dropout_rate)(x)

                    if j > 0:
                        x = x + residual

                    # Feed-forward layer
                    ffn = layers.Dense(d_model * 2, activation="gelu")(x)
                    ffn = layers.Dropout(dropout_rate)(ffn)
                    ffn = layers.Dense(d_model)(ffn)
                    x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

            # Prediction head with compression (TSMamba style)
            x = layers.Dense(d_model // 4, activation="gelu")(x)
            x = layers.GlobalAveragePooling1D()(x)
            channel_output = layers.Dense(forecast_horizon)(x)
            channel_outputs.append(channel_output)

        # Stack channel predictions
        outputs = layers.Lambda(lambda xs: tf.stack(xs, axis=-1))(channel_outputs)

    else:
        # Standard multi-channel processing
        if use_patching and observation_window >= patch_length:
            x = PatchEmbedding(d_model, patch_length)(inputs)
        else:
            x = layers.Dense(d_model)(inputs)

        # Stack bidirectional or unidirectional S4 blocks
        if use_bidirectional:
            for _ in range(num_ssm_blocks):
                x = BidirectionalS4Block(d_model, d_state, dropout_rate)(x)
        else:
            for i in range(num_ssm_blocks):
                residual = x
                x = S4Layer(d_model, d_state)(x)
                x = layers.LayerNormalization(epsilon=1e-6)(x)
                x = layers.Dropout(dropout_rate)(x)

                if i > 0:
                    x = x + residual

                # Feed-forward layer
                ffn = layers.Dense(d_model * 2, activation="gelu")(x)
                ffn = layers.Dropout(dropout_rate)(ffn)
                ffn = layers.Dense(d_model)(ffn)
                x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

        # Global pooling
        x = layers.GlobalAveragePooling1D()(x)

        # Output projection with compression
        compression_dim = max(64, d_model // 4)
        x = layers.Dense(compression_dim, activation="gelu")(x)
        x = layers.Dropout(dropout_rate)(x)
        x = layers.Dense(n_variables * forecast_horizon)(x)
        outputs = layers.Reshape((forecast_horizon, n_variables))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mean_squared_error",
    )

    return model


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
          - 'use_bidirectional': Use bidirectional S4 blocks (True/False).
          - 'use_patching': Use patch embedding (True/False).
          - 'patch_length': Patch length if patching enabled.
          - 'channel_independent': Process channels independently (True/False).

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

        # SOTA features inspired by TSMamba
        use_bidirectional = hp.Boolean("use_bidirectional", default=True)
        use_patching = hp.Boolean("use_patching", default=False)
        patch_length = hp.Choice("patch_length", [8, 16, 32])
        channel_independent = hp.Boolean("channel_independent", default=False)

        model = build_ssm_model(
            observation_window=OBSERVATION_WINDOW,
            n_variables=self.n_variables,
            forecast_horizon=FORECAST_HORIZON,
            d_model=d_model,
            d_state=d_state,
            num_ssm_blocks=num_blocks,
            dropout_rate=dropout_rate,
            learning_rate=learning_rate,
            use_bidirectional=use_bidirectional,
            use_patching=use_patching,
            patch_length=patch_length,
            channel_independent=channel_independent,
        )

        return model
