"""State-of-the-art model architectures for time series forecasting.

This module implements modern architectures including:
- Transformer-based models with temporal attention
- State Space Models (SSMs) inspired by S4/Mamba architectures
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Model


class PositionalEncoding(layers.Layer):
    """Positional encoding layer for transformer models.

    Adds positional information to input embeddings using sine and cosine functions.
    """

    def __init__(self, max_position=1000, **kwargs):
        super().__init__(**kwargs)
        self.max_position = max_position

    def build(self, input_shape):
        seq_len = input_shape[1]
        d_model = input_shape[2]

        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs):
        return inputs + self.pos_encoding[:, : tf.shape(inputs)[1], :]

    def get_config(self):
        config = super().get_config()
        config.update({"max_position": self.max_position})
        return config


class TransformerEncoderBlock(layers.Layer):
    """Transformer encoder block with multi-head attention and feed-forward network."""

    def __init__(self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate

        self.att = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim // num_heads, dropout=dropout_rate
        )
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dropout(dropout_rate),
                layers.Dense(embed_dim),
            ]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, inputs, training=False):
        # Multi-head attention
        attn_output = self.att(inputs, inputs, training=training)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        # Feed-forward network
        ffn_output = self.ffn(out1, training=training)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "embed_dim": self.embed_dim,
                "num_heads": self.num_heads,
                "ff_dim": self.ff_dim,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config


class S4Layer(layers.Layer):
    """Structured State Space (S4) layer for sequence modeling.

    Implements a simplified S4 layer inspired by the S4 and Mamba architectures.
    Uses a state space representation with learnable parameters.
    """

    def __init__(self, d_model, d_state=64, **kwargs):
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state

    def build(self, input_shape):
        # State space parameters
        self.A = self.add_weight(
            name="A",
            shape=(self.d_state, self.d_state),
            initializer=tf.keras.initializers.Orthogonal(),
            trainable=True,
        )
        self.B = self.add_weight(
            name="B",
            shape=(self.d_state, self.d_model),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.C = self.add_weight(
            name="C",
            shape=(self.d_model, self.d_state),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )
        self.D = self.add_weight(
            name="D",
            shape=(self.d_model,),
            initializer=tf.keras.initializers.Zeros(),
            trainable=True,
        )

        # Learnable step size
        self.log_step = self.add_weight(
            name="log_step",
            shape=(),
            initializer=tf.keras.initializers.Constant(-3.0),
            trainable=True,
        )

    def call(self, inputs, training=False):
        batch_size = tf.shape(inputs)[0]
        seq_len = tf.shape(inputs)[1]

        # Discretize continuous system
        step = tf.exp(self.log_step)
        dA = tf.eye(self.d_state) + step * self.A
        dB = step * self.B

        # Initialize state
        state = tf.zeros((batch_size, self.d_state))
        outputs = []

        # Recurrent computation
        for t in range(seq_len):
            x_t = inputs[:, t, :]  # (batch, d_model)

            # State update: x_{t+1} = A x_t + B u_t
            state = tf.matmul(state, dA, transpose_b=True) + tf.matmul(x_t, dB, transpose_b=True)

            # Output: y_t = C x_t + D u_t
            y_t = tf.matmul(state, self.C, transpose_b=True) + x_t * self.D
            outputs.append(y_t)

        return tf.stack(outputs, axis=1)

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "d_state": self.d_state,
            }
        )
        return config


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
):
    """Build a Transformer-based forecasting model.

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

    Returns:
        Model: Compiled Keras model.
    """
    inputs = layers.Input(shape=(observation_window, n_variables))

    # Project input to embedding dimension
    x = layers.Dense(embed_dim)(inputs)

    # Add positional encoding
    x = PositionalEncoding()(x)

    # Transformer encoder blocks
    for _ in range(num_transformer_blocks):
        x = TransformerEncoderBlock(embed_dim, num_heads, ff_dim, dropout_rate)(x)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Dropout for regularization
    x = layers.Dropout(dropout_rate)(x)

    # Output projection
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n_variables * forecast_horizon)(x)
    outputs = layers.Reshape((forecast_horizon, n_variables))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mean_squared_error",
    )

    return model


def build_ssm_model(
    observation_window,
    n_variables,
    forecast_horizon,
    d_model=64,
    d_state=64,
    num_ssm_blocks=2,
    dropout_rate=0.1,
    learning_rate=1e-3,
):
    """Build a State Space Model (SSM) for forecasting.

    Args:
        observation_window (int): Number of time steps in input sequence.
        n_variables (int): Number of variables/features per time step.
        forecast_horizon (int): Number of time steps to forecast.
        d_model (int): Model dimension.
        d_state (int): State space dimension.
        num_ssm_blocks (int): Number of S4 layers.
        dropout_rate (float): Dropout rate.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        Model: Compiled Keras model.
    """
    inputs = layers.Input(shape=(observation_window, n_variables))

    # Project input to model dimension
    x = layers.Dense(d_model)(inputs)

    # Stack S4 layers with residual connections
    for i in range(num_ssm_blocks):
        residual = x
        x = S4Layer(d_model, d_state)(x)
        x = layers.LayerNormalization(epsilon=1e-6)(x)
        x = layers.Dropout(dropout_rate)(x)

        # Add residual connection
        if i > 0:
            x = x + residual

        # Optional feed-forward layer
        ffn = layers.Dense(d_model * 2, activation="gelu")(x)
        ffn = layers.Dropout(dropout_rate)(ffn)
        ffn = layers.Dense(d_model)(ffn)
        x = layers.LayerNormalization(epsilon=1e-6)(x + ffn)

    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)

    # Output projection
    x = layers.Dense(256, activation="gelu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n_variables * forecast_horizon)(x)
    outputs = layers.Reshape((forecast_horizon, n_variables))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mean_squared_error",
    )

    return model


def build_hybrid_transformer_ssm_model(
    observation_window,
    n_variables,
    forecast_horizon,
    embed_dim=64,
    num_heads=4,
    d_state=64,
    dropout_rate=0.1,
    learning_rate=1e-3,
):
    """Build a hybrid model combining Transformer and SSM layers.

    This model uses SSM layers for capturing long-range dependencies and
    Transformer layers for complex temporal patterns.

    Args:
        observation_window (int): Number of time steps in input sequence.
        n_variables (int): Number of variables/features per time step.
        forecast_horizon (int): Number of time steps to forecast.
        embed_dim (int): Embedding dimension.
        num_heads (int): Number of attention heads.
        d_state (int): State space dimension.
        dropout_rate (float): Dropout rate.
        learning_rate (float): Learning rate for optimizer.

    Returns:
        Model: Compiled Keras model.
    """
    inputs = layers.Input(shape=(observation_window, n_variables))

    # Project input
    x = layers.Dense(embed_dim)(inputs)

    # S4 layer for long-range dependencies
    x = S4Layer(embed_dim, d_state)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Transformer for complex patterns
    x = PositionalEncoding()(x)
    x = TransformerEncoderBlock(embed_dim, num_heads, embed_dim * 2, dropout_rate)(x)

    # Another S4 layer
    x = S4Layer(embed_dim, d_state)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(dropout_rate)(x)

    # Global pooling and output
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(256, activation="gelu")(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(n_variables * forecast_horizon)(x)
    outputs = layers.Reshape((forecast_horizon, n_variables))(x)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0),
        loss="mean_squared_error",
    )

    return model
