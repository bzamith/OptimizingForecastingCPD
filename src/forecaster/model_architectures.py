"""Reusable custom layers for time series forecasting models.

This module implements custom Keras layers that can be shared across
different model architectures:
- PositionalEncoding: Adds positional information for transformers
- TransformerEncoderBlock: Multi-head attention with feed-forward network
- S4Layer: Structured State Space layer for sequence modeling
- SeasonalTrendDecomposition: Decomposes time series into seasonal and trend components
- TimestampEncoding: Encodes temporal information (hour, day, weekday, month)
- PatchEmbedding: Patch-based embedding for efficient sequence processing
- BidirectionalS4Block: Bidirectional S4 processing inspired by TSMamba
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class SeasonalTrendDecomposition(layers.Layer):
    """Seasonal-Trend decomposition layer using moving average.

    This layer decomposes a time series into seasonal and trend components
    using a moving average filter. According to the survey paper (Table 4),
    this decomposition can boost model performance by 50-80%.

    The trend is extracted using average pooling, and the seasonal component
    is the residual (original - trend).
    """

    def __init__(self, kernel_size=25, **kwargs):
        """Initialize the decomposition layer.

        Args:
            kernel_size (int): Size of the moving average window for trend extraction.
                             Default is 25 (roughly one month for daily data).
        """
        super().__init__(**kwargs)
        self.kernel_size = kernel_size

    def build(self, input_shape):
        self.avg_pool = layers.AveragePooling1D(
            pool_size=self.kernel_size, strides=1, padding="same"
        )

    def call(self, inputs):
        """Decompose input into seasonal and trend components.

        Args:
            inputs: Input tensor of shape (batch, time_steps, features)

        Returns:
            Tuple of (seasonal, trend) tensors with same shape as input
        """
        # Extract trend using moving average
        trend = self.avg_pool(inputs)

        # Seasonal component is the residual
        seasonal = inputs - trend

        return seasonal, trend

    def get_config(self):
        config = super().get_config()
        config.update({"kernel_size": self.kernel_size})
        return config


class TimestampEncoding(layers.Layer):
    """Timestamp encoding layer for temporal features.

    Encodes calendar information (month, day, weekday, hour) as learnable
    embeddings. This is particularly useful for real-world time series with
    calendar patterns, as suggested in the survey paper (Section 4.1).
    """

    def __init__(self, d_model, **kwargs):
        """Initialize the timestamp encoding layer.

        Args:
            d_model (int): Dimension of the embedding for each time feature.
        """
        super().__init__(**kwargs)
        self.d_model = d_model

        # Embedding layers for different time granularities
        self.month_embed = layers.Embedding(12, d_model)  # 0-11
        self.day_embed = layers.Embedding(31, d_model)  # 0-30
        self.weekday_embed = layers.Embedding(7, d_model)  # 0-6
        self.hour_embed = layers.Embedding(24, d_model)  # 0-23

    def call(self, time_features):
        """Encode timestamp features.

        Args:
            time_features: Tensor of shape (batch, time_steps, 4) containing
                         [month, day, weekday, hour] as integers.

        Returns:
            Encoded timestamps of shape (batch, time_steps, d_model)
        """
        # Extract individual time components
        months = tf.cast(time_features[..., 0], tf.int32)
        days = tf.cast(time_features[..., 1], tf.int32)
        weekdays = tf.cast(time_features[..., 2], tf.int32)
        hours = tf.cast(time_features[..., 3], tf.int32)

        # Embed each component
        month_emb = self.month_embed(months)
        day_emb = self.day_embed(days)
        weekday_emb = self.weekday_embed(weekdays)
        hour_emb = self.hour_embed(hours)

        # Combine embeddings (sum)
        timestamp_encoding = month_emb + day_emb + weekday_emb + hour_emb

        return timestamp_encoding

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model})
        return config


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
    """Transformer encoder block with multi-head attention and feed-forward network.

    Supports both standard O(N²) attention and sparse O(N log N) attention patterns
    for improved efficiency on long sequences.
    """

    def __init__(
        self, embed_dim, num_heads, ff_dim, dropout_rate=0.1, use_sparse_attention=False, **kwargs
    ):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        self.dropout_rate = dropout_rate
        self.use_sparse_attention = use_sparse_attention

        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(ff_dim, activation="relu"),
                layers.Dense(embed_dim),
            ]
        )

        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def build(self, input_shape):
        # input_shape: (batch, seq_len, embed_dim)
        self.seq_len = input_shape[1]
        super().build(input_shape)

    def _create_log_sparse_mask(self, seq_len):
        """
        Create a log-sparse attention mask using TensorFlow ops only.
        Output shape: (1, seq_len, seq_len)
        """
        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]

        # Allow attention to self and log-spaced positions
        diff = tf.abs(i - j)
        is_power_of_two = tf.logical_and(diff > 0, tf.equal(diff & (diff - 1), 0))

        mask = tf.logical_or(tf.equal(i, j), is_power_of_two)

        # Convert to additive attention mask: 0 for keep, -inf for mask out
        mask = tf.where(mask, 0.0, -1e9)
        mask = tf.expand_dims(mask, axis=0)
        return mask

    def call(self, inputs, training=False):
        if self.use_sparse_attention:
            seq_len = tf.shape(inputs)[1]
            attention_mask = self._create_log_sparse_mask(seq_len)
        else:
            attention_mask = None

        attn_output = self.att(
            inputs,
            inputs,
            attention_mask=attention_mask,
            training=training,
        )
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)

        ffn_output = self.ffn(out1)
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
                "use_sparse_attention": self.use_sparse_attention,
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

        step = tf.exp(self.log_step)
        dA = tf.eye(self.d_state) + step * self.A
        dB = step * self.B

        # initial state: (batch, d_state)
        init_state = tf.zeros((batch_size, self.d_state))

        # time-major: (seq_len, batch, d_model)
        inputs_t = tf.transpose(inputs, [1, 0, 2])

        def step_fn(state, x_t):
            # Update state
            new_state = tf.matmul(state, dA, transpose_b=True) + tf.matmul(
                x_t, dB, transpose_b=True
            )
            return new_state

        # Run scan over states only: (seq_len, batch, d_state)
        states = tf.scan(step_fn, inputs_t, initializer=init_state)

        # Compute outputs from states: y_t = C x + D u
        # states: (seq_len, batch, d_state) → (batch, seq_len, d_state)
        states = tf.transpose(states, [1, 0, 2])

        outputs = tf.matmul(states, self.C, transpose_b=True) + inputs * self.D
        return outputs

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "d_state": self.d_state,
            }
        )
        return config


class PatchEmbedding(layers.Layer):
    """Patch embedding layer for efficient sequence processing.

    Divides the input sequence into non-overlapping patches and embeds them
    using 1D convolution. This is inspired by PatchTST and used in TSMamba
    for improved efficiency and performance.
    """

    def __init__(self, d_model, patch_length=16, **kwargs):
        """Initialize the patch embedding layer.

        Args:
            d_model (int): Output dimension for each patch.
            patch_length (int): Length of each patch.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.patch_length = patch_length

    def build(self, input_shape):
        # 1D convolution for patch embedding
        self.patch_conv = layers.Conv1D(
            filters=self.d_model,
            kernel_size=self.patch_length,
            strides=self.patch_length,
            padding="valid",
        )

    def call(self, inputs):
        """Create patch embeddings from input sequence.

        Args:
            inputs: Tensor of shape (batch, time_steps, features)

        Returns:
            Patched embeddings of shape (batch, num_patches, d_model)
        """
        return self.patch_conv(inputs)

    def get_config(self):
        config = super().get_config()
        config.update({"d_model": self.d_model, "patch_length": self.patch_length})
        return config


class BidirectionalS4Block(layers.Layer):
    """Bidirectional S4 block inspired by TSMamba architecture.

    Processes sequences in both forward and backward directions using S4 layers,
    then combines the representations. This captures temporal dependencies from
    both past and future contexts.
    """

    def __init__(self, d_model, d_state=64, dropout_rate=0.1, **kwargs):
        """Initialize the bidirectional S4 block.

        Args:
            d_model (int): Model dimension.
            d_state (int): State space dimension.
            dropout_rate (float): Dropout rate for regularization.
        """
        super().__init__(**kwargs)
        self.d_model = d_model
        self.d_state = d_state
        self.dropout_rate = dropout_rate

    def build(self, input_shape):
        # Forward S4 layer
        self.forward_s4 = S4Layer(self.d_model, self.d_state)

        # Backward S4 layer (separate parameters)
        self.backward_s4 = S4Layer(self.d_model, self.d_state)

        # Temporal alignment convolution (aligns forward and backward representations)
        self.align_conv = layers.Conv1D(self.d_model, kernel_size=3, padding="same")

        # Layer normalization
        self.layernorm_forward = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_backward = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm_combined = layers.LayerNormalization(epsilon=1e-6)

        # Dropout
        self.dropout = layers.Dropout(self.dropout_rate)

        # Feed-forward network
        self.ffn = tf.keras.Sequential(
            [
                layers.Dense(self.d_model * 2, activation="gelu"),
                layers.Dropout(self.dropout_rate),
                layers.Dense(self.d_model),
            ]
        )
        self.layernorm_ffn = layers.LayerNormalization(epsilon=1e-6)

    def call(self, inputs, training=False):
        """Process inputs bidirectionally.

        Args:
            inputs: Tensor of shape (batch, time_steps, d_model)
            training: Boolean for training mode

        Returns:
            Bidirectionally processed tensor of shape (batch, time_steps, d_model)
        """
        # Forward pass
        forward = self.forward_s4(inputs, training=training)
        forward = self.layernorm_forward(forward)
        forward = self.dropout(forward, training=training)

        # Backward pass (reverse sequence, process, reverse back)
        backward = tf.reverse(inputs, axis=[1])
        backward = self.backward_s4(backward, training=training)
        backward = tf.reverse(backward, axis=[1])  # Reverse back to original order
        backward = self.align_conv(backward)  # Temporal alignment
        backward = self.layernorm_backward(backward)
        backward = self.dropout(backward, training=training)

        # Combine forward and backward with residual connection
        combined = inputs + forward + backward
        combined = self.layernorm_combined(combined)

        # Feed-forward network with residual
        ffn_output = self.ffn(combined, training=training)
        output = self.layernorm_ffn(combined + ffn_output)

        return output

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                "d_model": self.d_model,
                "d_state": self.d_state,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
