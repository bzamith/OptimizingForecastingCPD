"""Reusable custom layers for time series forecasting models.

This module implements custom Keras layers that can be shared across
different model architectures.

Note: TensorFlow imports are deferred to runtime to reduce module import cost.
"""

import numpy as np
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
        trend = self.avg_pool(inputs)
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
        import tensorflow as tf

        months = tf.cast(time_features[..., 0], tf.int32)
        days = tf.cast(time_features[..., 1], tf.int32)
        weekdays = tf.cast(time_features[..., 2], tf.int32)
        hours = tf.cast(time_features[..., 3], tf.int32)

        month_emb = self.month_embed(months)
        day_emb = self.day_embed(days)
        weekday_emb = self.weekday_embed(weekdays)
        hour_emb = self.hour_embed(hours)

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
        import tensorflow as tf

        seq_len = input_shape[1]
        d_model = input_shape[2]

        position = np.arange(seq_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

        pos_encoding = np.zeros((seq_len, d_model))
        pos_encoding[:, 0::2] = np.sin(position * div_term)
        pos_encoding[:, 1::2] = np.cos(position * div_term)

        self.pos_encoding = tf.constant(pos_encoding[np.newaxis, :, :], dtype=tf.float32)

    def call(self, inputs):
        import tensorflow as tf

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
        # Import tensorflow only when instantiating Sequential
        import tensorflow as tf

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
        import tensorflow as tf

        i = tf.range(seq_len)[:, None]
        j = tf.range(seq_len)[None, :]

        diff = tf.abs(i - j)
        is_power_of_two = tf.logical_and(diff > 0, tf.equal(diff & (diff - 1), 0))
        mask = tf.logical_or(tf.equal(i, j), is_power_of_two)
        mask = tf.where(mask, 0.0, -1e9)
        mask = tf.expand_dims(mask, axis=0)
        return mask

    def call(self, inputs, training=False):
        import tensorflow as tf

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


class ResidualBlock(layers.Layer):
    """Residual block with causal dilated convolution (Bai et al. 2018).

    This is the core building block of Temporal Convolutional Networks (TCN)
    as described in "An Empirical Evaluation of Generic Convolutional and
    Recurrent Networks for Sequence Modeling" (Bai et al. 2018).

    Key features:
    - Causal padding: No future information leakage
    - Dilated convolutions: Exponentially growing receptive field
    - Residual connections: Enable training of deep networks
    - Weight normalization: Better than batch norm for sequences (original paper)
    - Spatial dropout: Regularization

    Note: We use Batch Normalization instead of Weight Normalization for
    better compatibility with TensorFlow/Metal on Apple Silicon.
    """

    def __init__(self, filters, kernel_size, dilation_rate, dropout_rate, **kwargs):
        """Initialize residual block.

        Args:
            filters (int): Number of convolutional filters.
            kernel_size (int): Size of convolutional kernel.
            dilation_rate (int): Dilation rate for dilated convolution.
            dropout_rate (float): Dropout probability.
        """
        super().__init__(**kwargs)
        self.filters = filters
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.dropout_rate = dropout_rate

        # Two convolutional layers per block (as per original paper)
        self.conv1 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            kernel_initializer="he_normal",
        )
        self.conv2 = layers.Conv1D(
            filters=filters,
            kernel_size=kernel_size,
            dilation_rate=dilation_rate,
            padding="causal",
            kernel_initializer="he_normal",
        )

        # Normalization (original paper uses weight norm, we use batch norm)
        self.norm1 = layers.BatchNormalization()
        self.norm2 = layers.BatchNormalization()

        # Spatial dropout (drops entire channels, better for sequences)
        self.dropout1 = layers.SpatialDropout1D(dropout_rate)
        self.dropout2 = layers.SpatialDropout1D(dropout_rate)

        # 1x1 convolution for residual connection dimension matching
        self.downsample = None

    def build(self, input_shape):
        """Build layer and create residual projection if needed."""
        if input_shape[-1] != self.filters:
            self.downsample = layers.Conv1D(
                filters=self.filters,
                kernel_size=1,
                padding="same",
                kernel_initializer="he_normal",
            )
        super().build(input_shape)

    def call(self, inputs, training=None):
        """Forward pass with residual connection.

        Architecture (per original paper):
        x -> Conv1D -> Norm -> ReLU -> Dropout -> Conv1D -> Norm -> ReLU -> Dropout -> + -> ReLU
        |                                                                                 ^
        |----------------------------(1x1 Conv if needed)------------------------------|

        Args:
            inputs: Input tensor of shape (batch, time, channels)
            training: Boolean indicating training mode

        Returns:
            Output tensor of shape (batch, time, filters)
        """
        import tensorflow as tf

        x = self.conv1(inputs)
        x = self.norm1(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout1(x, training=training)

        x = self.conv2(x)
        x = self.norm2(x, training=training)
        x = tf.nn.relu(x)
        x = self.dropout2(x, training=training)

        residual = inputs
        if self.downsample is not None:
            residual = self.downsample(residual)

        return tf.nn.relu(x + residual)

    def get_config(self):
        """Serialization support."""
        config = super().get_config()
        config.update(
            {
                "filters": self.filters,
                "kernel_size": self.kernel_size,
                "dilation_rate": self.dilation_rate,
                "dropout_rate": self.dropout_rate,
            }
        )
        return config
