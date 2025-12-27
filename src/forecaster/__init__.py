"""Forecaster module with multiple model architectures.

This module provides a factory pattern for creating different types of
forecasting models including LSTM, Transformer, SSM, and other models.
"""

from src.forecaster.base_forecaster import BaseForecasterHyperModel, InternalForecaster
from src.forecaster.factory import ForecasterFactory, ForecasterType
from src.forecaster.lstm_forecaster import LSTMForecasterHyperModel
from src.forecaster.ssm_forecaster import SSMForecasterHyperModel
from src.forecaster.transformer_forecaster import TransformerForecasterHyperModel

__all__ = [
    "BaseForecasterHyperModel",
    "InternalForecaster",
    "LSTMForecasterHyperModel",
    "TransformerForecasterHyperModel",
    "SSMForecasterHyperModel",
    "ForecasterFactory",
    "ForecasterType",
]
