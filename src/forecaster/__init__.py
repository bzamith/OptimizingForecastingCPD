"""Forecaster module with multiple model architectures.

This module provides a factory pattern for creating different types of
forecasting models including LSTM, Transformer, TCN, and other models.
"""

from src.forecaster.arima_forecaster import ARIMAForecasterHyperModel
from src.forecaster.base_forecaster import BaseForecasterHyperModel, InternalForecaster
from src.forecaster.factory import ForecasterFactory, ForecasterType
from src.forecaster.gru_forecaster import GRUForecasterHyperModel
from src.forecaster.lstm_forecaster import LSTMForecasterHyperModel
from src.forecaster.tcn_forecaster import TCNForecasterHyperModel
from src.forecaster.transformer_forecaster import TransformerForecasterHyperModel

__all__ = [
    "BaseForecasterHyperModel",
    "InternalForecaster",
    "ARIMAForecasterHyperModel",
    "GRUForecasterHyperModel",
    "LSTMForecasterHyperModel",
    "TCNForecasterHyperModel",
    "TransformerForecasterHyperModel",
    "ForecasterFactory",
    "ForecasterType",
]
