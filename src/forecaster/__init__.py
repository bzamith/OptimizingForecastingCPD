"""Forecaster module with multiple model architectures.

This module provides a factory pattern for creating different types of
forecasting models including LSTM, Transformer, SSM, and Hybrid models.
"""

from src.forecaster.arima_forecaster import ARIMAForecasterHyperModel
from src.forecaster.base_forecaster import BaseForecasterHyperModel, InternalForecaster
from src.forecaster.chronos_forecaster import ChronosForecasterHyperModel
from src.forecaster.factory import ForecasterFactory, ForecasterType
from src.forecaster.hybrid_forecaster import HybridForecasterHyperModel
from src.forecaster.lstm_forecaster import LSTMForecasterHyperModel
from src.forecaster.ssm_forecaster import SSMForecasterHyperModel
from src.forecaster.transformer_forecaster import TransformerForecasterHyperModel
from src.forecaster.ts2vec_forecaster import TS2VECForecasterHyperModel

__all__ = [
    "BaseForecasterHyperModel",
    "InternalForecaster",
    "LSTMForecasterHyperModel",
    "TransformerForecasterHyperModel",
    "SSMForecasterHyperModel",
    "HybridForecasterHyperModel",
    "ARIMAForecasterHyperModel",
    "ChronosForecasterHyperModel",
    "TS2VECForecasterHyperModel",
    "ForecasterFactory",
    "ForecasterType",
]
