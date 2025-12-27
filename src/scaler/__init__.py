"""Scaler module with multiple scaling methods.

This module provides a factory pattern for creating different types of
scalers including Standard, MinMax, Robust, MaxAbs, and None (no scaling).
"""

from src.scaler.base_scaler import BaseScaler
from src.scaler.factory import ScalerFactory, ScalerType
from src.scaler.maxabs_scaler import MaxAbsScaler
from src.scaler.minmax_scaler import MinMaxScaler
from src.scaler.none_scaler import NoneScaler
from src.scaler.robust_scaler import RobustScaler
from src.scaler.standard_scaler import StandardScaler

__all__ = [
    "BaseScaler",
    "ScalerType",
    "StandardScaler",
    "MinMaxScaler",
    "RobustScaler",
    "MaxAbsScaler",
    "NoneScaler",
    "ScalerFactory",
]
