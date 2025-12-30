"""Data Reader module for loading and processing time series datasets.

This module provides a factory pattern for working with different dataset domains
and includes utilities for reading, preprocessing, and splitting time series data.
"""

from src.data_reader.factory import (
    AUTOFORMERDatasets,
    DataReaderFactory,
    DatasetDomain,
    DummyDatasets,
    INMETDatasets,
    TCPDDatasets,
    UCIDatasets,
)
from src.data_reader.utils import (
    create_missing_mask_for_y,
    fill_na,
    read_dataset,
    split_train_test,
    split_X_y,
)

__all__ = [
    "DatasetDomain",
    "INMETDatasets",
    "AUTOFORMERDatasets",
    "UCIDatasets",
    "TCPDDatasets",
    "DummyDatasets",
    "DataReaderFactory",
    "read_dataset",
    "split_train_test",
    "split_X_y",
    "fill_na",
    "create_missing_mask_for_y",
    "rolling_window_split",
    "simple_rolling_window_split",
]
