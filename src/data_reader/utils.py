"""Utility functions for data reading and processing.

This module contains helper functions for reading datasets, filling missing values,
splitting data into train/test sets, and preparing data for time series forecasting.
"""

import math
from typing import List, Literal, Tuple, Union

import numpy as np
import pandas as pd

from config.constants import DATE_COLUMN, FORECAST_HORIZON, OBSERVATION_WINDOW, TRAIN_PERC
from src.data_reader.factory import (
    AUTOFORMERDatasets,
    DataReaderFactory,
    DatasetDomain,
    DummyDatasets,
    INMETDatasets,
    TCPDDatasets,
    UCIDatasets,
)


def fill_na(
    df: pd.DataFrame,
    variables: List[str],
    limit_direction: Literal["forward", "backward", "both"] = "forward",
) -> pd.DataFrame:
    """Fill missing values in specified columns of the DataFrame using linear interpolation.

    IMPORTANT: This function should be called separately on train and test sets AFTER splitting
    to avoid data leakage. Use limit_direction='forward' for training data and 'both' for test data.

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        variables (List[str]): List of column names to fill missing values.
        limit_direction (Literal["forward", "backward", "both"]): Direction for interpolation.
            'forward' only interpolates forward (no future leakage), 'backward' only backward,
            'both' in both directions. Default is 'forward' to prevent data leakage in time series.

    Returns:
        pd.DataFrame: The DataFrame with missing values filled in the specified columns.
    """
    df = df.copy()
    first_valid_index = 0
    for variable in variables:
        df[variable] = df[variable].interpolate(method="linear", limit_direction=limit_direction)
        first_valid_index = max(first_valid_index, df[variable].first_valid_index())

    if first_valid_index > 0:
        df = df.iloc[first_valid_index:].reset_index(drop=True)

    return df


def read_dataset(
    dataset_domain: Union[str, DatasetDomain],
    dataset: Union[
        str, INMETDatasets, AUTOFORMERDatasets, UCIDatasets, TCPDDatasets, DummyDatasets
    ],
) -> Tuple[pd.DataFrame, List[str]]:
    """Read a dataset based on the provided domain and dataset identifiers.

    The CSV file is expected to be located in the "datasets/{dataset_domain}" directory.
    The resulting DataFrame is filtered to include the date column and the specified variables.

    IMPORTANT: Missing values are NOT filled here to prevent data leakage.
    Use fill_na() separately on train and test sets after splitting.

    Args:
        dataset_domain (Union[str, DatasetDomain]): The domain of the dataset.
        dataset (Union[str, INMETDatasets, AUTOFORMERDatasets, UCIDatasets, TCPDDatasets, DummyDatasets]): The specific dataset to read.

    Returns:
        Tuple[pd.DataFrame, List[str]]: A tuple containing the DataFrame and a list of variable names.

    Raises:
        ValueError: If the dataset domain or dataset is not recognized.
    """
    # Convert string to DatasetDomain enum if needed
    if isinstance(dataset_domain, str):
        dataset_domain = DatasetDomain.from_str(dataset_domain)

    # Convert string to appropriate dataset enum if needed
    if isinstance(dataset, str):
        dataset = DataReaderFactory.get_dataset(dataset_domain, dataset)

    folder = f"datasets/{dataset_domain.value}"
    file = dataset.value[0]
    variables = dataset.value[1]

    df = pd.read_csv(f"{folder}/{file}")
    df = df[[DATE_COLUMN] + variables]

    return df, variables


def split_train_test(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split the DataFrame into training and testing sets based on a predefined training percentage.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing the training set and testing set DataFrames.
    """
    train_size = math.floor(df.shape[0] * TRAIN_PERC)
    train = df.iloc[:train_size].reset_index(drop=True)
    test = df.iloc[train_size:].reset_index(drop=True)
    return train, test


def split_X_y(df: pd.DataFrame) -> Tuple[np.array, np.array]:
    """Split the DataFrame into feature and target arrays for time series forecasting.

    The function drops the DATE_COLUMN (if present) and generates samples based on OBSERVATION_WINDOW and FORECAST_HORIZON.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        Tuple[np.array, np.array]: A tuple containing:
            - X (np.array): Feature array with shape (n_samples, OBSERVATION_WINDOW, n_features).
            - y (np.array): Target array with shape (n_samples, n_features).

    Notes:
        - The OBSERVATION_WINDOW constant defines the number of time steps in each input sample.
        - The FORECAST_HORIZON constant defines the number of future time steps to predict.
    """
    X, y = [], []
    if DATE_COLUMN in df.columns:
        df = df.drop(columns=DATE_COLUMN)
    for i in range(len(df) - OBSERVATION_WINDOW - FORECAST_HORIZON + 1):
        X.append(df.iloc[i : i + OBSERVATION_WINDOW].values)
        y.append(df.iloc[i + OBSERVATION_WINDOW : i + OBSERVATION_WINDOW + FORECAST_HORIZON].values)
    X = np.array(X)
    y = np.array(y)
    return X, y
