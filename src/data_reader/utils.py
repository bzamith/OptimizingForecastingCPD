"""Utility functions for data reading and processing.

This module contains helper functions for reading datasets, filling missing values,
splitting data into train/test sets, and preparing data for time series forecasting.
"""

import math
from typing import List, Tuple, Union

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
    df: Union[pd.Series, pd.DataFrame],
    variables: List[str],
) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
    """Fill missing values in specified columns using forward fill (last observation carried forward).

    This function uses forward fill (ffill) to avoid data leakage in time series data.
    Missing values are filled using only past observations, never future values.

    IMPORTANT: This function should be called BEFORE splitting to ensure consistent preprocessing
    across train and test sets while still avoiding data leakage (forward fill only uses past values).

    Args:
        df (pd.DataFrame): The DataFrame containing the data.
        variables (List[str]): List of column names to fill missing values.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing:
            - The DataFrame with missing values filled in the specified columns.
            - A boolean mask DataFrame indicating which values were originally missing (True = was missing).
    """
    df = df.copy()

    missing_mask = df[variables].isna()

    for variable in variables:
        df[variable] = df[variable].ffill()

    rows_to_keep = ~df[variables].isna().any(axis=1)  # type: ignore
    df = df[rows_to_keep].reset_index(drop=True)
    missing_mask = missing_mask[rows_to_keep].reset_index(drop=True)

    return df, missing_mask


def read_dataset(
    dataset_domain: Union[str, DatasetDomain],
    dataset: Union[
        str, INMETDatasets, AUTOFORMERDatasets, UCIDatasets, TCPDDatasets, DummyDatasets
    ],
) -> Tuple[Union[pd.Series, pd.DataFrame], List[str]]:
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
    if isinstance(dataset_domain, str):
        dataset_domain = DatasetDomain.from_str(dataset_domain)

    if isinstance(dataset, str):
        dataset = DataReaderFactory.get_dataset(dataset_domain, dataset)

    folder = f"datasets/{dataset_domain.value}"
    file = dataset.value[0]
    variables = dataset.value[1]

    df = pd.read_csv(f"{folder}/{file}")
    df = df[[DATE_COLUMN] + variables]

    return df, variables


def split_train_test(
    df: Union[pd.Series, pd.DataFrame]
) -> Tuple[Union[pd.Series, pd.DataFrame], Union[pd.Series, pd.DataFrame]]:
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


def create_missing_mask_for_y(
    missing_mask: Union[pd.Series, pd.DataFrame],
) -> np.ndarray:
    """Create a missing value mask for the y values (forecast targets) from split_X_y.

    This function processes the missing mask to match the shape of y values created by split_X_y.
    It extracts the forecast horizon portion for each window to track which target values were originally missing.

    Args:
        missing_mask (pd.DataFrame): Boolean mask indicating missing values (True = was missing).

    Returns:
        np.ndarray: Boolean mask array with shape matching y from split_X_y (n_samples, FORECAST_HORIZON, n_features).
    """
    y_masks = []
    if DATE_COLUMN in missing_mask.columns:
        missing_mask = missing_mask.drop(columns=DATE_COLUMN)
    for i in range(len(missing_mask) - OBSERVATION_WINDOW - FORECAST_HORIZON + 1):
        y_masks.append(
            missing_mask.iloc[
                i + OBSERVATION_WINDOW : i + OBSERVATION_WINDOW + FORECAST_HORIZON
            ].values
        )
    return np.array(y_masks)


def split_X_y(df: Union[pd.Series, pd.DataFrame]) -> Tuple[np.array, np.array]:
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
