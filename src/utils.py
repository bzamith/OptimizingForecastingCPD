from typing import List

import numpy as np

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error
)


def _wape(y_true: np.array, y_pred: np.array) -> float:
    """Compute the Weighted Absolute Percentage Error (WAPE).

    WAPE is defined as the sum of the absolute errors divided by the sum of the absolute true values.
    If the sum of the absolute true values is zero, this function returns NaN to avoid division by zero.

    Args:
        y_true (np.ndarray): Array of true values.
        y_pred (np.ndarray): Array of predicted values.

    Returns:
        float: The computed WAPE value.
    """
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)

    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan  # Avoid division by zero

    return np.sum(np.abs(y_true - y_pred)) / denominator


def get_error_results(y_true: pd.DataFrame, y_pred: pd.DataFrame, variables: List[str]) -> dict:
    """Calculate error metrics for true and predicted values.

    Computes overall error metrics (MAPE, MAE, MSE, RMSE, R2, WAPE) for all variables combined and for each
    individual variable provided.

    Args:
        y_true (pd.DataFrame): DataFrame containing the true values.
        y_pred (pd.DataFrame): DataFrame containing the predicted values.
        variables (List[str]): List of variable names corresponding to the columns in y_true and y_pred.

    Returns:
        dict: A dictionary with overall error metrics (prefixed with 'Avg_') and per-variable error metrics.
    """
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    results = {
        "Avg_MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "Avg_MAE": mean_absolute_error(y_true, y_pred),
        "Avg_MSE": mean_squared_error(y_true, y_pred),
        "Avg_RMSE": root_mean_squared_error(y_true, y_pred),
        "Avg_R2": r2_score(y_true, y_pred),
        "Avg_WAPE": _wape(y_true, y_pred),
    }

    for i in range(len(variables)):
        y_true_i = [sublist[i] for sublist in y_true]
        y_pred_i = [sublist[i] for sublist in y_pred]
        variable = variables[i]
        results.update({
            f"{variable}_MAPE": mean_absolute_percentage_error(y_true_i, y_pred_i),
            f"{variable}_MAE": mean_absolute_error(y_true_i, y_pred_i),
            f"{variable}_MSE": mean_squared_error(y_true_i, y_pred_i),
            f"{variable}_RMSE": root_mean_squared_error(y_true_i, y_pred_i),
            f"{variable}_R2": r2_score(y_true_i, y_pred_i),
            f"{variable}_WAPE": _wape(y_true_i, y_pred_i),
        })
    return results
