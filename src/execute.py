import math
import time
from typing import List, Tuple

import numpy as np

import pandas as pd

import ruptures as rpt

from statistics import mean, median

from src.forecaster import CNNForecaster
from src.scaler import MaxAbsScaler

from tensorflow.keras import metrics

from src.forecaster import OBSERVATION_WINDOW

CUT_COLUMN = "Cut"
CUT_SECONDS_COLUMN = "Cut Seconds"


def get_stacked_X_train(X_train: np.vstack, variables: List[str]) -> np.vstack:
    stack_list = []
    for col in variables:
        stack_list.append(X_train[col].values)
    return np.vstack(stack_list).T


def get_window_cut(X_train: pd.DataFrame, model: str, variables: List[str]) -> Tuple[int, float]:
    stack = get_stacked_X_train(X_train, variables)
    start_time = time.time()
    cut = rpt.Window(model=model, min_size=OBSERVATION_WINDOW + 1).fit_predict(stack, n_bkps=1)
    end_time = time.time()
    return cut[0], (end_time - start_time)


def get_binary_seg_cut(X_train: pd.DataFrame, model: str, variables: List[str]) -> Tuple[int, float]:
    stack = get_stacked_X_train(X_train, variables)
    start_time = time.time()
    cut = rpt.Binseg(model=model, min_size=OBSERVATION_WINDOW + 1).fit_predict(stack, n_bkps=1)
    end_time = time.time()
    return cut[0], (end_time - start_time)


def get_bottom_up_cut(X_train: pd.DataFrame, model: str, variables: List[str]) -> Tuple[int, float]:
    stack = get_stacked_X_train(X_train, variables)
    start_time = time.time()
    cut = rpt.BottomUp(model=model, min_size=OBSERVATION_WINDOW + 1).fit_predict(stack, n_bkps=1)
    end_time = time.time()
    return cut[0], (end_time - start_time)


def fit(X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[np.array, np.array, List[float], float]:
    scaler = MaxAbsScaler(variables)
    scaled_X_train = scaler.fit_scale(X_train)
    scaled_X_test = scaler.scale(X_test)

    cnn = CNNForecaster(variables)
    _, seconds = cnn.fit(scaled_X_train)
    y_pred, y_true = cnn.predict(scaled_X_test)
    metrics = cnn.evaluate(scaled_X_test)
    return y_true, y_pred, metrics, seconds


def get_execution_results(X_train: pd.DataFrame, X_test: pd.DataFrame, cut: int, cut_perc: float, cut_seconds: float, train_seconds: float) -> pd.DataFrame:
    df = {}
    df["# Train"] = [X_train.shape[0]]
    df["# Test"] = [X_test.shape[0]]
    cut_value = []
    if cut:
        cut_value = [cut]
    else:
        cut_value = ["N/A"]
    df[CUT_COLUMN] = cut_value
    cut_perc_value = []
    if cut_perc:
        cut_perc_value = [cut_perc]
    else:
        cut_perc_value = ["N/A"]
    df["% Cut"] = cut_perc_value
    cut_seconds_value = []
    if cut_seconds:
        cut_seconds_value = [cut_seconds]
    else:
        cut_seconds_value = ["N/A"]
    df[CUT_SECONDS_COLUMN] = cut_seconds_value
    df["Train Seconds"] = [train_seconds]
    total_seconds_value = []
    if cut_seconds:
        total_seconds_value = cut_seconds + train_seconds
        total_seconds_value = [total_seconds_value]
    else:
        total_seconds_value = [train_seconds]
    df["Total Seconds"] = total_seconds_value
    return pd.DataFrame(df)


def get_error_results(y_true: np.array, y_pred: np.array, variables: List[str]) -> pd.DataFrame:
    columns = []
    errors = []
    i = 0
    for column in variables:
        columns.append(column + "_MAPE")
        errors.append(metrics.mean_absolute_percentage_error(y_true[:, i], y_pred[:, i]).numpy())
        columns.append(column + "_MAE")
        errors.append(metrics.mean_absolute_error(y_true[:, i], y_pred[:, i]).numpy())
        columns.append(column + "_MSE")
        errors.append(metrics.mean_squared_error(y_true[:, i], y_pred[:, i]).numpy())
        i += 1
    columns.append("Avg_MAPE")
    errors.append(mean(metrics.mean_absolute_percentage_error(y_true, y_pred).numpy()))
    columns.append("Avg_MAE")
    errors.append(mean(metrics.mean_absolute_error(y_true, y_pred).numpy()))
    columns.append("Avg_MSE")
    errors.append(mean(metrics.mean_squared_error(y_true, y_pred).numpy()))
    df = pd.DataFrame(errors)
    df = df.T
    df.columns = columns
    return df


def execute_full(X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true, y_pred, metrics, train_seconds = fit(X_train, X_test, variables)
    execution_df = get_execution_results(X_train, X_test, None, None, None, train_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    return execution_df, errors_df


def execute_fixed_cut(cut_perc: float, X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut = math.floor(X_train.shape[0] * cut_perc)
    X_train = X_train[cut:]
    y_true, y_pred, metrics, train_seconds = fit(X_train, X_test, variables)
    cut_perc = (cut / X_train.shape[0]) * 100
    execution_df = get_execution_results(X_train, X_test, cut, cut_perc, None, train_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    return execution_df, errors_df


def execute_window_cut(method: str, X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut, cut_seconds = get_window_cut(X_train, method, variables)
    cut_X_train = X_train[cut:]
    y_true, y_pred, metrics, train_seconds = fit(cut_X_train, X_test, variables)
    cut_perc = (cut / X_train.shape[0]) * 100
    execution_df = get_execution_results(cut_X_train, X_test, cut, cut_perc, cut_seconds, train_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    return execution_df, errors_df


def execute_binary_seg_cut(method: str, X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut, cut_seconds = get_binary_seg_cut(X_train, method, variables)
    cut_X_train = X_train[cut:]
    y_true, y_pred, metrics, train_seconds = fit(cut_X_train, X_test, variables)
    cut_perc = (cut / X_train.shape[0]) * 100
    execution_df = get_execution_results(cut_X_train, X_test, cut, cut_perc, cut_seconds, train_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    return execution_df, errors_df


def execute_bottom_up_cut(method: str, X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut, cut_seconds = get_bottom_up_cut(X_train, method, variables)
    cut_X_train = X_train[cut:]
    y_true, y_pred, metrics, train_seconds = fit(cut_X_train, X_test, variables)
    cut_perc = (cut / X_train.shape[0]) * 100
    execution_df = get_execution_results(cut_X_train, X_test, cut, cut_perc, cut_seconds, train_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    return execution_df, errors_df


def execute_mean_cut(cuts: List[int], cut_seconds: List[float], X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_cut_seconds = sum(cut_seconds)
    cuts.sort()

    cut = math.floor(mean(cuts))
    mean_X_train = X_train[cut:]
    cut_perc = (cut / X_train.shape[0]) * 100

    y_true, y_pred, metrics, train_seconds = fit(mean_X_train, X_test, variables)
    execution_df = get_execution_results(mean_X_train, X_test, cut, cut_perc, total_cut_seconds, train_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    return execution_df, errors_df


def execute_median_cut(cuts: List[int], cut_seconds: List[float], X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    total_cut_seconds = sum(cut_seconds)
    cuts.sort()

    cut = math.floor(median(cuts))
    median_X_train = X_train[cut:]
    cut_perc = (cut / X_train.shape[0]) * 100

    y_true, y_pred, metrics, train_seconds = fit(median_X_train, X_test, variables)
    execution_df = get_execution_results(median_X_train, X_test, cut, cut_perc, total_cut_seconds, train_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    return execution_df, errors_df
