import math
import time
from statistics import mean, median
from typing import List, Tuple

import numpy as np

import pandas as pd

import ruptures as rpt

from src.dataset import TRAIN_PERC
from src.forecaster import Forecaster
from src.scaler import MaxAbsScaler

from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, mean_squared_error

from multiprocessing import Queue

OBSERVATION_WINDOW_GRID = [7, 14, 21]
NB_UNITS_GRID = [25, 50, 100]
TRAIN_BATCH_SIZE_GRID = [8, 16, 32]

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
    cut = rpt.Window(model=model, min_size=OBSERVATION_WINDOW_GRID[0] + 1).fit_predict(stack, n_bkps=1)
    end_time = time.time()
    return cut[0], (end_time - start_time)


def get_binary_seg_cut(X_train: pd.DataFrame, model: str, variables: List[str]) -> Tuple[int, float]:
    stack = get_stacked_X_train(X_train, variables)
    start_time = time.time()
    cut = rpt.Binseg(model=model, min_size=OBSERVATION_WINDOW_GRID[0] + 1).fit_predict(stack, n_bkps=1)
    end_time = time.time()
    return cut[0], (end_time - start_time)


def get_bottom_up_cut(X_train: pd.DataFrame, model: str, variables: List[str]) -> Tuple[int, float]:
    stack = get_stacked_X_train(X_train, variables)
    start_time = time.time()
    cut = rpt.BottomUp(model=model, min_size=OBSERVATION_WINDOW_GRID[0] + 1).fit_predict(stack, n_bkps=1)
    end_time = time.time()
    return cut[0], (end_time - start_time)


def fit(X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str], hpo: bool = False) -> Tuple[List[float], List[float], List[float], float, float]:
    scaler = MaxAbsScaler(variables)
    scaled_X_train = scaler.fit_scale(X_train)[variables]
    scaled_X_test = scaler.scale(X_test)[variables]

    if hpo:
        return hpo_fit(scaled_X_train, scaled_X_test)

    forecaster = Forecaster(OBSERVATION_WINDOW_GRID[0], NB_UNITS_GRID[0], TRAIN_BATCH_SIZE_GRID[0])
    train_seconds = forecaster.fit(scaled_X_train)
    y_true, y_pred = forecaster.predict(scaled_X_test)
    metrics = mean_squared_error(y_true, y_pred)
    return y_true, y_pred, metrics, train_seconds, None


def hpo_fit(X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[List[float], List[float], List[float], float, float]:
    index_split = math.floor(X_train.shape[0] * TRAIN_PERC)
    hpo_X_train = X_train[:index_split].reset_index(drop=True)
    hpo_X_val = X_train[index_split:].reset_index(drop=True)

    start_time = time.time()

    best_mse = float('inf')
    best_dict = {}

    for observation_window in OBSERVATION_WINDOW_GRID:
        for nb_units in NB_UNITS_GRID:
            for train_batch_size in TRAIN_BATCH_SIZE_GRID:
                try:
                    forecaster = Forecaster(observation_window, nb_units, train_batch_size)
                    _ = forecaster.fit(hpo_X_train)
                    y_true, y_pred = forecaster.predict(hpo_X_val)
                    curr_mse = mean_squared_error(y_true, y_pred)
                    if curr_mse < best_mse:
                        best_mse = curr_mse
                        best_dict = {
                            "observation_window": observation_window,
                            "nb_units": nb_units,
                            "train_batch_size": train_batch_size,
                        }
                except ValueError:
                    pass

    end_time = time.time()

    forecaster = Forecaster(best_dict["observation_window"], best_dict["nb_units"], best_dict["train_batch_size"])
    train_seconds = forecaster.fit(X_train)
    y_pred, y_true = forecaster.predict(X_test)
    metrics = mean_squared_error(y_pred, y_true)
    return y_true, y_pred, metrics, train_seconds, (end_time - start_time)


def get_execution_results(X_train: pd.DataFrame, X_test: pd.DataFrame, cut: int, cut_perc: float, cut_seconds: float, train_seconds: float, hpo_seconds: float) -> pd.DataFrame:
    df = {}
    df["# Train"] = [X_train.shape[0]]
    df["# Test"] = [X_test.shape[0]]
    df[CUT_COLUMN] = [cut] if cut else ["N/A"]
    df["% Cut"] = [cut_perc] if cut_perc else ["N/A"]
    df[CUT_SECONDS_COLUMN] = [cut_seconds] if cut_seconds else ["N/A"]
    if hpo_seconds:
        df["HPO Seconds"] = [hpo_seconds]
    df["Train Seconds"] = [train_seconds]
    total_seconds_value = train_seconds
    if cut_seconds:
        total_seconds_value += cut_seconds
    if hpo_seconds:
        total_seconds_value += hpo_seconds
    df["Total Seconds"] = [total_seconds_value]
    return pd.DataFrame(df)


def get_dummy_execution_results(X_train: pd.DataFrame, X_test: pd.DataFrame, cut: int, cut_perc: float, hpo: bool) -> pd.DataFrame:
    df = {}
    df["# Train"] = [X_train.shape[0]]
    df["# Test"] = [X_test.shape[0]]
    df[CUT_COLUMN] = [cut]
    df["% Cut"] = [cut_perc]
    df[CUT_SECONDS_COLUMN] = ["N/A"]
    if hpo:
        df["HPO Seconds"] = ["N/A"]
    df["Train Seconds"] = ["N/A"]
    df["Total Seconds"] = ["N/A"]
    return pd.DataFrame(df)


def get_error_results(y_true: np.array, y_pred: np.array, variables: List[str]) -> pd.DataFrame:
    columns = []
    errors = []
    for i in range(len(variables)):
        y_true_i = [sublist[i] for sublist in y_true]
        y_pred_i = [sublist[i] for sublist in y_pred]
        column = variables[i]
        columns.append(column + "_MAPE")
        errors.append(mean_absolute_percentage_error(y_true_i, y_pred_i))
        columns.append(column + "_MAE")
        errors.append(mean_absolute_error(y_true_i, y_pred_i))
        columns.append(column + "_MSE")
        errors.append(mean_squared_error(y_true_i, y_pred_i))
    columns.append("Avg_MAPE")
    errors.append(mean_absolute_percentage_error(y_true, y_pred))
    columns.append("Avg_MAE")
    errors.append(mean_absolute_error(y_true, y_pred))
    columns.append("Avg_MSE")
    errors.append(mean_squared_error(y_true, y_pred))
    df = pd.DataFrame(errors)
    df = df.T
    df.columns = columns
    return df


def get_dummy_error_results(variables: List[str]) -> pd.DataFrame:
    columns = []
    errors = []
    for i in range(len(variables)):
        column = variables[i]
        columns.append(column + "_MAPE")
        errors.append("N/A")
        columns.append(column + "_MAE")
        errors.append("N/A")
        columns.append(column + "_MSE")
        errors.append("N/A")
    columns.append("Avg_MAPE")
    errors.append("N/A")
    columns.append("Avg_MAE")
    errors.append("N/A")
    columns.append("Avg_MSE")
    errors.append("N/A")
    df = pd.DataFrame(errors)
    df = df.T
    df.columns = columns
    return df


def execute_full(X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str], hpo: bool = False, queue: Queue = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    y_true, y_pred, metrics, train_seconds, hpo_seconds = fit(X_train, X_test, variables, hpo)
    execution_df = get_execution_results(X_train, X_test, None, None, None, train_seconds, hpo_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    if queue:
        queue.put((execution_df, errors_df))
    return execution_df, errors_df


def execute_fixed_cut(cut_perc: float, X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str], hpo: bool = False, queue: Queue = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut = math.floor(X_train.shape[0] * cut_perc)
    cut_X_train = X_train[cut:]
    cut_perc = (cut / X_train.shape[0]) * 100
    try:
        y_true, y_pred, metrics, train_seconds, hpo_seconds = fit(cut_X_train, X_test, variables, hpo)
        execution_df = get_execution_results(cut_X_train, X_test, cut, cut_perc, None, train_seconds, hpo_seconds)
        errors_df = get_error_results(y_true, y_pred, variables)
    except KeyError:
        execution_df = get_dummy_execution_results(cut_X_train, X_test, cut, cut_perc, hpo)
        errors_df = get_dummy_error_results(variables)
    if queue:
        queue.put((execution_df, errors_df))
    return execution_df, errors_df


def execute_window_cut(method: str, X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str], hpo: bool = False, queue: Queue = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut, cut_seconds = get_window_cut(X_train, method, variables)
    cut_X_train = X_train[cut:]
    cut_perc = (cut / X_train.shape[0]) * 100
    if cut_X_train.shape[0] == 0:
        return get_dummy_execution_results(cut_X_train, X_test, cut, cut_perc, hpo), get_dummy_error_results(variables)
    try:
        y_true, y_pred, metrics, train_seconds, hpo_seconds = fit(cut_X_train, X_test, variables, hpo)
        execution_df = get_execution_results(cut_X_train, X_test, cut, cut_perc, cut_seconds, train_seconds, hpo_seconds)
        errors_df = get_error_results(y_true, y_pred, variables)
    except KeyError:
        execution_df = get_dummy_execution_results(cut_X_train, X_test, cut, cut_perc, hpo)
        errors_df = get_dummy_error_results(variables)
    if queue:
        queue.put((execution_df, errors_df))
    return execution_df, errors_df


def execute_binary_seg_cut(method: str, X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str], hpo: bool = False, queue: Queue = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut, cut_seconds = get_binary_seg_cut(X_train, method, variables)
    cut_X_train = X_train[cut:]
    cut_perc = (cut / X_train.shape[0]) * 100
    if cut_X_train.shape[0] == 0:
        return get_dummy_execution_results(cut_X_train, X_test, cut, cut_perc, hpo), get_dummy_error_results(variables)
    try:
        y_true, y_pred, metrics, train_seconds, hpo_seconds = fit(cut_X_train, X_test, variables, hpo)
        execution_df = get_execution_results(cut_X_train, X_test, cut, cut_perc, cut_seconds, train_seconds, hpo_seconds)
        errors_df = get_error_results(y_true, y_pred, variables)
    except KeyError:
        execution_df = get_dummy_execution_results(cut_X_train, X_test, cut, cut_perc, hpo)
        errors_df = get_dummy_error_results(variables)
    if queue:
        queue.put((execution_df, errors_df))
    return execution_df, errors_df


def execute_bottom_up_cut(method: str, X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str], hpo: bool = False, queue: Queue = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cut, cut_seconds = get_bottom_up_cut(X_train, method, variables)
    cut_X_train = X_train[cut:]
    cut_perc = (cut / X_train.shape[0]) * 100
    if cut_X_train.shape[0] == 0:
        return get_dummy_execution_results(cut_X_train, X_test, cut, cut_perc, hpo), get_dummy_error_results(variables)
    try:
        y_true, y_pred, metrics, train_seconds, hpo_seconds = fit(cut_X_train, X_test, variables, hpo)
        execution_df = get_execution_results(cut_X_train, X_test, cut, cut_perc, cut_seconds, train_seconds, hpo_seconds)
        errors_df = get_error_results(y_true, y_pred, variables)
    except KeyError:
        execution_df = get_dummy_execution_results(cut_X_train, X_test, cut, cut_perc, hpo)
        errors_df = get_dummy_error_results(variables)
    if queue:
        queue.put((execution_df, errors_df))
    return execution_df, errors_df


def execute_mean_cut(cuts: List[int], cut_seconds: List[float], X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str], hpo: bool = False, queue: Queue = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cuts = [x for x in cuts if x != "N/A"]
    cut_seconds = [x for x in cut_seconds if x != "N/A"]

    total_cut_seconds = sum(cut_seconds)

    cut = math.floor(mean(cuts))
    mean_X_train = X_train[cut:]
    cut_perc = (cut / X_train.shape[0]) * 100

    y_true, y_pred, metrics, train_seconds, hpo_seconds = fit(mean_X_train, X_test, variables, hpo)
    execution_df = get_execution_results(mean_X_train, X_test, cut, cut_perc, total_cut_seconds, train_seconds, hpo_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    if queue:
        queue.put((execution_df, errors_df))
    return execution_df, errors_df


def execute_median_cut(cuts: List[int], cut_seconds: List[float], X_train: pd.DataFrame, X_test: pd.DataFrame, variables: List[str], hpo: bool = False, queue: Queue = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    cuts = [x for x in cuts if x != "N/A"]
    cut_seconds = [x for x in cut_seconds if x != "N/A"]

    total_cut_seconds = sum(cut_seconds)

    cuts.sort()
    cut = math.floor(median(cuts))
    median_X_train = X_train[cut:]
    cut_perc = (cut / X_train.shape[0]) * 100

    y_true, y_pred, metrics, train_seconds, hpo_seconds = fit(median_X_train, X_test, variables, hpo)
    execution_df = get_execution_results(median_X_train, X_test, cut, cut_perc, total_cut_seconds, train_seconds, hpo_seconds)
    errors_df = get_error_results(y_true, y_pred, variables)
    if queue:
        queue.put((execution_df, errors_df))
    return execution_df, errors_df
