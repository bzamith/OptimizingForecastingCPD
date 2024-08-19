import json
import os
import random
import sys
import time
from datetime import datetime
from typing import List

from keras_tuner import RandomSearch

import numpy as np

import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score
)

import tensorflow as tf

from config.constants import (
    FORECASTER_MODEL, FORECASTER_OBJECTIVE, NB_TRIALS,
    OBSERVATION_WINDOW, SEED, TRAIN_PERC
)

from src.cut_point_detector import CutPointMethod, CutPointModel, get_cut_point_detector
from src.dataset import read_dataset, split_X_y, split_train_test
from src.forecaster import InternalForecaster, TimeSeriesHyperModel
from src.scaler import Scaler

tf.get_logger().setLevel('ERROR')

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def get_error_results(y_true: pd.DataFrame, y_pred: pd.DataFrame, variables: List[str]) -> dict:
    y_true = np.array(y_true, dtype=np.float64)
    y_pred = np.array(y_pred, dtype=np.float64)

    results = {
        "Avg_MAPE": mean_absolute_percentage_error(y_true, y_pred),
        "Avg_MAE": mean_absolute_error(y_true, y_pred),
        "Avg_MSE": mean_squared_error(y_true, y_pred),
        "Avg_R2": r2_score(y_true, y_pred)
    }

    for i in range(len(variables)):
        y_true_i = [sublist[i] for sublist in y_true]
        y_pred_i = [sublist[i] for sublist in y_pred]
        variable = variables[i]
        results.update({
            f"{variable}_MAPE": mean_absolute_percentage_error(y_true_i, y_pred_i),
            f"{variable}_MAE": mean_absolute_error(y_true_i, y_pred_i),
            f"{variable}_MSE": mean_squared_error(y_true_i, y_pred_i),
            f"{variable}_R2": r2_score(y_true_i, y_pred_i),
        })
    return results


def run(timestamp: str, dataset_domain_argv: str, dataset_argv: str, cut_point_model: str, cut_point_method: str) -> None:
    execution_id = f"{timestamp}_{dataset_domain_argv}_{dataset_argv}_{cut_point_model}_{cut_point_method}"

    print(f"Extracting cut point model enum ({cut_point_model})")
    cut_point_model = CutPointModel.from_str(cut_point_model)

    print(f"Extracting cut point model enum ({cut_point_method})")
    cut_point_method = CutPointMethod.from_str(cut_point_method)

    print(f"Reading dataset {dataset_argv} from {dataset_domain_argv}")
    df, variables = read_dataset(dataset_domain_argv, dataset_argv)
    print(f"Variables: {variables}")

    print("Splitting data into train and test")
    train, test = split_train_test(df)

    print("Training and applying scaler")
    scaler = Scaler(variables)
    scaled_train = scaler.fit_scale(train)
    scaled_test = scaler.scale(test)

    print("Initializing report")
    cut_point_approach = f"{cut_point_model.value.title()} {cut_point_method.value.title()}"
    report = {
        'execution_id': execution_id,
        'timestamp': timestamp,
        'cut_point_model': cut_point_model.value,
        'cut_point_method': cut_point_method.value,
        'cut_point_approach': cut_point_approach,
        'seed': SEED,
        'forecaster_model': FORECASTER_MODEL,
        'forecaster_objective': FORECASTER_OBJECTIVE,
        'observation_window': OBSERVATION_WINDOW,
        'train_perc': TRAIN_PERC,
        'nb_trials': NB_TRIALS,
        'dataset_domain': dataset_domain_argv,
        'dataset': dataset_argv,
        'variables': variables,
        'dataset_shape': df.shape,
        'train_shape': train.shape,
        'test_shape': test.shape,
    }

    print(f"Started cut point for {cut_point_approach}")
    start_time = time.time()
    cut_point_detector = get_cut_point_detector(cut_point_model, cut_point_method)
    cut_point, cut_point_perc = cut_point_detector.find_cut_point(train, variables)
    end_time = time.time()
    cut_duration = end_time - start_time
    print(f"Cut point: {cut_point}, Cut point percentage: {cut_point_perc}")
    print(f"Finished cut point for {cut_point_approach}, duration: {cut_duration}")

    report.update({
        'cut_duration': cut_duration,
        'cut_point': str(cut_point),
        'cut_point_perc': str(cut_point_perc)
    })

    print("Applying subset to train based on cut point")
    reduced_scaled_train = cut_point_detector.apply_cut_point(scaled_train, cut_point)

    print("Splitting into X and y")
    X_reduced_scaled_train, y_reduced_scaled_train = split_X_y(reduced_scaled_train)
    X_scaled_test, y_scaled_test = split_X_y(scaled_test)

    print(f"Started running HPO and NAS for {cut_point_approach}")
    forecaster_hypermodel = TimeSeriesHyperModel(
        model_type=FORECASTER_MODEL,
        n_variables=len(variables)
    )
    forecaster_tuner = RandomSearch(
        forecaster_hypermodel,
        objective=FORECASTER_OBJECTIVE,
        max_trials=NB_TRIALS,
        directory=f"outputs/tuner/{execution_id}",
        project_name=f"{cut_point_model.value}/{cut_point_method.value}",
        seed=SEED,
        overwrite=True
    )
    start_time = time.time()
    forecaster_tuner.search(
        X_reduced_scaled_train,
        y_reduced_scaled_train,
        validation_split=(1 - TRAIN_PERC),
        shuffle=False
    )
    end_time = time.time()
    tuner_duration = end_time - start_time
    best_trial = forecaster_tuner.oracle.get_best_trials(num_trials=1)[0]
    best_forecaster_model = forecaster_tuner.get_best_models(num_models=1)[0]
    print(f"Finished running HPO and NAS for {cut_point_approach}, duration: {tuner_duration}")

    print(f"Trial ID: {best_trial.trial_id}")
    print(f"Hyperparameters: {best_trial.hyperparameters.values}")
    print(f"Score: {best_trial.score}")
    print("-" * 40)

    print("Retrieving best model")
    best_forecaster_model.summary()
    best_forecaster_model = InternalForecaster(best_forecaster_model)

    print("Running forecasting")
    y_scaled_pred = best_forecaster_model.forecast(X_scaled_test)

    print("Calculating error")
    y_test = scaler.descale(pd.DataFrame(y_scaled_test, columns=variables))
    y_pred = scaler.descale(pd.DataFrame(y_scaled_pred, columns=variables))
    error_results = get_error_results(y_test, y_pred, variables)
    print(f"Obtained error results: {error_results}")

    print("Writing report")
    report.update({
        'tuner_duration': tuner_duration,
        'total_duration': cut_duration + tuner_duration,
        'error_results': error_results,
        'reduced_scaled_train_shape': reduced_scaled_train.shape,
        'best_trial_id': best_trial.trial_id,
        'best_trial_hyperparameters': best_trial.hyperparameters.values,
        'best_trial_score': best_trial.score,
        'best_forecaster_model': best_forecaster_model.summary(),
    })
    report_path = f"outputs/report/{cut_point_model}/{cut_point_method}/{timestamp}"

    os.makedirs(report_path, exist_ok=True)
    with open(f"{report_path}/report.json", 'w') as file:
        json.dump(report, file, indent=4)

    print("Finished execution")


if __name__ == "__main__":
    dataset_domain_argv = sys.argv[1]
    dataset_argv = sys.argv[2]
    cut_point_model = sys.argv[3]
    cut_point_method = sys.argv[4]

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run(timestamp, dataset_domain_argv, dataset_argv, cut_point_model, cut_point_method)
