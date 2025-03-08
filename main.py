import json
import os
import random
import sys
import time
from datetime import datetime

from keras_tuner import RandomSearch

import numpy as np

import pandas as pd

import tensorflow as tf

from config.constants import (
    NB_TRIALS, OBSERVATION_WINDOW,
    SEED, TRAIN_PERC
)

from src.change_point_detector import ChangePointCostFunction, ChangePointMethod, get_change_point_detector
from src.dataset import read_dataset, split_X_y, split_train_test
from src.forecaster import InternalForecaster, TimeSeriesHyperModel
from src.scaler import Scaler
from src.utils import get_error_results

tf.get_logger().setLevel('ERROR')
tf.config.set_visible_devices([], "GPU")
tf.config.experimental.set_memory_growth(tf.config.list_physical_devices("GPU")[0], True)

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def run(timestamp: str, dataset_domain_argv: str, dataset_argv: str,
        change_point_method_argv: str, change_point_cost_function_argv: str) -> None:
    """Execute the forecasting process with hyperparameter optimization and neural architecture search.

    Args:
        timestamp (str): Timestamp of the execution.
        dataset_domain_argv (str): Domain of the dataset.
        dataset_argv (str): Specific dataset to be used.
        change_point_method_argv (str): Identifier for the change point model.
        change_point_cost_function_argv (str): Identifier for the change point method.

    Returns:
        None
    """
    def save_report() -> None:
        with open(f"{report_path}/report.json", 'w') as file:
            json.dump(report, file, indent=4)

    execution_id = f"{timestamp}_{dataset_domain_argv}_{dataset_argv}_{change_point_method_argv}_{change_point_cost_function_argv}_{SEED}"
    change_point_method = ChangePointMethod.from_str(change_point_method_argv)
    change_point_cost_function = ChangePointCostFunction.from_str(change_point_cost_function_argv)
    change_point_approach = f"{change_point_method.value.title()} {change_point_cost_function.value.title()}"
    outputs_sub_path = f"seed={SEED}/{dataset_domain_argv}/{dataset_argv}/{change_point_method.value}/{change_point_cost_function.value}/{timestamp}"

    print(f"[Step 1] Reading dataset {dataset_argv} from {dataset_domain_argv}")
    df, variables = read_dataset(dataset_domain_argv, dataset_argv)
    print(f"Variables: {variables}")
    report_path = f"outputs/report/{outputs_sub_path}"
    os.makedirs(report_path, exist_ok=True)

    print("[Step 2] Splitting data into train_val and test")
    train_val, test = split_train_test(df)
    report = {
        'execution_id': execution_id,
        'timestamp': timestamp,
        'change_point_method': change_point_method.value,
        'change_point_cost_function': change_point_cost_function.value,
        'change_point_approach': change_point_approach,
        'seed': SEED,
        'observation_window': OBSERVATION_WINDOW,
        'train_perc': TRAIN_PERC,
        'nb_trials': NB_TRIALS,
        'dataset_domain': dataset_domain_argv,
        'dataset': dataset_argv,
        'variables': variables,
        'dataset_shape': df.shape,
        'train_val_shape': train_val.shape,
        'test_shape': test.shape,
    }
    save_report()

    print(f"[Step 3] Detecting change point ({change_point_approach})")
    start_time = time.time()
    change_point_detector = get_change_point_detector(change_point_method, change_point_cost_function)
    change_point, change_point_perc = change_point_detector.find_change_point(train_val, variables)
    end_time = time.time()
    detect_change_point_duration = end_time - start_time
    print(f"Change point: {change_point}, Change point percentage: {change_point_perc}")
    report.update({
        'detect_change_point_duration': detect_change_point_duration,
        'change_point': str(change_point),
        'change_point_perc': change_point_perc
    })
    save_report()

    print("[Step 4] Reducing train_val based on change point")
    start_time = time.time()
    reduced_train_val = change_point_detector.apply_change_point(train_val, change_point)
    end_time = time.time()
    apply_change_point_duration = end_time - start_time
    report.update({
        'apply_change_point_duration': apply_change_point_duration,
        'reduced_train_val.shape': reduced_train_val.shape,
    })
    save_report()

    print("[Step 5] Splitting train_val into train and val")
    reduced_train, reduced_val = split_train_test(reduced_train_val)
    report.update({
        'reduced_train.shape': reduced_train.shape,
        'reduced_val.shape': reduced_val.shape,
    })
    save_report()

    print("[Step 6] Fitting scaler on train and applying on train and val")
    start_time = time.time()
    scaler = Scaler(variables)
    scaled_reduced_train = scaler.fit_scale(reduced_train)
    scaled_reduced_val = scaler.scale(reduced_val)
    end_time = time.time()
    fit_apply_scaler_train_val_duration = end_time - start_time
    report.update({
        'fit_apply_scaler_train_val_duration': fit_apply_scaler_train_val_duration,
    })
    save_report()

    print("[Step 7] Splitting train and val into X and y")
    X_reduced_scaled_train, y_reduced_scaled_train = split_X_y(scaled_reduced_train)
    X_reduced_scaled_val, y_reduced_scaled_val = split_X_y(scaled_reduced_val)
    report.update({
        'X_reduced_scaled_train.shape': X_reduced_scaled_train.shape,
        'y_reduced_scaled_train.shape': y_reduced_scaled_train.shape,
        'X_reduced_scaled_val.shape': X_reduced_scaled_val.shape,
        'y_reduced_scaled_val.shape': y_reduced_scaled_val.shape,
    })
    save_report()

    print("[Step 8] Running HPO and NAS")
    n_variables = len(variables)
    forecaster_hypermodel = TimeSeriesHyperModel(
        n_variables=n_variables
    )
    forecaster_tuner = RandomSearch(
        forecaster_hypermodel,
        objective='val_loss',
        max_trials=NB_TRIALS,
        executions_per_trial=1,
        directory=f"outputs/tuner/{outputs_sub_path}",
        project_name=execution_id,
        seed=SEED,
        overwrite=True,
        distribution_strategy=tf.distribute.MirroredStrategy()
    )
    start_time = time.time()
    forecaster_tuner.search(
        X_reduced_scaled_train,
        y_reduced_scaled_train,
        validation_data=(X_reduced_scaled_val, y_reduced_scaled_val),
        shuffle=False,
    )
    end_time = time.time()
    tuner_duration = end_time - start_time
    report.update({
        'tuner_duration': tuner_duration
    })
    save_report()

    print("[Step 9] Retrieving best model")
    best_trial = forecaster_tuner.oracle.get_best_trials(num_trials=1)[0]
    best_forecaster_model = forecaster_tuner.get_best_models(num_models=1)[0]
    print(f"Trial ID: {best_trial.trial_id}")
    print(f"Hyperparameters: {best_trial.hyperparameters.values}")
    print(f"Score: {best_trial.score}")
    print("-" * 40)
    best_forecaster_model.summary()
    best_forecaster_model = InternalForecaster(
        best_forecaster_model,
        len(variables),
        best_trial.hyperparameters.values['batch_size'],
        best_trial.hyperparameters.values['epochs'],
    )
    report.update({
        'best_trial_id': best_trial.trial_id,
        'best_trial_hyperparameters': best_trial.hyperparameters.values,
        'best_trial_score': best_trial.score,
        'best_forecaster_model': best_forecaster_model.summary(),
    })
    save_report()

    print("[Step 10] Fitting scaler on train_val and applying on train_val and test")
    start_time = time.time()
    scaler = Scaler(variables)
    scaled_reduced_train_val = scaler.fit_scale(reduced_train_val)
    scaled_test = scaler.scale(test)
    end_time = time.time()
    fit_apply_scaler_train_val_test_duration = end_time - start_time
    report.update({
        'fit_apply_scaler_train_val_test_duration': fit_apply_scaler_train_val_test_duration,
    })
    save_report()

    print("[Step 11] Splitting train_val and test into X and y")
    X_reduced_scaled_train_val, y_reduced_scaled_train_val = split_X_y(scaled_reduced_train_val)
    X_scaled_test, y_scaled_test = split_X_y(scaled_test)
    report.update({
        'X_reduced_scaled_train_val.shape': X_reduced_scaled_train_val.shape,
        'y_reduced_scaled_train_val.shape': y_reduced_scaled_train_val.shape,
        'X_scaled_test.shape': X_scaled_test.shape,
        'y_scaled_test.shape': y_scaled_test.shape,
    })
    save_report()

    print("[Step 12] Retraining best model")
    start_time = time.time()
    best_forecaster_model.fit(
        X_reduced_scaled_train_val,
        y_reduced_scaled_train_val,
        shuffle=False
    )
    end_time = time.time()
    retrain_duration = end_time - start_time
    report.update({
        'retrain_duration': retrain_duration,
    })
    save_report()

    print("[Step 13] Forecasting for test")
    start_time = time.time()
    y_scaled_pred = best_forecaster_model.forecast(X_scaled_test)
    y_scaled_test_flat = y_scaled_test.reshape(-1, n_variables)
    y_scaled_pred_flat = y_scaled_pred.reshape(-1, n_variables)
    end_time = time.time()
    forecasting_test_duration = end_time - start_time
    report.update({
        'forecasting_test_duration': forecasting_test_duration,
    })
    save_report()

    print("[Step 14] Descaling data")
    start_time = time.time()
    y_test = scaler.descale(pd.DataFrame(y_scaled_test_flat, columns=variables))
    y_pred = scaler.descale(pd.DataFrame(y_scaled_pred_flat, columns=variables))
    end_time = time.time()
    descaling_duration = end_time - start_time
    report.update({
        'descaling_duration': descaling_duration,
    })
    save_report()

    print("[Step 15] Calculating evaluation metrics")
    total_duration = sum(value for key, value in report.items() if key.endswith('_duration'))
    error_results = get_error_results(y_test, y_pred, variables)
    print(f"Obtained error results: {error_results}")
    report.update({
        'total_duration': total_duration,
        'error_results': error_results,
    })
    save_report()

    print("Finished execution")


if __name__ == "__main__":
    dataset_domain_argv = sys.argv[1]
    dataset_argv = sys.argv[2]
    change_point_method_argv = sys.argv[3]
    change_point_cost_function_argv = sys.argv[4]

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run(timestamp, dataset_domain_argv, dataset_argv, change_point_method_argv, change_point_cost_function_argv)
