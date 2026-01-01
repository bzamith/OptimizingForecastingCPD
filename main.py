import json
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime

from keras_tuner import BayesianOptimization

import numpy as np

import pandas as pd

import tensorflow as tf

from config.constants import (
    NB_TRIALS, OBSERVATION_WINDOW, TRAIN_PERC
)

from src.cpd import CPDCostFunction, CPDMethod, CPDDetectorFactory
from src.data_reader import create_missing_mask_for_y, fill_na, read_dataset, split_X_y, split_train_test
from src.forecaster import InternalForecaster, ForecasterFactory, ForecasterType
from src.scaler import ScalerFactory, ScalerType
from src.utils import get_error_results

# Suppress third-party library warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="keras.src.export.tf2onnx_lib")
warnings.filterwarnings("ignore", category=UserWarning, module="ruptures.costs.costnormal")

tf.get_logger().setLevel('ERROR')

# GPU Configuration: Enable GPU if available, otherwise use CPU
gpu_devices = tf.config.list_physical_devices("GPU")
if gpu_devices:
    # GPU available - enable memory growth to avoid OOM errors
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    # Enable mixed precision for faster training on GPU
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"Running on GPU: {len(gpu_devices)} device(s) detected")
else:
    # CPU-only mode - threading is controlled via environment variables
    # OMP_NUM_THREADS, TF_NUM_INTRAOP_THREADS set in experiment_executions.sh
    # Mixed precision disabled on CPU (provides no benefit, may slow down)
    print("Running on CPU: Using all available cores")

def run(timestamp: str,
        dataset_domain_argv: str,
        dataset_name_argv: str,
        cpd_method_argv: str,
        cpd_cost_function_argv: str,
        forecaster_type_argv: str,
        seed: int = 42) -> None:
    """Execute the forecasting process with hyperparameter optimization and neural architecture search.

    Args:
        timestamp (str): Timestamp of the execution.
        dataset_domain_argv (str): Domain of the dataset (e.g., 'TCPD').
        dataset_name_argv (str): Specific dataset to be used (e.g., 'APPLE').
        cpd_method_argv (str): Identifier for the change point method (e.g., 'Window').
        cpd_cost_function_argv (str): Identifier for the change point cost function (e.g., 'L1').
        forecaster_type_argv (str): Type of forecasting model ('LSTM', 'Transformer', 'TCN').
        seed (int): Random seed for reproducibility (default: 42).

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    def save_report(report_path) -> None:
        with open(f"{report_path}/report.json", 'w') as file:
            json.dump(report, file, indent=4)

    execution_id = f"{timestamp}_{dataset_domain_argv}_{dataset_name_argv}_{cpd_method_argv}_{cpd_cost_function_argv}_{forecaster_type_argv}_{seed}"
    cpd_method = CPDMethod.from_str(cpd_method_argv)
    cpd_cost_function = CPDCostFunction.from_str(cpd_cost_function_argv)
    forecaster_type = ForecasterType.from_str(forecaster_type_argv)
    change_point_approach = f"{cpd_method.value.title()} {cpd_cost_function.value.title()}"
    outputs_sub_path = f"seed={seed}/dataset_domain={dataset_domain_argv}/dataset_name={dataset_name_argv}/cpd_method={cpd_method.value}/cpd_cost_function={cpd_cost_function.value}/forecaster_type={forecaster_type.value}/timestamp={timestamp}"

    print(f"[Step 1] Reading dataset {dataset_name_argv} from {dataset_domain_argv}")
    df, variables = read_dataset(dataset_domain_argv, dataset_name_argv)
    print(f"Variables: {variables}")
    report_path = f"outputs/report/{outputs_sub_path}"
    os.makedirs(report_path, exist_ok=True)

    print("[Step 2] Filling missing values and splitting data into train_val and test")
    df, missing_mask = fill_na(df, variables)
    train_val, test = split_train_test(df)
    missing_mask_train_val, missing_mask_test = split_train_test(missing_mask)

    # Calculate missing value statistics
    total_values_train_val = missing_mask_train_val.size
    total_values_test = missing_mask_test.size
    missing_values_train_val = missing_mask_train_val.sum().sum()
    missing_values_test = missing_mask_test.sum().sum()
    missing_pct_train_val = (missing_values_train_val / total_values_train_val * 100) if total_values_train_val > 0 else 0
    missing_pct_test = (missing_values_test / total_values_test * 100) if total_values_test > 0 else 0

    report = {
        'execution_id': execution_id,
        'timestamp': timestamp,
        'forecaster_type': forecaster_type.value,
        'cpd_method': cpd_method.value,
        'cpd_cost_function': cpd_cost_function.value,
        'change_point_approach': change_point_approach,
        'seed': seed,
        'observation_window': OBSERVATION_WINDOW,
        'train_perc': TRAIN_PERC,
        'nb_trials': NB_TRIALS,
        'dataset_domain': dataset_domain_argv,
        'dataset': dataset_name_argv,
        'variables': variables,
        'dataset_shape': df.shape,
        'train_val_shape': train_val.shape,
        'test_shape': test.shape,
        'missing_values_train_val': int(missing_values_train_val),
        'missing_values_test': int(missing_values_test),
        'missing_pct_train_val': float(missing_pct_train_val),
        'missing_pct_test': float(missing_pct_test),
    }
    save_report(report_path)

    print(f"[Step 3] Detecting change point ({change_point_approach})")
    start_time = time.time()
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    change_point_detector = CPDDetectorFactory.create_detector(cpd_method, cpd_cost_function)
    change_point, change_point_perc = change_point_detector.find_change_point(train_val, variables)
    end_time = time.time()
    end_time_perf = time.perf_counter()
    end_time_process = time.process_time()
    print(f"Change point: {change_point}, Change point percentage: {change_point_perc}")
    report.update({
        'detect_change_point_time_duration': end_time - start_time,
        'detect_change_point_perf_duration': end_time_perf - start_time_perf,
        'detect_change_point_process_duration': end_time_process - start_time_process,
        'change_point': str(change_point),
        'change_point_perc': change_point_perc
    })
    save_report(report_path)

    print("[Step 4] Reducing train_val based on change point")
    start_time = time.time()
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    reduced_train_val = change_point_detector.apply_change_point(train_val, change_point)
    end_time = time.time()
    end_time_perf = time.perf_counter()
    end_time_process = time.process_time()
    report.update({
        'apply_change_point_time_duration': end_time - start_time,
        'apply_change_point_perf_duration': end_time_perf - start_time_perf,
        'apply_change_point_process_duration': end_time_process - start_time_process,
        'reduced_train_val.shape': reduced_train_val.shape,
    })
    save_report(report_path)

    print("[Step 5] Splitting train_val into train and val")
    reduced_train, reduced_val = split_train_test(reduced_train_val)
    report.update({
        'reduced_train.shape': reduced_train.shape,
        'reduced_val.shape': reduced_val.shape,
    })
    save_report(report_path)

    print("[Step 6] Fitting scaler on train and applying on train and val")
    start_time = time.time()
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    scaler = ScalerFactory.create_scaler(ScalerType.STANDARD, variables)
    scaled_reduced_train = scaler.fit_scale(reduced_train)
    scaled_reduced_val = scaler.scale(reduced_val)
    end_time = time.time()
    end_time_perf = time.perf_counter()
    end_time_process = time.process_time()
    report.update({
        'fit_apply_scaler_train_val_time_duration': end_time - start_time,
        'fit_apply_scaler_train_val_perf_duration': end_time_perf - start_time_perf,
        'fit_apply_scaler_train_val_process_duration': end_time_process - start_time_process,
    })
    save_report(report_path)

    print("[Step 7] Splitting train and val into X and y")
    X_reduced_scaled_train, y_reduced_scaled_train = split_X_y(scaled_reduced_train)
    X_reduced_scaled_val, y_reduced_scaled_val = split_X_y(scaled_reduced_val)
    report.update({
        'X_reduced_scaled_train.shape': X_reduced_scaled_train.shape,
        'y_reduced_scaled_train.shape': y_reduced_scaled_train.shape,
        'X_reduced_scaled_val.shape': X_reduced_scaled_val.shape,
        'y_reduced_scaled_val.shape': y_reduced_scaled_val.shape,
    })
    save_report(report_path)

    print(f"[Step 8] Running HPO and NAS with {forecaster_type.value.upper()} model")
    n_variables = len(variables)
    forecaster_hypermodel = ForecasterFactory.create_forecaster(
        forecaster_type=forecaster_type,
        n_variables=n_variables
    )

    # Note: MirroredStrategy only needed for multi-GPU setups
    gpu_devices = tf.config.list_physical_devices("GPU")
    if len(gpu_devices) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = None

    tuner_temp_dir = tempfile.mkdtemp(prefix="keras_tuner_")

    forecaster_tuner = BayesianOptimization(
        forecaster_hypermodel,
        objective='val_loss',
        max_trials=NB_TRIALS,
        executions_per_trial=1,
        directory=tuner_temp_dir,
        project_name=outputs_sub_path,
        seed=seed,
        overwrite=True,
        distribution_strategy=strategy,
        max_consecutive_failed_trials=int(NB_TRIALS/2)
    )
    start_time = time.time()
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    forecaster_tuner.search(
        X_reduced_scaled_train,
        y_reduced_scaled_train,
        validation_data=(X_reduced_scaled_val, y_reduced_scaled_val),
        shuffle=False,
    )
    end_time = time.time()
    end_time_perf = time.perf_counter()
    end_time_process = time.process_time()
    report.update({
        'tuner_time_duration': end_time - start_time,
        'tuner_perf_duration': end_time_perf - start_time_perf,
        'tuner_process_duration': end_time_process - start_time_process,
    })
    save_report(report_path)

    print("[Step 9] Retrieving best hyperparameters and rebuilding model")
    best_trial = forecaster_tuner.oracle.get_best_trials(num_trials=1)[0]
    print(f"Trial ID: {best_trial.trial_id}")
    print(f"Hyperparameters: {best_trial.hyperparameters.values}")
    print(f"Score: {best_trial.score}")
    print("-" * 40)

    best_forecaster_model = forecaster_hypermodel.build(best_trial.hyperparameters)
    best_forecaster_model.summary()
    best_forecaster_model = InternalForecaster(
        best_forecaster_model,
        len(variables),
    )
    report.update({
        'best_trial_id': best_trial.trial_id,
        'best_trial_hyperparameters': best_trial.hyperparameters.values,
        'best_trial_score': best_trial.score,
        'best_forecaster_model': best_forecaster_model.summary(),
    })

    # Clean up temp tuner directory
    try:
        shutil.rmtree(tuner_temp_dir)
        print(f"Cleaned up temporary tuner directory: {tuner_temp_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up tuner directory {tuner_temp_dir}: {e}")
    save_report(report_path)

    print("[Step 10] Fitting scaler on train_val and applying on train_val and test")
    start_time = time.time()
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    scaler = ScalerFactory.create_scaler(ScalerType.STANDARD, variables)
    scaled_reduced_train_val = scaler.fit_scale(reduced_train_val)
    scaled_test = scaler.scale(test)
    end_time = time.time()
    end_time_perf = time.perf_counter()
    end_time_process = time.process_time()
    report.update({
        'fit_apply_scaler_train_val_test_time_duration': end_time - start_time,
        'fit_apply_scaler_train_val_test_perf_duration': end_time_perf - start_time_perf,
        'fit_apply_scaler_train_val_test_process_duration': end_time_process - start_time_process,
    })
    save_report(report_path)

    print("[Step 11] Splitting train_val and test into X and y")
    X_reduced_scaled_train_val, y_reduced_scaled_train_val = split_X_y(scaled_reduced_train_val)
    X_scaled_test, y_scaled_test = split_X_y(scaled_test)
    # Create missing mask for y_test to track which values were originally missing
    y_test_missing_mask = create_missing_mask_for_y(missing_mask_test)
    report.update({
        'X_reduced_scaled_train_val.shape': X_reduced_scaled_train_val.shape,
        'y_reduced_scaled_train_val.shape': y_reduced_scaled_train_val.shape,
        'X_scaled_test.shape': X_scaled_test.shape,
        'y_scaled_test.shape': y_scaled_test.shape,
    })
    save_report(report_path)

    print("[Step 12] Retraining best model")
    start_time = time.time()
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    best_forecaster_model.fit(
        X_reduced_scaled_train_val,
        y_reduced_scaled_train_val,
        shuffle=False
    )
    end_time = time.time()
    end_time_perf = time.perf_counter()
    end_time_process = time.process_time()
    report.update({
        'retrain_time_duration': end_time - start_time,
        'retrain_perf_duration': end_time_perf - start_time_perf,
        'retrain_process_duration': end_time_process - start_time_process,
    })
    save_report(report_path)

    print("[Step 13] Forecasting for test")
    start_time = time.time()
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    y_scaled_pred = best_forecaster_model.forecast(X_scaled_test)
    y_scaled_test_flat = y_scaled_test.reshape(-1, n_variables)
    y_scaled_pred_flat = y_scaled_pred.reshape(-1, n_variables)
    end_time = time.time()
    end_time_perf = time.perf_counter()
    end_time_process = time.process_time()
    report.update({
        'forecasting_test_time_duration': end_time - start_time,
        'forecasting_test_perf_duration': end_time_perf - start_time_perf,
        'forecasting_test_process_duration': end_time_process - start_time_process,
    })
    save_report(report_path)

    print("[Step 14] Descaling data")
    start_time = time.time()
    start_time_perf = time.perf_counter()
    start_time_process = time.process_time()
    y_test = scaler.descale(pd.DataFrame(y_scaled_test_flat, columns=variables))
    y_pred = scaler.descale(pd.DataFrame(y_scaled_pred_flat, columns=variables))
    end_time = time.time()
    end_time_perf = time.perf_counter()
    end_time_process = time.process_time()
    report.update({
        'descaling_time_duration': end_time - start_time,
        'descaling_perf_duration': end_time_perf - start_time_perf,
        'descaling_process_duration': end_time_process - start_time_process,
    })
    save_report(report_path)

    print("[Step 15] Calculating evaluation metrics")
    # Flatten the missing mask to match y_test and y_pred shapes
    y_test_missing_mask_flat = y_test_missing_mask.reshape(-1, n_variables)

    # Calculate metrics on ALL test data (including filled values)
    error_results_all = get_error_results(y_test, y_pred, variables)

    # Calculate metrics on REAL test data only (excluding filled values)
    # Keep only rows where ALL variables are real (not missing)
    # This ensures all variables have the same number of samples
    real_values_mask = ~y_test_missing_mask_flat
    rows_all_real = ~y_test_missing_mask_flat.any(axis=1)

    y_test_real = y_test[rows_all_real].reset_index(drop=True)
    y_pred_real = y_pred[rows_all_real].reset_index(drop=True)

    error_results_real = get_error_results(y_test_real, y_pred_real, variables)

    # Count rows and real vs filled values in test set
    total_test_rows = len(y_test_missing_mask_flat)
    rows_with_all_real = rows_all_real.sum()
    total_test_values = y_test_missing_mask_flat.size
    filled_test_values = y_test_missing_mask_flat.sum()
    real_test_values = total_test_values - filled_test_values

    print(f"Evaluation on ALL test data: {error_results_all}")
    print(f"Evaluation on REAL test data only ({rows_with_all_real}/{total_test_rows} complete rows, {real_test_values}/{total_test_values} real values): {error_results_real}")

    report.update({
        'total_time_duration': sum(value for key, value in report.items() if key.endswith('_time_duration')),
        'total_perf_duration': sum(value for key, value in report.items() if key.endswith('_perf_duration')),
        'total_process_duration': sum(value for key, value in report.items() if key.endswith('_process_duration')),
        'test_total_rows': int(total_test_rows),
        'test_rows_all_real': int(rows_with_all_real),
        'test_total_values': int(total_test_values),
        'test_real_values': int(real_test_values),
        'test_filled_values': int(filled_test_values),
        'test_filled_pct': float((filled_test_values / total_test_values * 100) if total_test_values > 0 else 0),
        'error_results_all': error_results_all,
        'error_results_real_only': error_results_real,
    })
    save_report(report_path)

    print("Finished execution")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("""
            Wrong number of parameters!
            Usage: python main.py <dataset_domain> <dataset> <cpd_method> <cpd_cost_function> <forecaster_type> [seed]

            Arguments:
              dataset_domain: Domain of the dataset (e.g., 'TCPD', 'UCI')
              dataset: Specific dataset name (e.g., 'APPLE', 'SAOPAULO_SP')
              cpd_method: Change point detection method (e.g., 'Window', 'Bin_Seg')
              cpd_cost_function: Cost function for CPD (e.g., 'L1', 'L2')
              forecaster_type: Forecasting model type (e.g., 'LSTM', 'TCN')
              seed: Random seed for reproducibility (optional, default: 42)
        """)
        sys.exit(1)

    dataset_domain_argv = sys.argv[1]
    dataset_name_argv = sys.argv[2]
    cpd_method_argv = sys.argv[3]
    cpd_cost_function_argv = sys.argv[4]
    forecaster_type_argv = sys.argv[5]

    # Optional seed parameter (default: 42)
    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run(timestamp, dataset_domain_argv, dataset_name_argv, cpd_method_argv,
        cpd_cost_function_argv, forecaster_type_argv, seed)
