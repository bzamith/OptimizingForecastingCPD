import json
import os
import random
import shutil
import sys
import tempfile
import time
import warnings
from datetime import datetime
from typing import List, Tuple

from keras_tuner import RandomSearch

import numpy as np

import pandas as pd

import tensorflow as tf

from config.constants import (
    FORECAST_HORIZON, NB_TRIALS, OBSERVATION_WINDOW, ROLLING_WINDOW_N_SPLITS, ROLLING_WINDOW_TEST_SIZE
)

from src.cpd import CPDCostFunction, CPDMethod, CPDDetectorFactory
from src.data_reader import create_missing_mask_for_y, fill_na, read_dataset, split_X_y
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
    for gpu in gpu_devices:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.keras.mixed_precision.set_global_policy('mixed_float16')
    print(f"Running on GPU: {len(gpu_devices)} device(s) detected")
else:
    print("Running on CPU: Using all available cores")


def create_rolling_window_splits(
    df: pd.DataFrame,
    n_splits: int = ROLLING_WINDOW_N_SPLITS,
    test_size: float = ROLLING_WINDOW_TEST_SIZE
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """Create rolling window splits for time series cross-validation.

    Each split has:
    - train: expanding window from start
    - val: validation set (portion of training data for HPO)
    - test: fixed-size test window that moves forward

    The test windows don't overlap and cover the latter part of the data.

    Args:
        df: The full dataset as a DataFrame
        n_splits: Number of rolling window splits
        test_size: Proportion of data for each test fold

    Returns:
        List of tuples (train_val, val, test) DataFrames for each fold
    """
    n_samples = len(df)
    test_samples = int(n_samples * test_size)

    # Calculate where test windows start
    # The last test window ends at n_samples
    # We work backwards to find start positions
    total_test_samples = test_samples * n_splits
    first_test_start = n_samples - total_test_samples

    splits = []
    for i in range(n_splits):
        test_start = first_test_start + i * test_samples
        test_end = test_start + test_samples

        # Training data is everything before the test window
        train_val = df.iloc[:test_start].reset_index(drop=True)
        test = df.iloc[test_start:test_end].reset_index(drop=True)

        splits.append((train_val, test))

    return splits


def run_rolling(timestamp: str,
                dataset_domain_argv: str,
                dataset_name_argv: str,
                cpd_method_argv: str,
                cpd_cost_function_argv: str,
                forecaster_type_argv: str,
                seed: int = 42,
                n_splits: int = ROLLING_WINDOW_N_SPLITS) -> None:
    """Execute rolling window cross-validation for forecasting with CPD.

    Args:
        timestamp: Timestamp of the execution.
        dataset_domain_argv: Domain of the dataset.
        dataset_name_argv: Specific dataset to be used.
        cpd_method_argv: Identifier for the change point method.
        cpd_cost_function_argv: Identifier for the change point cost function.
        forecaster_type_argv: Type of forecasting model.
        seed: Random seed for reproducibility.
        n_splits: Number of rolling window folds.

    Returns:
        None
    """
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

    cpd_method = CPDMethod.from_str(cpd_method_argv)
    cpd_cost_function = CPDCostFunction.from_str(cpd_cost_function_argv)
    forecaster_type = ForecasterType.from_str(forecaster_type_argv)
    change_point_approach = f"{cpd_method.value.title()} {cpd_cost_function.value.title()}"

    outputs_sub_path = f"seed={seed}/dataset_domain={dataset_domain_argv}/dataset_name={dataset_name_argv}/cpd_method={cpd_method.value}/cpd_cost_function={cpd_cost_function.value}/forecaster_type={forecaster_type.value}/timestamp={timestamp}"
    report_path = f"outputs/rolling_report/{outputs_sub_path}"
    os.makedirs(report_path, exist_ok=True)

    def save_report(report_dict, filename="report.json"):
        with open(f"{report_path}/{filename}", 'w') as file:
            json.dump(report_dict, file, indent=4)

    execution_id = f"{timestamp}_{dataset_domain_argv}_{dataset_name_argv}_{cpd_method_argv}_{cpd_cost_function_argv}_{forecaster_type_argv}_{seed}"

    print(f"[Step 1] Reading dataset {dataset_name_argv} from {dataset_domain_argv}")
    df, variables = read_dataset(dataset_domain_argv, dataset_name_argv)
    print(f"Variables: {variables}")

    print("[Step 2] Filling missing values")
    df, missing_mask = fill_na(df, variables)

    print(f"[Step 3] Creating {n_splits} rolling window splits")
    splits = create_rolling_window_splits(df, n_splits=n_splits)

    # Store fold results
    fold_results = []
    all_error_results = []
    all_error_results_real = []

    master_report = {
        'execution_id': execution_id,
        'timestamp': timestamp,
        'forecaster_type': forecaster_type.value,
        'cpd_method': cpd_method.value,
        'cpd_cost_function': cpd_cost_function.value,
        'change_point_approach': change_point_approach,
        'seed': seed,
        'observation_window': OBSERVATION_WINDOW,
        'n_splits': n_splits,
        'test_size': ROLLING_WINDOW_TEST_SIZE,
        'nb_trials': NB_TRIALS,
        'dataset_domain': dataset_domain_argv,
        'dataset': dataset_name_argv,
        'variables': variables,
        'dataset_shape': df.shape,
    }
    save_report(master_report)

    n_variables = len(variables)
    change_point_detector = CPDDetectorFactory.create_detector(cpd_method, cpd_cost_function)

    for fold_idx, (train_val, test) in enumerate(splits):
        fold_num = fold_idx + 1
        print(f"\n{'='*60}")
        print(f"FOLD {fold_num}/{n_splits}")
        print(f"{'='*60}")
        print(f"Train+Val size: {len(train_val)}, Test size: {len(test)}")

        fold_report = {
            'fold': fold_num,
            'train_val_shape': train_val.shape,
            'test_shape': test.shape,
        }

        # Get corresponding missing masks
        fold_start_idx = len(df) - len(test) - len(train_val)
        train_val_end_idx = fold_start_idx + len(train_val)
        test_end_idx = train_val_end_idx + len(test)

        missing_mask_train_val = missing_mask.iloc[fold_start_idx:train_val_end_idx].reset_index(drop=True)
        missing_mask_test = missing_mask.iloc[train_val_end_idx:test_end_idx].reset_index(drop=True)

        # Step: Detect change point on train_val
        print(f"[Fold {fold_num}] Detecting change point ({change_point_approach})")
        start_time = time.time()
        change_point, change_point_perc = change_point_detector.find_change_point(train_val, variables)
        cpd_duration = time.time() - start_time
        print(f"Change point: {change_point}, percentage: {change_point_perc:.2%}")

        fold_report.update({
            'change_point': str(change_point),
            'change_point_perc': change_point_perc,
            'cpd_duration': cpd_duration,
        })

        # Step: Apply change point
        print(f"[Fold {fold_num}] Reducing train_val based on change point")
        reduced_train_val = change_point_detector.apply_change_point(train_val, change_point)
        
        # MINIMAL FIX: Ensure reduced dataset is large enough for HPO
        # Both train (80%) and val (20%) need at least OBSERVATION_WINDOW + FORECAST_HORIZON samples
        # to create at least 1 window. Val is smaller, so: min_total = min_val_samples / 0.2
        min_samples_per_split = OBSERVATION_WINDOW + FORECAST_HORIZON  # 21 samples for 1 window
        min_size = int(min_samples_per_split / 0.2) + 1  # ~106 samples to ensure val has enough
        if len(reduced_train_val) < min_size:
            print(f"[Fold {fold_num}] WARNING: Reduced dataset too small ({len(reduced_train_val)}). Falling back to last {min_size} samples.")
            reduced_train_val = train_val.iloc[-min_size:].reset_index(drop=True)
            
        fold_report['reduced_train_val_shape'] = reduced_train_val.shape

        # Step: Split reduced train_val into train and val (80/20)
        print(f"[Fold {fold_num}] Splitting train_val into train and val")
        train_size = int(len(reduced_train_val) * 0.8)
        reduced_train = reduced_train_val.iloc[:train_size].reset_index(drop=True)
        reduced_val = reduced_train_val.iloc[train_size:].reset_index(drop=True)
        fold_report.update({
            'reduced_train_shape': reduced_train.shape,
            'reduced_val_shape': reduced_val.shape,
        })

        # Step: Scale data for HPO
        print(f"[Fold {fold_num}] Scaling data for hyperparameter tuning")
        scaler = ScalerFactory.create_scaler(ScalerType.STANDARD, variables)
        scaled_reduced_train = scaler.fit_scale(reduced_train)
        scaled_reduced_val = scaler.scale(reduced_val)

        # Step: Split into X and y
        X_train, y_train = split_X_y(scaled_reduced_train)
        X_val, y_val = split_X_y(scaled_reduced_val)

        # Step: HPO with RandomSearch
        print(f"[Fold {fold_num}] Running HPO and NAS with {forecaster_type.value.upper()} model")
        forecaster_hypermodel = ForecasterFactory.create_forecaster(
            forecaster_type=forecaster_type,
            n_variables=n_variables
        )

        gpu_devices_check = tf.config.list_physical_devices("GPU")
        strategy = tf.distribute.MirroredStrategy() if len(gpu_devices_check) > 1 else None

        tuner_temp_dir = tempfile.mkdtemp(prefix=f"keras_tuner_fold{fold_num}_")

        forecaster_tuner = RandomSearch(
            forecaster_hypermodel,
            objective='val_loss',
            max_trials=NB_TRIALS,
            executions_per_trial=1,
            directory=tuner_temp_dir,
            project_name=f"fold_{fold_num}",
            seed=seed,
            overwrite=True,
            distribution_strategy=strategy,
            max_consecutive_failed_trials=int(NB_TRIALS/2)
        )

        start_time = time.time()
        forecaster_tuner.search(
            X_train, y_train,
            validation_data=(X_val, y_val),
            shuffle=False,
        )
        tuner_duration = time.time() - start_time
        fold_report['tuner_duration'] = tuner_duration

        # Get best model
        print(f"[Fold {fold_num}] Retrieving best hyperparameters")
        best_trial = forecaster_tuner.oracle.get_best_trials(num_trials=1)[0]
        print(f"Best trial score: {best_trial.score}")

        best_forecaster_model = forecaster_hypermodel.build(best_trial.hyperparameters)
        best_forecaster_model = InternalForecaster(best_forecaster_model, n_variables)

        fold_report.update({
            'best_trial_id': best_trial.trial_id,
            'best_trial_hyperparameters': best_trial.hyperparameters.values,
            'best_trial_score': best_trial.score,
        })

        # Clean up tuner directory
        try:
            shutil.rmtree(tuner_temp_dir)
        except Exception as e:
            print(f"Warning: Could not clean up tuner directory: {e}")

        # Step: Retrain on full reduced train_val
        print(f"[Fold {fold_num}] Retraining on full train_val")
        scaler = ScalerFactory.create_scaler(ScalerType.STANDARD, variables)
        scaled_reduced_train_val = scaler.fit_scale(reduced_train_val)
        scaled_test = scaler.scale(test)

        X_train_val, y_train_val = split_X_y(scaled_reduced_train_val)
        X_test, y_test_scaled = split_X_y(scaled_test)
        y_test_missing_mask = create_missing_mask_for_y(missing_mask_test)

        start_time = time.time()
        best_forecaster_model.fit(X_train_val, y_train_val, shuffle=False)
        retrain_duration = time.time() - start_time
        fold_report['retrain_duration'] = retrain_duration

        # Step: Forecast on test
        print(f"[Fold {fold_num}] Forecasting on test set")
        start_time = time.time()
        y_pred_scaled = best_forecaster_model.forecast(X_test)
        forecast_duration = time.time() - start_time
        fold_report['forecast_duration'] = forecast_duration

        # Reshape and descale
        y_test_scaled_flat = y_test_scaled.reshape(-1, n_variables)
        y_pred_scaled_flat = y_pred_scaled.reshape(-1, n_variables)

        y_test_df = scaler.descale(pd.DataFrame(y_test_scaled_flat, columns=variables))
        y_pred_df = scaler.descale(pd.DataFrame(y_pred_scaled_flat, columns=variables))

        # Calculate metrics
        print(f"[Fold {fold_num}] Calculating evaluation metrics")
        y_test_missing_mask_flat = y_test_missing_mask.reshape(-1, n_variables)

        # All data metrics
        error_results_all = get_error_results(y_test_df, y_pred_df, variables)

        # Real data only metrics
        rows_all_real = ~y_test_missing_mask_flat.any(axis=1)
        y_test_real = y_test_df[rows_all_real].reset_index(drop=True)
        y_pred_real = y_pred_df[rows_all_real].reset_index(drop=True)
        error_results_real = get_error_results(y_test_real, y_pred_real, variables)

        fold_report.update({
            'error_results_all': error_results_all,
            'error_results_real_only': error_results_real,
            'test_total_rows': int(len(y_test_missing_mask_flat)),
            'test_rows_all_real': int(rows_all_real.sum()),
        })

        print(f"[Fold {fold_num}] Results (all): {error_results_all}")
        print(f"[Fold {fold_num}] Results (real only): {error_results_real}")

        fold_results.append(fold_report)
        all_error_results.append(error_results_all)
        all_error_results_real.append(error_results_real)

        # Save fold-specific report
        save_report(fold_report, f"fold_{fold_num}_report.json")

    # Aggregate results across folds
    print(f"\n{'='*60}")
    print("AGGREGATING RESULTS ACROSS ALL FOLDS")
    print(f"{'='*60}")

    def aggregate_metrics(results_list):
        """Aggregate metrics across folds (mean and std)."""
        if not results_list:
            return {}

        aggregated = {}
        # Get all metrics from first result
        for key in results_list[0].keys():
            values = [r[key] for r in results_list if key in r]
            if values and isinstance(values[0], (int, float)):
                aggregated[f"{key}_mean"] = float(np.mean(values))
                aggregated[f"{key}_std"] = float(np.std(values))
                aggregated[f"{key}_values"] = values
        return aggregated

    aggregated_all = aggregate_metrics(all_error_results)
    aggregated_real = aggregate_metrics(all_error_results_real)

    master_report.update({
        'fold_results': fold_results,
        'aggregated_metrics_all': aggregated_all,
        'aggregated_metrics_real_only': aggregated_real,
    })
    save_report(master_report)

    print("\nAggregated Metrics (All Data):")
    for key, value in aggregated_all.items():
        if key.endswith('_mean'):
            metric_name = key.replace('_mean', '')
            std_key = f"{metric_name}_std"
            print(f"  {metric_name}: {value:.4f} (+/- {aggregated_all.get(std_key, 0):.4f})")

    print("\nAggregated Metrics (Real Data Only):")
    for key, value in aggregated_real.items():
        if key.endswith('_mean'):
            metric_name = key.replace('_mean', '')
            std_key = f"{metric_name}_std"
            print(f"  {metric_name}: {value:.4f} (+/- {aggregated_real.get(std_key, 0):.4f})")

    print("\nFinished rolling window validation!")


if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("""
            Wrong number of parameters!
            Usage: python main_rolling.py <dataset_domain> <dataset> <cpd_method> <cpd_cost_function> <forecaster_type> [seed] [n_splits]

            Arguments:
              dataset_domain: Domain of the dataset (e.g., 'TCPD', 'UCI')
              dataset: Specific dataset name (e.g., 'APPLE', 'SAOPAULO_SP')
              cpd_method: Change point detection method (e.g., 'Window', 'Bin_Seg')
              cpd_cost_function: Cost function for CPD (e.g., 'L1', 'L2')
              forecaster_type: Forecasting model type (e.g., 'LSTM', 'TCN')
              seed: Random seed for reproducibility (optional, default: 42)
              n_splits: Number of rolling window folds (optional, default: 5)
        """)
        sys.exit(1)

    dataset_domain_argv = sys.argv[1]
    dataset_name_argv = sys.argv[2]
    cpd_method_argv = sys.argv[3]
    cpd_cost_function_argv = sys.argv[4]
    forecaster_type_argv = sys.argv[5]

    seed = int(sys.argv[6]) if len(sys.argv) > 6 else 42
    n_splits = int(sys.argv[7]) if len(sys.argv) > 7 else ROLLING_WINDOW_N_SPLITS

    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    run_rolling(timestamp, dataset_domain_argv, dataset_name_argv, cpd_method_argv,
                cpd_cost_function_argv, forecaster_type_argv, seed, n_splits)
