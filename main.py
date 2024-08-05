import json
import os
import random
import sys
import time
from datetime import datetime
from typing import List, Union

from keras_tuner import RandomSearch

import numpy as np

import pandas as pd

from sklearn.metrics import mean_absolute_percentage_error

import tensorflow as tf

from config.constants import (
    CUT_POINT_METHODS, FIXED_CUTS_PERCS, FORECASTER_OBJECTIVE,
    MODEL_TYPE, NB_TRIALS, OBSERVATION_WINDOW, SEED, TRAIN_PERC
)

from src.cut_point_detector import get_cut_point_detector
from src.dataset import read_dataset, split_X_y, split_train_test
from src.forecaster import InternalForecaster, TimeSeriesHyperModel
from src.scaler import Scaler

tf.get_logger().setLevel('ERROR')

np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)


def run(execution_id: str, dataset_domain_argv: str, dataset_argv: str) -> None:
    print(f"Reading dataset {dataset_argv} from {dataset_domain_argv}")
    df, variables = read_dataset(dataset_domain_argv, dataset_argv)
    print(f"Variables: {variables}")

    print("Splitting data into train and test")
    train, test = split_train_test(df)

    print("Training and applying scaler")
    scaler = Scaler(variables)
    scaled_train = scaler.fit_scale(train)
    scaled_test = scaler.scale(test)

    print("Creating hypermodel and tuner")
    forecaster_hypermodel = TimeSeriesHyperModel(
        model_type=MODEL_TYPE,
        n_variables=len(variables)
    )

    report = {
        'execution_id': execution_id,
        'seed': SEED,
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

    def run_inner(cut_point_model: str, cut_point_methods: Union[List[str], List[float]]) -> None:
        for cut_point_method in [cut_point_methods[0]]:
            forecaster_tuner = RandomSearch(
                forecaster_hypermodel,
                objective=FORECASTER_OBJECTIVE,
                max_trials=NB_TRIALS,
                directory=f"outputs/tuner/{execution_id}",
                project_name=f"{cut_point_model}/{cut_point_method}",
                seed=SEED,
                overwrite=True
            )
            approach = f"{cut_point_model.title()} {cut_point_method}"
            print(f"Started cut point for {approach}")
            start_time = time.time()
            cut_point_detector = get_cut_point_detector(cut_point_model, cut_point_method)
            cut_point = cut_point_detector.find_cut_point(train, variables)
            end_time = time.time()
            cut_duration = end_time - start_time
            print(f"Cut point: {cut_point}, Duration: {cut_duration}")
            print("Finished " + approach + "\n")

            print("Applying subset to train based on cut point")
            reduced_scaled_train = cut_point_detector.apply_cut_point(scaled_train, cut_point)

            print("Splitting into X and y")
            X_reduced_scaled_train, y_reduced_scaled_train = split_X_y(reduced_scaled_train)
            X_scaled_test, y_scaled_test = split_X_y(scaled_test)

            print("Started running HPO and NAS")
            start_time = time.time()
            forecaster_tuner.search(
                X_reduced_scaled_train,
                y_reduced_scaled_train,
                validation_split=(1 - TRAIN_PERC),
                shuffle=False
            )
            end_time = time.time()
            tuner_duration = end_time - start_time
            print(f"Finished running HPO and NAS, duration: {tuner_duration}")

            best_trial = forecaster_tuner.oracle.get_best_trials(num_trials=1)[0]
            print(f"Trial ID: {best_trial.trial_id}")
            print(f"Hyperparameters: {best_trial.hyperparameters.values}")
            print(f"Score: {best_trial.score}")
            print("-" * 40)

            print("Retrieving best model")
            best_forecaster_model = forecaster_tuner.get_best_models(num_models=1)[0]
            best_forecaster_model.summary()
            best_forecaster_model = InternalForecaster(best_forecaster_model)

            print("Running forecasting")
            y_scaled_pred = best_forecaster_model.forecast(X_scaled_test)

            print("Calculating error")
            y_test = scaler.descale(pd.DataFrame(y_scaled_test, columns=variables))
            y_pred = scaler.descale(pd.DataFrame(y_scaled_pred, columns=variables))
            mape = mean_absolute_percentage_error(y_test, y_pred)
            print(f"Obtained MAPE: {mape}")

            print("Writing report")
            report[f"model_{cut_point_model}"] = {
                f"method_{cut_point_method}": {
                    'MAPE': mape,
                    'cut_duration': cut_duration,
                    'tuner_duration': tuner_duration,
                    'reduced_scaled_train_shape': reduced_scaled_train.shape,
                    'best_trial_id': best_trial.trial_id,
                    'best_trial_hyperparameters': best_trial.hyperparameters.values,
                    'best_trial_score': best_trial.score,
                    'best_forecaster_model': best_forecaster_model.summary(),
                }
            }

    run_inner("Fixed_Perc", FIXED_CUTS_PERCS)
    run_inner("Window", CUT_POINT_METHODS)
    run_inner("Bin_Seg", CUT_POINT_METHODS)
    run_inner("Bottom_Up", CUT_POINT_METHODS)

    report_path = f"outputs/report/{execution_id}"
    os.makedirs(report_path, exist_ok=True)
    with open(f"outputs/report/{execution_id}/report.json", 'w') as file:
        json.dump(report, file, indent=4)


if __name__ == "__main__":
    dataset_domain_argv = sys.argv[1]
    dataset_argv = sys.argv[2]

    execution_id = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f"Execution id: {execution_id}")

    run(execution_id, dataset_domain_argv, dataset_argv)
