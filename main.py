import sys
from datetime import datetime

import pandas as pd

import numpy as np

import tensorflow as tf

from statistics import mean, stdev
from multiprocessing import Queue, Process

from src.dataset import read_dataset
from src.execute import CUT_COLUMN, CUT_SECONDS_COLUMN, execute_binary_seg_cut, execute_bottom_up_cut, execute_fixed_cut, execute_full, execute_mean_cut, execute_median_cut, execute_window_cut

METHODS = ["L1", "L2", "Normal", "RBF", "Cosine", "Linear", "Clinear", "Rank", "Mahalanobis", "AR"]
NB_EXECUTIONS = 1

tf.get_logger().setLevel('ERROR')


def execute_once(X_train, X_test, variables):
    approaches_list = []
    errors_df_list = []
    executions_df_list = []

    # Full
    approach = "Full"
    print("Started " + approach)
    process_queue = Queue()
    process = Process(target=execute_full, args=(X_train, X_test, variables, hpo, process_queue))
    process.start()
    process.join()
    full_execution_df, full_errors_df = process_queue.get()
    executions_df_list.append(full_execution_df)
    errors_df_list.append(full_errors_df)
    approaches_list.append(approach)
    print("Finished " + approach + "\n")

    # Fixed cuts
    fixed_cut = 0.05
    while fixed_cut <= 0.95:
        approach = "Fixed Cut " + format(fixed_cut * 100, '.1f') + "%"
        print("Started " + approach)
        process_queue = Queue()
        process = Process(target=execute_fixed_cut, args=(fixed_cut, X_train, X_test, variables, hpo, process_queue))
        process.start()
        process.join()
        fixed_cut_execution_df, fixed_cut_errors_df = process_queue.get()
        executions_df_list.append(fixed_cut_execution_df)
        errors_df_list.append(fixed_cut_errors_df)
        approaches_list.append(approach)
        fixed_cut += 0.05
        print("Finished " + approach + "\n")

    # Window cuts
    window_cuts = []
    window_cut_seconds = []
    for method in METHODS:
        approach = "Window " + method
        print("Started " + approach)
        process_queue = Queue()
        process = Process(target=execute_window_cut, args=(str.lower(method), X_train, X_test, variables, hpo, process_queue))
        process.start()
        process.join()
        window_execution_df, window_errors_df = process_queue.get()
        window_cuts.append(window_execution_df[CUT_COLUMN].iloc[0])
        window_cut_seconds.append(window_execution_df[CUT_SECONDS_COLUMN].iloc[0])
        executions_df_list.append(window_execution_df)
        errors_df_list.append(window_errors_df)
        approaches_list.append(approach)
        print("Finished " + approach + "\n")
    approach = "Window Mean"
    print("Started " + approach)
    process_queue = Queue()
    process = Process(target=execute_mean_cut, args=(window_cuts, window_cut_seconds, X_train, X_test, variables, hpo, process_queue))
    process.start()
    process.join()
    mean_window_execution_df, mean_window_errors_df = process_queue.get()
    executions_df_list.append(mean_window_execution_df)
    errors_df_list.append(mean_window_errors_df)
    approaches_list.append(approach)
    print("Finished " + approach + "\n")
    approach = "Window Median"
    print("Started " + approach)
    process_queue = Queue()
    process = Process(target=execute_median_cut, args=(window_cuts, window_cut_seconds, X_train, X_test, variables, hpo, process_queue))
    process.start()
    process.join()
    median_window_execution_df, median_window_errors_df = process_queue.get()
    executions_df_list.append(median_window_execution_df)
    errors_df_list.append(median_window_errors_df)
    approaches_list.append(approach)
    print("Finished " + approach + "\n")

    # Binary Segmentation cuts
    binary_seg_cuts = []
    binary_seg_cut_seconds = []
    for method in METHODS:
        approach = "Binary Segmentation " + method
        print("Started " + approach)
        process_queue = Queue()
        process = Process(target=execute_binary_seg_cut, args=(str.lower(method), X_train, X_test, variables, hpo, process_queue))
        process.start()
        process.join()
        binary_seg_execution_df, binary_seg_errors_df = process_queue.get()
        binary_seg_cuts.append(binary_seg_execution_df[CUT_COLUMN].iloc[0])
        binary_seg_cut_seconds.append(binary_seg_execution_df[CUT_SECONDS_COLUMN].iloc[0])
        executions_df_list.append(binary_seg_execution_df)
        errors_df_list.append(binary_seg_errors_df)
        approaches_list.append(approach)
        print("Finished " + approach + "\n")
    approach = "Binary Segmentation Mean"
    print("Started " + approach)
    process_queue = Queue()
    process = Process(target=execute_mean_cut, args=(binary_seg_cuts, binary_seg_cut_seconds, X_train, X_test, variables, hpo, process_queue))
    process.start()
    process.join()
    mean_binary_seg_execution_df, mean_binary_seg_errors_df = process_queue.get()
    executions_df_list.append(mean_binary_seg_execution_df)
    errors_df_list.append(mean_binary_seg_errors_df)
    approaches_list.append(approach)
    print("Finished " + approach + "\n")
    approach = "Binary Segmentation Median"
    print("Started " + approach)
    process_queue = Queue()
    process = Process(target=execute_median_cut, args=(binary_seg_cuts, binary_seg_cut_seconds, X_train, X_test, variables, hpo, process_queue))
    process.start()
    process.join()
    median_binary_seg_execution_df, median_binary_seg_errors_df = process_queue.get()
    executions_df_list.append(median_binary_seg_execution_df)
    errors_df_list.append(median_binary_seg_errors_df)
    approaches_list.append(approach)
    print("Finished " + approach + "\n")

    # Bottom Up cuts
    bottom_up_cuts = []
    bottom_up_cut_seconds = []
    for method in METHODS:
        approach = "Bottom Up " + method
        print("Started " + approach)
        process_queue = Queue()
        process = Process(target=execute_bottom_up_cut, args=(str.lower(method), X_train, X_test, variables, hpo, process_queue))
        process.start()
        process.join()
        bottom_up_execution_df, bottom_up_errors_df = process_queue.get()
        bottom_up_cuts.append(bottom_up_execution_df[CUT_COLUMN].iloc[0])
        bottom_up_cut_seconds.append(bottom_up_execution_df[CUT_SECONDS_COLUMN].iloc[0])
        executions_df_list.append(bottom_up_execution_df)
        errors_df_list.append(bottom_up_errors_df)
        approaches_list.append(approach)
        print("Finished " + approach + "\n")
    approach = "Bottom Up Mean"
    print("Started " + approach)
    process_queue = Queue()
    process = Process(target=execute_mean_cut, args=(bottom_up_cuts, bottom_up_cut_seconds, X_train, X_test, variables, hpo, process_queue))
    process.start()
    process.join()
    mean_bottom_up_execution_df, mean_bottom_up_errors_df = process_queue.get()
    executions_df_list.append(mean_bottom_up_execution_df)
    errors_df_list.append(mean_bottom_up_errors_df)
    approaches_list.append(approach)
    print("Finished " + approach + "\n")
    approach = "Bottom Up Median"
    print("Started " + approach)
    process_queue = Queue()
    process = Process(target=execute_median_cut, args=(bottom_up_cuts, bottom_up_cut_seconds, X_train, X_test, variables, hpo, process_queue))
    process.start()
    process.join()
    median_bottom_up_execution_df, median_bottom_up_errors_df = process_queue.get()
    executions_df_list.append(median_bottom_up_execution_df)
    errors_df_list.append(median_bottom_up_errors_df)
    approaches_list.append(approach)
    print("Finished " + approach + "\n")

    # Save
    errors_df = pd.concat(errors_df_list)
    executions_df = pd.concat(executions_df_list)
    results_df = pd.concat([executions_df, errors_df], axis=1)
    results_df["Approach"] = approaches_list
    columns = results_df.columns.tolist()
    columns = columns[-1:] + columns[:-1]
    results_df = results_df.reindex(columns=columns)

    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    if hpo:
        results_df.to_csv("outputs/" + dataset_domain_argv + "/" + dataset_argv + "-" + now + "-hpo.csv", index=False)
    else:
        results_df.to_csv("outputs/" + dataset_domain_argv + "/" + dataset_argv + "-" + now + ".csv", index=False)
    return results_df


if __name__ == "__main__":
    dataset_domain_argv = sys.argv[1]
    dataset_argv = sys.argv[2]

    hpo = False
    try:
        hpo_argv = sys.argv[3]
        if hpo_argv == "HPO":
            hpo = True
    except IndexError:
        pass
    print("HPO = " + str(hpo))

    df, X_train, X_test, variables = read_dataset(dataset_domain_argv, dataset_argv)

    results_df_list = []
    for i in range(NB_EXECUTIONS):
        print("-----------------------------------\n")
        print("Started Execution " + str(i + 1))
        results_df_list.append(execute_once(X_train, X_test, variables))
        print("Finished Execution " + str(i + 1))

    final_results_df = pd.DataFrame()
    for column in results_df_list[0].columns[:5]:
        final_results_df[column] = results_df_list[0][column]
    for column in results_df_list[0].columns[5:]:
        avg_column_values = []
        std_column_values = []
        for i in range(results_df_list[0].shape[0]):
            curr_list = [df.iloc[i][column]for df in results_df_list]
            curr_list = [x for x in curr_list if x != "N/A"]
            if len(curr_list) > 0:
                avg_column_values.append(mean(curr_list))
                std_column_values.append(stdev(curr_list))
            else:
                avg_column_values.append(np.nan)
                std_column_values.append(np.nan)
        final_results_df["Avg " + column] = avg_column_values
        final_results_df["Std " + column] = std_column_values
    final_results_df = final_results_df.fillna("N/A")

    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    if hpo:
        final_results_df.to_csv("outputs/" + dataset_domain_argv + "/" + dataset_argv + "-" + now + "-hpo-combined.csv", index=False)
    else:
        final_results_df.to_csv("outputs/" + dataset_domain_argv + "/" + dataset_argv + "-" + now + "-combined.csv", index=False)
