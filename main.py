from datetime import datetime

import sys

import pandas as pd

from src.dataset import read_dataset
from src.execute import CUT_COLUMN, CUT_SECONDS_COLUMN, execute_binary_seg_cut, execute_bottom_up_cut, execute_fixed_cut, execute_full, execute_mean_cut, execute_median_cut, execute_window_cut

METHODS = ["L1"] #, "L2", "Normal", "RBF", "Cosine", "Linear", "Clinear", "Rank", "Mahalanobis", "AR"]

if __name__ == "__main__":
    approaches_list = []
    errors_df_list = []
    executions_df_list = []

    dataset_domain_argv = sys.argv[1]
    dataset_argv = sys.argv[2]

    df, X_train, X_test, variables = read_dataset(dataset_domain_argv, dataset_argv)

    # Full
    full_execution_df, full_errors_df = execute_full(X_train, X_test, variables)
    executions_df_list.append(full_execution_df)
    errors_df_list.append(full_errors_df)
    approaches_list.append("Full")
    print("Finished " + approaches_list[-1])

    # Fixed cuts
    fixed_cut = 0.05
    while fixed_cut <= 0.15: # Change 0.95
        fixed_cut_execution_df, fixed_cut_errors_df = execute_fixed_cut(fixed_cut, X_train, X_test, variables)
        executions_df_list.append(fixed_cut_execution_df)
        errors_df_list.append(fixed_cut_errors_df)
        approaches_list.append("Fixed Cut " + str(fixed_cut * 100) + "%")
        fixed_cut += 0.05
        print("Finished " + approaches_list[-1])

    # Window cuts
    window_cuts = []
    window_cut_seconds = []
    for method in METHODS:
        window_execution_df, window_errors_df = execute_window_cut(str.lower(method), X_train, X_test, variables)
        window_cuts.append(window_execution_df[CUT_COLUMN].iloc[0])
        window_cut_seconds.append(window_execution_df[CUT_SECONDS_COLUMN].iloc[0])
        executions_df_list.append(window_execution_df)
        errors_df_list.append(window_errors_df)
        approaches_list.append("Window " + method)
        print("Finished " + approaches_list[-1])
    mean_window_execution_df, mean_window_errors_df = execute_mean_cut(window_cuts, window_cut_seconds, X_train, X_test, variables)
    executions_df_list.append(mean_window_execution_df)
    errors_df_list.append(mean_window_errors_df)
    approaches_list.append("Window Mean")
    print("Finished " + approaches_list[-1])
    median_window_execution_df, median_window_errors_df = execute_median_cut(window_cuts, window_cut_seconds, X_train, X_test, variables)
    executions_df_list.append(median_window_execution_df)
    errors_df_list.append(median_window_errors_df)
    approaches_list.append("Window Median")
    print("Finished " + approaches_list[-1])

    # Binary Segmentation cuts
    binary_seg_cuts = []
    binary_seg_cut_seconds = []
    for method in METHODS:
        binary_seg_execution_df, binary_seg_errors_df = execute_binary_seg_cut(str.lower(method), X_train, X_test, variables)
        binary_seg_cuts.append(binary_seg_execution_df[CUT_COLUMN].iloc[0])
        binary_seg_cut_seconds.append(binary_seg_execution_df[CUT_SECONDS_COLUMN].iloc[0])
        executions_df_list.append(binary_seg_execution_df)
        errors_df_list.append(binary_seg_errors_df)
        approaches_list.append("Binary Segmentation " + method)
        print("Finished " + approaches_list[-1])
    mean_binary_seg_execution_df, mean_binary_seg_errors_df = execute_mean_cut(binary_seg_cuts, binary_seg_cut_seconds, X_train, X_test, variables)
    executions_df_list.append(mean_binary_seg_execution_df)
    errors_df_list.append(mean_binary_seg_errors_df)
    approaches_list.append("Binary Segmentation Mean")
    print("Finished " + approaches_list[-1])
    median_binary_seg_execution_df, median_binary_seg_errors_df = execute_median_cut(binary_seg_cuts, binary_seg_cut_seconds, X_train, X_test, variables)
    executions_df_list.append(median_binary_seg_execution_df)
    errors_df_list.append(median_binary_seg_errors_df)
    approaches_list.append("Binary Segmentation Median")
    print("Finished " + approaches_list[-1])

    # Bottom Up cuts
    bottom_up_cuts = []
    bottom_up_cut_seconds = []
    for method in METHODS:
        bottom_up_execution_df, bottom_up_errors_df = execute_bottom_up_cut(str.lower(method), X_train, X_test, variables)
        bottom_up_cuts.append(bottom_up_execution_df[CUT_COLUMN].iloc[0])
        bottom_up_cut_seconds.append(bottom_up_execution_df[CUT_SECONDS_COLUMN].iloc[0])
        executions_df_list.append(bottom_up_execution_df)
        errors_df_list.append(bottom_up_errors_df)
        approaches_list.append("Bottom Up " + method)
        print("Finished " + approaches_list[-1])
    mean_bottom_up_execution_df, mean_bottom_up_errors_df = execute_mean_cut(bottom_up_cuts, bottom_up_cut_seconds, X_train, X_test, variables)
    executions_df_list.append(mean_bottom_up_execution_df)
    errors_df_list.append(mean_bottom_up_errors_df)
    approaches_list.append("Bottom Up Mean")
    print("Finished " + approaches_list[-1])
    median_bottom_up_execution_df, median_bottom_up_errors_df = execute_median_cut(bottom_up_cuts, bottom_up_cut_seconds, X_train, X_test, variables)
    executions_df_list.append(median_bottom_up_execution_df)
    errors_df_list.append(median_bottom_up_errors_df)
    approaches_list.append("Bottom Up Median")
    print("Finished " + approaches_list[-1])

    # Save
    errors_df = pd.concat(errors_df_list)
    executions_df = pd.concat(executions_df_list)
    results_df = pd.concat([executions_df, errors_df], axis=1)
    results_df["Approach"] = approaches_list
    now = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
    results_df.to_csv("outputs/" + dataset_domain_argv + "/" + dataset_argv + "-" + now + ".csv")
