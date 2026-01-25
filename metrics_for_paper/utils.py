"""
Utility functions for metrics analysis notebooks.

This module provides functions organized into the following categories:
- Data Loading: Reading experiment results from JSON files
- Plotting: Visualization functions for scatter plots, Pareto fronts
- Statistical Analysis: ADF tests, Friedman tests
- LaTeX Generation: Creating publication-ready tables
- Multi-Objective Optimization: Pareto front identification, weighted scalarization
"""
import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from adjustText import adjust_text
from scipy import stats
from statsmodels.tsa.stattools import adfuller

from metrics_config import (
    CHANGE_POINT_APPROACH_COL, CHANGE_POINT_PERC_COL, DATASET_NAME_COL,
    DATASET_CSV_CONFIG, DATASETS_CONFIG, DATASETS_DISPLAY_NAMES, DATASETS_NAMES,
    ET_CPD_COL, ET_HPO_COL, ET_RETRAIN_COL, ET_TOTAL_COL,
    INPUT_FOLDER, METRIC_CONFIG, MODEL_TYPE_COL, R2_COL, RMSE_COL, SEEDS, TRAIN_PERC
)


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def get_df_for_dataset(
    seed: int,
    database_name: str,
    dataset_name: str,
    forecaster_type: str,
    input_folder: str = None
) -> pd.DataFrame:
    """Load experiment results for a specific dataset and seed.

    Args:
        seed: Random seed used in the experiment
        database_name: Database identifier (e.g., 'UCI', 'INMET')
        dataset_name: Dataset name within the database
        forecaster_type: Type of forecaster (e.g., 'LSTM', 'TCN')
        input_folder: Path to results folder (defaults to INPUT_FOLDER)

    Returns:
        DataFrame with experiment results including metrics and timing
    """
    if input_folder is None:
        input_folder = INPUT_FOLDER

    directory_path = f"{input_folder}/seed={seed}/dataset_domain={database_name}/dataset_name={dataset_name}"
    print(f"Reading from {directory_path}")

    # Collect file information
    file_info = []
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith('.json'):
                file_path = os.path.join(root, filename)
                path_parts = file_path.split(os.sep)

                cpd_method = cpd_cost_function = forecaster_type_found = timestamp = None
                for part in path_parts:
                    if part.startswith('cpd_method='):
                        cpd_method = part.split('=', 1)[1]
                    elif part.startswith('cpd_cost_function='):
                        cpd_cost_function = part.split('=', 1)[1]
                    elif part.startswith('forecaster_type='):
                        forecaster_type_found = part.split('=', 1)[1]
                    elif part.startswith('timestamp='):
                        timestamp = part.split('=', 1)[1]

                file_info.append({
                    'file_path': file_path,
                    'cpd_method': cpd_method,
                    'cpd_cost_function': cpd_cost_function,
                    'forecaster_type': forecaster_type_found,
                    'timestamp': timestamp
                })

    file_info_df = pd.DataFrame(file_info)
    if len(file_info_df) == 0:
        print(f"  WARNING: No files found in {directory_path}")
        return pd.DataFrame()

    # Filter by forecaster type
    file_info_df = file_info_df[file_info_df['forecaster_type'] == forecaster_type]
    if len(file_info_df) == 0:
        print(f"  WARNING: No files found for forecaster_type={forecaster_type}")
        return pd.DataFrame()

    # Keep only most recent run for each configuration
    file_info_df = file_info_df.sort_values('timestamp', ascending=False)
    file_info_df = file_info_df.drop_duplicates(
        subset=['cpd_method', 'cpd_cost_function', 'forecaster_type'], keep='first'
    )

    # Parse JSON files and extract metrics
    metrics_dfs = []
    for _, row in file_info_df.iterrows():
        try:
            with open(row['file_path'], 'r') as file:
                data = json.load(file)

                # Extract timing metrics (convert to minutes)
                et_cpd = data.get('detect_change_point_perf_duration', np.nan) / 60
                et_hpo = data.get('tuner_process_duration', np.nan) / 60
                et_retrain = data.get('retrain_perf_duration', np.nan) / 60
                et_total = et_cpd + et_hpo + et_retrain

                metrics_dfs.append(pd.DataFrame({
                    'cpd_method': [row['cpd_method'] or data.get('cpd_method', np.nan)],
                    'cpd_cost_function': [row['cpd_cost_function'] or data.get('cpd_cost_function', np.nan)],
                    'forecaster_type': [row['forecaster_type'] or data.get('forecaster_type', np.nan)],
                    'change_point_approach': [data.get('change_point_approach', np.nan)],
                    'change_point_perc': [data.get('change_point_perc', np.nan)],
                    ET_CPD_COL: [et_cpd],
                    ET_HPO_COL: [et_hpo],
                    ET_RETRAIN_COL: [et_retrain],
                    ET_TOTAL_COL: [et_total],
                    'Avg_MAPE': [data.get('error_results_real_only', {}).get('Avg_MAPE', np.nan)],
                    'Avg_MAE': [data.get('error_results_real_only', {}).get('Avg_MAE', np.nan)],
                    'Avg_MSE': [data.get('error_results_real_only', {}).get('Avg_MSE', np.nan)],
                    'Avg_RMSE': [data.get('error_results_real_only', {}).get('Avg_RMSE', np.nan)],
                    'Avg_R2': [data.get('error_results_real_only', {}).get('Avg_R2', np.nan)],
                }))
        except KeyError as e:
            print(f"Error in {row['file_path']}: {e}")

    if len(metrics_dfs) == 0:
        return pd.DataFrame()

    metrics_df = pd.concat(metrics_dfs, axis=0)

    # Clean up approach names for readability
    replacements = {
        'Rbf': 'RBF', ' Ar': ' AR', 'Bin_Seg': 'BinSeg', 'Bottom_Up': 'BottomUp',
        'Fixed_Perc Fixed_Cut_0.0': 'No CPD', 'Fixed_Perc Fixed_Cut_': 'Fixed Cut ',
        '0.1': '10%', '0.2': '20%', '0.3': '30%', '0.4': '40%', '0.5': '50%',
        '0.6': '60%', '0.7': '70%', '0.8': '80%', '0.9': '90%',
    }
    for old, new in replacements.items():
        metrics_df['change_point_approach'] = metrics_df['change_point_approach'].str.replace(old, new)

    metrics_df['database'] = database_name
    metrics_df['dataset'] = dataset_name
    return metrics_df.reset_index(drop=True)


def get_individual_results(datasets_dict: dict, dataset_name: str) -> pd.DataFrame:
    """Aggregate results across seeds for a dataset.

    Args:
        datasets_dict: Dictionary mapping dataset names to seed-indexed DataFrames
        dataset_name: Name of the dataset to aggregate

    Returns:
        DataFrame with mean and std computed across seeds for each metric
    """
    relevant_metrics = [
        CHANGE_POINT_PERC_COL, RMSE_COL, R2_COL,
        ET_CPD_COL, ET_HPO_COL, ET_RETRAIN_COL, ET_TOTAL_COL
    ]
    concat_df = None

    for seed in datasets_dict[dataset_name].keys():
        df = datasets_dict[dataset_name][seed].copy()
        df[DATASET_NAME_COL] = dataset_name
        df = df[[DATASET_NAME_COL, CHANGE_POINT_APPROACH_COL, MODEL_TYPE_COL] + relevant_metrics]
        df = df.rename(columns={m: f"{m}_seed_{seed}" for m in relevant_metrics})

        if concat_df is None:
            concat_df = df
        else:
            concat_df = pd.merge(
                concat_df, df,
                on=[DATASET_NAME_COL, CHANGE_POINT_APPROACH_COL, MODEL_TYPE_COL],
                how='left'
            )

    # Calculate mean and std across seeds
    for metric in relevant_metrics:
        metric_cols = [col for col in concat_df.columns if f"{metric}_seed_" in col]
        if metric_cols:
            concat_df[f"{metric}_mean"] = concat_df[metric_cols].mean(axis=1, skipna=True)
            concat_df[f"{metric}_std"] = concat_df[metric_cols].std(axis=1, skipna=True)

    # Reorder columns for clarity
    id_cols = [DATASET_NAME_COL, CHANGE_POINT_APPROACH_COL, MODEL_TYPE_COL]
    mean_std_cols = sorted([c for c in concat_df.columns if c.endswith('_mean') or c.endswith('_std')])
    seed_cols = [c for c in concat_df.columns if c not in id_cols and c not in mean_std_cols]

    concat_df = concat_df[id_cols + mean_std_cols + seed_cols]
    return concat_df.round(4).sort_values(by=[CHANGE_POINT_APPROACH_COL, MODEL_TYPE_COL])


# =============================================================================
# DATASET ANALYSIS FUNCTIONS
# =============================================================================

def load_dataset(dataset_name: str) -> pd.DataFrame:
    """Load a dataset from CSV file.

    Args:
        dataset_name: Name of the dataset (key in DATASETS_CONFIG)

    Returns:
        DataFrame with the dataset, filtered to keep_cols
    """
    config = DATASET_CSV_CONFIG[dataset_name]
    df = pd.read_csv(config['csv_path'])

    # Handle 'ds' column if present - try to parse datetime, but don't fail if it can't
    keep_cols = config['keep_cols']
    if 'ds' in keep_cols and 'ds' in df.columns:
        try:
            df['ds'] = pd.to_datetime(df['ds'])
        except Exception:
            # Keep as string if parsing fails
            pass

    # Filter to only keep_cols that exist in the dataframe
    existing_cols = [c for c in keep_cols if c in df.columns]
    return df[existing_cols]


def run_adf_tests(
    dataset_name: str,
    variable_display_names: dict = None
) -> pd.DataFrame:
    """Run Augmented Dickey-Fuller tests on all variables in a dataset.

    Args:
        dataset_name: Name of the dataset (key in DATASETS_CONFIG)
        variable_display_names: Optional dict mapping column names to display names

    Returns:
        DataFrame with ADF test results (statistic and p-value for each variable)
    """
    df = load_dataset(dataset_name)

    # Get numeric columns (exclude 'ds')
    numeric_cols = [c for c in df.columns if c != 'ds' and pd.api.types.is_numeric_dtype(df[c])]

    results = {'Variable': [], 'ADF Statistic': [], 'p-value': [], 'Stationary': []}

    for col in numeric_cols:
        series = df[col].dropna()
        if len(series) > 0:
            adf_result = adfuller(series)
            display_name = variable_display_names.get(col, col) if variable_display_names else col
            results['Variable'].append(display_name)
            results['ADF Statistic'].append(adf_result[0])
            results['p-value'].append(adf_result[1])
            results['Stationary'].append('Yes' if adf_result[1] < 0.05 else 'No')

    return pd.DataFrame(results)


def get_adf_latex_table(
    dataset_name: str,
    output_folder: str,
    variable_display_names: dict = None,
    file_name: str = None
) -> pd.DataFrame:
    """Generate LaTeX table for ADF test results.

    Args:
        dataset_name: Name of the dataset (key in DATASETS_CONFIG)
        output_folder: Directory to save the table
        variable_display_names: Optional dict mapping column names to display names
        file_name: Base filename for the output (default: {dataset_name}_adf)

    Returns:
        DataFrame with ADF test results
    """
    adf_df = run_adf_tests(dataset_name, variable_display_names)

    if file_name is None:
        file_name = f"{dataset_name.lower()}_adf"

    display_name = DATASETS_DISPLAY_NAMES.get(dataset_name, dataset_name)

    # Build LaTeX table
    n_vars = len(adf_df)
    col_headers = []
    for var in adf_df['Variable']:
        # Handle multi-word headers
        parts = var.split()
        if len(parts) > 1:
            col_headers.append(r"\makecell{\textbf{" + parts[0] + r"}\\\textbf{" + " ".join(parts[1:]) + r"}}")
        else:
            col_headers.append(r"\textbf{" + var + r"}")

    # Format values
    adf_stats = []
    p_values = []
    for _, row in adf_df.iterrows():
        adf_stats.append(f"{row['ADF Statistic']:.2f}")
        pval = row['p-value']
        if pval == 0:
            p_values.append(r"0.0$\times10^{0}$")
        elif pval < 0.01:
            exp = int(np.floor(np.log10(pval)))
            mantissa = pval / (10 ** exp)
            p_values.append(f"{mantissa:.1f}$\\times10^{{{exp}}}$")
        else:
            p_values.append(f"{pval:.2e}".replace('e', r'$\times10^{').replace('-0', '-') + '}$')

    col_spec = "l" + "c" * n_vars

    latex_table = f"""\\begin{{table}}[!htbp] \\scriptsize
    \\caption{{Augmented Dickey-Fuller test results of the {display_name} dataset, to assess stationarity.}}
    \\centering
    \\begin{{tabular}}{{{col_spec}}}
        \\toprule
        & {" & ".join(col_headers)} \\\\
        \\midrule
        \\textbf{{ADF Statistic}} & {" & ".join(adf_stats)} \\\\
        \\textbf{{p-value}} & {" & ".join(p_values)} \\\\
        \\bottomrule
    \\end{{tabular}}
    \\label{{tab:{dataset_name.lower()}ADF}}
\\end{{table}}"""

    os.makedirs(output_folder, exist_ok=True)
    filepath = f"{output_folder}/{file_name}.tex"
    with open(filepath, 'w') as f:
        f.write(latex_table)
    print(f"Saved to {filepath}")

    return adf_df


def plot_change_points(
    dataset_name: str,
    individual_metrics_df: pd.DataFrame,
    output_folder: str,
    n_approaches: int = 5,
    filename: str = None
) -> None:
    """Plot time series with detected change points from different CPD approaches.

    Args:
        dataset_name: Name of the dataset (key in DATASETS_CONFIG)
        individual_metrics_df: DataFrame with individual results containing change_point_perc
        output_folder: Directory to save the plot
        n_approaches: Number of top approaches to show (by lowest ET_CPD)
        filename: Optional filename to save the plot
    """
    # Load dataset
    df = load_dataset(dataset_name)
    numeric_cols = [c for c in df.columns if c != 'ds' and pd.api.types.is_numeric_dtype(df[c])]

    if len(numeric_cols) == 0:
        print(f"No numeric columns found in {dataset_name}")
        return

    # Use first numeric column for visualization
    target_col = numeric_cols[0]

    # Get change point percentages from different approaches
    # Filter out Fixed Cut approaches and get unique CPD methods
    cpd_df = individual_metrics_df[
        ~individual_metrics_df[CHANGE_POINT_APPROACH_COL].str.startswith('Fixed Cut') &
        (individual_metrics_df[CHANGE_POINT_APPROACH_COL] != 'No CPD')
    ].copy()

    if len(cpd_df) == 0:
        print(f"No CPD approaches found for {dataset_name}")
        return

    # Get unique approaches sorted by ET_CPD (fastest first)
    cpd_df = cpd_df.sort_values(f"{ET_CPD_COL}_mean")
    approaches = cpd_df[CHANGE_POINT_APPROACH_COL].unique()[:n_approaches]

    # Get change point percentages for selected approaches
    cp_data = []
    for approach in approaches:
        row = cpd_df[cpd_df[CHANGE_POINT_APPROACH_COL] == approach].iloc[0]
        cp_perc = row[f"{CHANGE_POINT_PERC_COL}_seed_{SEEDS[0]}"]
        cp_data.append((approach, cp_perc))

    # Calculate train/test split
    train_size = int(len(df) * TRAIN_PERC)

    # Create plot
    fig, ax = plt.subplots(figsize=(14, 5))

    # Plot time series
    ax.plot(df.index, df[target_col], linewidth=1, color='steelblue', alpha=0.8, label=target_col)

    # Add train/test split line
    ax.axvline(x=train_size, color='black', linestyle='-', linewidth=2, alpha=0.7, label='Train/Test Split')

    # Add change points (all in red)
    for approach, cp_perc in cp_data:
        if pd.notna(cp_perc) and cp_perc > 0:
            cp_idx = int(len(df) * cp_perc / 100)
            ax.axvline(x=cp_idx, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
                      label=f'{approach} ({cp_perc:.1f}%)')

    ax.set_xlabel('Time Index', fontsize=11)
    ax.set_ylabel(target_col, fontsize=11)

    ax.legend(loc='upper left', fontsize=9, bbox_to_anchor=(1.02, 1))
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename is None:
        filename = f"{dataset_name.lower()}_change_points.png"

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/{filename}", dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================

def plot_scatter_plot(
    df: pd.DataFrame,
    output_folder: str,
    metric_col: str,
    primary_metric_col: str = None,
    filename: str = None,
    dataset_name: str = None,
    forecaster_type: str = None
) -> None:
    """Plot scatter plot of Metric vs Execution Time.

    Args:
        df: DataFrame with results
        output_folder: Directory to save the plot
        metric_col: Column name for the metric (x-axis)
        primary_metric_col: Alternative column name (deprecated, use metric_col)
        filename: Optional filename to save the plot
        dataset_name: Optional dataset name to include in title
        forecaster_type: Optional forecaster type (e.g., 'LSTM', 'TCN') to include in title
    """
    if metric_col is None:
        metric_col = primary_metric_col

    metric_config = METRIC_CONFIG[metric_col]
    plt.figure(figsize=(10, 6))

    scatter = plt.scatter(
        df[metric_col], df[ET_TOTAL_COL],
        c=df[CHANGE_POINT_PERC_COL], cmap="viridis_r",
        s=100, alpha=0.8, edgecolors="k"
    )
    plt.colorbar(scatter, label="Change Point %")

    texts = [
        plt.text(row[metric_col], row[ET_TOTAL_COL], row[CHANGE_POINT_APPROACH_COL], fontsize=9)
        for _, row in df.iterrows()
    ]
    adjust_text(
        texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        force_text=(0.3, 0.3), force_points=(0.3, 0.3),
        expand_text=(1.02, 1.02), expand_points=(1.02, 1.02)
    )

    plt.xlabel(f"Average {metric_config['display_name']}")
    plt.ylabel("Total Execution Time (minutes)")

    # Build title with dataset and forecaster info
    title = f"{metric_config['display_name']} vs Execution Time"
    subtitle_parts = []
    if dataset_name:
        subtitle_parts.append(DATASETS_DISPLAY_NAMES.get(dataset_name, dataset_name))
    if forecaster_type:
        subtitle_parts.append(forecaster_type)
    if subtitle_parts:
        title = f"{title}\n({' - '.join(subtitle_parts)})"
    plt.title(title, fontsize=14, fontweight='bold')

    plt.grid(True)
    plt.tight_layout()

    if filename:
        plt.savefig(f"{output_folder}/{filename}", dpi=300, bbox_inches='tight')
    plt.show()


def plot_aggregated_scatter(
    comp_df: pd.DataFrame,
    output_folder: str,
    metric_col: str = RMSE_COL,
    filename: str = "agg_scatter_plot.png"
) -> pd.DataFrame:
    """Plot aggregated scatter plot with rankings across all datasets.

    Args:
        comp_df: DataFrame with comparison data
        output_folder: Directory to save the plot
        metric_col: Column name for the metric
        filename: Filename for the saved plot

    Returns:
        Aggregated DataFrame with rankings
    """
    agg_df = comp_df.groupby(CHANGE_POINT_APPROACH_COL).agg({
        f"{metric_col}_mean": "mean",
        f"{ET_TOTAL_COL}_mean": "mean",
        f"{CHANGE_POINT_PERC_COL}_seed_{SEEDS[0]}": "mean"
    }).reset_index()

    agg_df["metric_rank"] = agg_df[f"{metric_col}_mean"].rank(ascending=True)
    agg_df["et_rank"] = agg_df[f"{ET_TOTAL_COL}_mean"].rank(ascending=True)

    plt.figure(figsize=(12, 8))
    scatter = plt.scatter(
        agg_df["metric_rank"], agg_df["et_rank"],
        c=agg_df[f"{CHANGE_POINT_PERC_COL}_seed_{SEEDS[0]}"],
        cmap="viridis_r", s=100, alpha=0.85, edgecolors="k"
    )
    plt.colorbar(scatter, label="Avg Change Point %")

    texts = [
        plt.text(row["metric_rank"], row["et_rank"], row[CHANGE_POINT_APPROACH_COL], fontsize=9)
        for _, row in agg_df.iterrows()
    ]
    adjust_text(
        texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        force_text=(0.3, 0.3), force_points=(0.3, 0.3),
        expand_text=(1.02, 1.02), expand_points=(1.02, 1.02)
    )

    metric_display = METRIC_CONFIG[metric_col]['display_name']
    plt.xlabel(f"Global {metric_display} Rank")
    plt.ylabel("Global Execution Time Rank")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/{filename}", dpi=300, bbox_inches='tight')
    plt.show()

    return agg_df


# =============================================================================
# PARETO FRONT ANALYSIS
# =============================================================================

def identify_pareto_front(
    df: pd.DataFrame,
    metric_col: str,
    et_col: str,
    minimize_metric: bool = True,
    minimize_et: bool = True
) -> list:
    """Identify Pareto-optimal approaches.

    Args:
        df: DataFrame with results
        metric_col: Column name for the accuracy metric
        et_col: Column name for execution time
        minimize_metric: True if lower metric values are better
        minimize_et: True if lower execution time is better

    Returns:
        List of boolean values indicating Pareto optimality for each row
    """
    pareto_optimal = []

    for i in range(len(df)):
        is_dominated = False
        for j in range(len(df)):
            if i != j:
                # Check if j dominates i
                if minimize_metric:
                    metric_j_better_or_equal = df[metric_col].iloc[j] <= df[metric_col].iloc[i]
                    metric_j_strictly_better = df[metric_col].iloc[j] < df[metric_col].iloc[i]
                else:
                    metric_j_better_or_equal = df[metric_col].iloc[j] >= df[metric_col].iloc[i]
                    metric_j_strictly_better = df[metric_col].iloc[j] > df[metric_col].iloc[i]

                if minimize_et:
                    et_j_better_or_equal = df[et_col].iloc[j] <= df[et_col].iloc[i]
                    et_j_strictly_better = df[et_col].iloc[j] < df[et_col].iloc[i]
                else:
                    et_j_better_or_equal = df[et_col].iloc[j] >= df[et_col].iloc[i]
                    et_j_strictly_better = df[et_col].iloc[j] > df[et_col].iloc[i]

                if (metric_j_better_or_equal and et_j_better_or_equal and
                    (metric_j_strictly_better or et_j_strictly_better)):
                    is_dominated = True
                    break

        pareto_optimal.append(not is_dominated)

    return pareto_optimal


def plot_pareto_front(
    df: pd.DataFrame,
    output_folder: str,
    metric_col: str,
    primary_metric_col: str = None,
    filename: str = None,
    dataset_name: str = None,
    forecaster_type: str = None
) -> pd.DataFrame:
    """Plot Pareto front analysis.

    Args:
        df: DataFrame with results
        output_folder: Directory to save the plot
        metric_col: Column name for the metric
        primary_metric_col: Alternative column name (deprecated)
        filename: Optional filename to save the plot
        dataset_name: Optional dataset name to include in title
        forecaster_type: Optional forecaster type (e.g., 'LSTM', 'TCN') to include in title

    Returns:
        DataFrame with only Pareto-optimal approaches
    """
    if metric_col is None:
        metric_col = primary_metric_col

    metric_config = METRIC_CONFIG[metric_col]
    minimize_metric = metric_config["minimize"]

    df = df.copy()
    df['is_pareto'] = identify_pareto_front(
        df, metric_col, ET_TOTAL_COL,
        minimize_metric=minimize_metric, minimize_et=True
    )

    pareto_df = df[df['is_pareto']].sort_values(by=metric_col, ascending=minimize_metric)

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot dominated points
    non_pareto = df[~df['is_pareto']]
    ax.scatter(
        non_pareto[metric_col], non_pareto[ET_TOTAL_COL],
        c='lightgray', s=100, alpha=0.5, edgecolors="k", label='Dominated', zorder=1
    )

    # Plot Pareto front
    scatter = ax.scatter(
        pareto_df[metric_col], pareto_df[ET_TOTAL_COL],
        c=pareto_df[CHANGE_POINT_PERC_COL], cmap="viridis_r",
        s=150, alpha=0.9, edgecolors="black", linewidths=2,
        label='Pareto front', zorder=3
    )
    ax.plot(pareto_df[metric_col], pareto_df[ET_TOTAL_COL], 'r--', linewidth=2, alpha=0.6, zorder=2)

    plt.colorbar(scatter, ax=ax, label="Change Point %")

    texts = [
        ax.text(
            row[metric_col], row[ET_TOTAL_COL], row[CHANGE_POINT_APPROACH_COL],
            fontsize=9, color='black' if row['is_pareto'] else 'gray',
            weight='bold' if row['is_pareto'] else 'normal'
        )
        for _, row in df.iterrows()
    ]
    adjust_text(
        texts, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5),
        force_text=(0.3, 0.3), force_points=(0.3, 0.3),
        expand_text=(1.02, 1.02), expand_points=(1.02, 1.02)
    )

    ax.set_xlabel(f"Average {metric_config['display_name']}", fontsize=12)
    ax.set_ylabel("Total Execution Time (minutes)", fontsize=12)
    title = f"Pareto Front: {metric_config['display_name']} vs Execution Time"
    subtitle_parts = []
    if dataset_name:
        subtitle_parts.append(DATASETS_DISPLAY_NAMES.get(dataset_name, dataset_name))
    if forecaster_type:
        subtitle_parts.append(forecaster_type)
    if subtitle_parts:
        title = f"{title}\n({' - '.join(subtitle_parts)})"
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if filename:
        plt.savefig(f"{output_folder}/{filename}", dpi=300, bbox_inches='tight')
    plt.show()

    # Print summary
    context_parts = []
    if dataset_name:
        context_parts.append(DATASETS_DISPLAY_NAMES.get(dataset_name, dataset_name))
    if forecaster_type:
        context_parts.append(forecaster_type)
    context_str = f" - {' / '.join(context_parts)}" if context_parts else ""
    print(f"\nPareto Optimal Approaches{context_str} (optimizing {metric_config['display_name']} and ET):")
    print("=" * 80)
    metric_fmt = metric_config['format']
    for _, row in pareto_df.iterrows():
        print(f"{row[CHANGE_POINT_APPROACH_COL]:30s} | {metric_config['display_name']}: {row[metric_col]:{metric_fmt}} | ET: {row[ET_TOTAL_COL]:8.2f} min")

    return pareto_df


def analyze_pareto_consistency(
    comp_df: pd.DataFrame,
    metric_col: str,
    output_folder: str
) -> tuple:
    """Analyze which methods appear consistently across Pareto fronts.

    Args:
        comp_df: DataFrame with comparison data across datasets
        metric_col: Column name for the primary metric
        output_folder: Directory to save outputs

    Returns:
        Tuple of (pareto_summary DataFrame, pareto_methods_by_dataset dict)
    """
    metric_config = METRIC_CONFIG[metric_col]
    minimize_metric = metric_config["minimize"]
    metric_display = metric_config["display_name"]

    print(f"Running Pareto Analysis with: {metric_display} ({'minimize' if minimize_metric else 'maximize'}) + ET (minimize)")
    print()

    pareto_methods_by_dataset = {}
    all_pareto_methods = []

    # Get unique datasets from the DataFrame instead of using global constant
    datasets = comp_df[DATASET_NAME_COL].unique().tolist()

    for dataset in datasets:
        df = comp_df[comp_df[DATASET_NAME_COL] == dataset].copy()
        df = df.rename(columns={
            f"{CHANGE_POINT_PERC_COL}_seed_{SEEDS[0]}": CHANGE_POINT_PERC_COL,
            f"{metric_col}_mean": metric_col,
            f"{ET_TOTAL_COL}_mean": ET_TOTAL_COL,
        })
        df['is_pareto'] = identify_pareto_front(
            df, metric_col, ET_TOTAL_COL,
            minimize_metric=minimize_metric, minimize_et=True
        )
        pareto_methods = df[df['is_pareto']][CHANGE_POINT_APPROACH_COL].tolist()
        pareto_methods_by_dataset[dataset] = set(pareto_methods)
        all_pareto_methods.extend(pareto_methods)

    # Count occurrences
    pareto_counts = Counter(all_pareto_methods)
    n_datasets = len(datasets)

    # Create summary DataFrame
    pareto_summary = pd.DataFrame([
        {
            'Method': method,
            'Pareto Count': count,
            'Pareto Percentage': round(count / n_datasets * 100, 1),
            'Datasets': ', '.join([ds for ds in datasets if method in pareto_methods_by_dataset[ds]])
        }
        for method, count in pareto_counts.items()
    ]).sort_values('Pareto Count', ascending=False)

    # Print summary
    print("=" * 80)
    print("PARETO FRONT CONSISTENCY ANALYSIS")
    print(f"Optimizing: {metric_display} + Execution Time")
    print("=" * 80)
    print(f"\nTotal datasets analyzed: {n_datasets}")
    print(f"Total unique methods in any Pareto front: {len(pareto_counts)}")

    methods_in_all = [m for m, c in pareto_counts.items() if c == n_datasets]
    print(f"\nMethods appearing in ALL Pareto fronts ({n_datasets}/{n_datasets} = 100%):")
    if methods_in_all:
        for m in methods_in_all:
            print(f"  - {m}")
    else:
        print("  None")

    methods_in_majority = [m for m, c in pareto_counts.items() if c >= n_datasets / 2 and c < n_datasets]
    print(f"\nMethods appearing in majority of Pareto fronts (>= 50%):")
    if methods_in_majority:
        for m in methods_in_majority:
            pct = round(pareto_counts[m] / n_datasets * 100, 1)
            print(f"  - {m}: {pareto_counts[m]}/{n_datasets} ({pct}%)")
    else:
        print("  None")

    return pareto_summary, pareto_methods_by_dataset


def plot_pareto_heatmap(
    pareto_summary: pd.DataFrame,
    pareto_methods_by_dataset: dict,
    metric_col: str,
    output_folder: str
) -> None:
    """Plot heatmap showing Pareto front membership across datasets.

    Args:
        pareto_summary: Summary DataFrame from analyze_pareto_consistency
        pareto_methods_by_dataset: Dict mapping datasets to Pareto methods
        metric_col: Column name for the primary metric
        output_folder: Directory to save the plot
    """
    metric_display = METRIC_CONFIG[metric_col]['display_name']
    all_methods_sorted = pareto_summary['Method'].tolist()

    # Get datasets from the pareto_methods_by_dataset keys
    datasets = list(pareto_methods_by_dataset.keys())

    heatmap_data = pd.DataFrame(index=all_methods_sorted, columns=datasets)
    for method in all_methods_sorted:
        for dataset in datasets:
            heatmap_data.loc[method, dataset] = 1 if method in pareto_methods_by_dataset[dataset] else 0

    heatmap_data = heatmap_data.astype(int)

    fig, ax = plt.subplots(figsize=(14, max(8, len(all_methods_sorted) * 0.4)))
    sns.heatmap(
        heatmap_data, annot=True, cmap='YlGn',
        cbar_kws={'label': 'In Pareto Front'},
        linewidths=0.5, ax=ax, fmt='d'
    )
    ax.set_xlabel('Dataset', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    ax.set_title(f'Pareto Front Membership Across Datasets\n(Optimizing {metric_display} + ET)', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    os.makedirs(output_folder, exist_ok=True)
    plt.savefig(f"{output_folder}/pareto_front_heatmap.png", dpi=300, bbox_inches='tight')
    plt.show()


# =============================================================================
# MULTI-CRITERIA EVALUATION METHODS
# =============================================================================

def compute_weighted_scalarization(
    df: pd.DataFrame,
    metric_col: str,
    et_col: str,
    approach_col: str,
    alpha: float = 0.5,
    minimize_metric: bool = True,
    minimize_et: bool = True
) -> pd.DataFrame:
    """Compute weighted scalarization scores.

    Score = α·(normalized metric) + (1-α)·(normalized ET)
    Lower score is better when both objectives are minimized.

    Args:
        df: DataFrame with results
        metric_col: Column name for accuracy metric
        et_col: Column name for execution time
        approach_col: Column name for method identifier
        alpha: Weight for accuracy metric (0-1), higher = more weight on accuracy
        minimize_metric: True if lower metric is better
        minimize_et: True if lower ET is better

    Returns:
        DataFrame with normalized values and weighted scores, sorted by score
    """
    df = df.copy()

    # Min-max normalization
    metric_min, metric_max = df[metric_col].min(), df[metric_col].max()
    et_min, et_max = df[et_col].min(), df[et_col].max()

    # Normalize so that lower = better for both
    if metric_max != metric_min:
        if minimize_metric:
            df['metric_norm'] = (df[metric_col] - metric_min) / (metric_max - metric_min)
        else:
            df['metric_norm'] = (metric_max - df[metric_col]) / (metric_max - metric_min)
    else:
        df['metric_norm'] = 0

    if et_max != et_min:
        if minimize_et:
            df['et_norm'] = (df[et_col] - et_min) / (et_max - et_min)
        else:
            df['et_norm'] = (et_max - df[et_col]) / (et_max - et_min)
    else:
        df['et_norm'] = 0

    # Weighted score (lower is better)
    df['weighted_score'] = alpha * df['metric_norm'] + (1 - alpha) * df['et_norm']
    df['rank'] = df['weighted_score'].rank(method='min')

    return df.sort_values('weighted_score')


def compute_weighted_scalarization_sensitivity(
    comp_df: pd.DataFrame,
    metric_col: str,
    alphas: list = None
) -> pd.DataFrame:
    """Compute weighted scalarization for multiple alpha values.

    Args:
        comp_df: Comparison DataFrame with all datasets
        metric_col: Primary metric column
        alphas: List of alpha values to test (default: 0.1 to 0.9)

    Returns:
        DataFrame showing rankings for each alpha value
    """
    if alphas is None:
        alphas = [0.1, 0.3, 0.5, 0.7, 0.9]

    # Aggregate across datasets
    agg_df = comp_df.groupby(CHANGE_POINT_APPROACH_COL).agg({
        f"{metric_col}_mean": "mean",
        f"{ET_TOTAL_COL}_mean": "mean",
    }).reset_index()

    results = {CHANGE_POINT_APPROACH_COL: agg_df[CHANGE_POINT_APPROACH_COL].tolist()}

    for alpha in alphas:
        scored = compute_weighted_scalarization(
            agg_df, f"{metric_col}_mean", f"{ET_TOTAL_COL}_mean",
            CHANGE_POINT_APPROACH_COL, alpha=alpha,
            minimize_metric=METRIC_CONFIG[metric_col]['minimize']
        )
        # Map ranks back
        rank_map = dict(zip(scored[CHANGE_POINT_APPROACH_COL], scored['rank']))
        results[f'α={alpha}'] = [rank_map[m] for m in results[CHANGE_POINT_APPROACH_COL]]

    result_df = pd.DataFrame(results)
    result_df['avg_rank'] = result_df[[c for c in result_df.columns if c.startswith('α=')]].mean(axis=1)
    return result_df.sort_values('avg_rank')


def compute_dominance_ranking(
    comp_df: pd.DataFrame,
    metric_col: str
) -> pd.DataFrame:
    """Compute dominance ranking based on how many solutions dominate each approach.

    For each approach, counts how many other approaches dominate it (i.e., are better
    in both metric and execution time). Lower dominated_count means better performance.

    Args:
        comp_df: Comparison DataFrame with all datasets
        metric_col: Primary metric column

    Returns:
        DataFrame with dominated counts and rankings for each approach
    """
    minimize_metric = METRIC_CONFIG[metric_col]['minimize']

    # Aggregate across datasets
    agg_df = comp_df.groupby(CHANGE_POINT_APPROACH_COL).agg({
        f"{metric_col}_mean": "mean",
        f"{ET_TOTAL_COL}_mean": "mean",
    }).reset_index()

    metric_values = agg_df[f"{metric_col}_mean"].values
    et_values = agg_df[f"{ET_TOTAL_COL}_mean"].values
    n = len(agg_df)

    dominated_counts = []
    for i in range(n):
        count = 0
        for j in range(n):
            if i != j:
                # Check if j dominates i
                if minimize_metric:
                    metric_j_better_or_equal = metric_values[j] <= metric_values[i]
                    metric_j_strictly_better = metric_values[j] < metric_values[i]
                else:
                    metric_j_better_or_equal = metric_values[j] >= metric_values[i]
                    metric_j_strictly_better = metric_values[j] > metric_values[i]

                et_j_better_or_equal = et_values[j] <= et_values[i]
                et_j_strictly_better = et_values[j] < et_values[i]

                # j dominates i if j is at least as good in both objectives
                # and strictly better in at least one
                if (metric_j_better_or_equal and et_j_better_or_equal and
                    (metric_j_strictly_better or et_j_strictly_better)):
                    count += 1
        dominated_counts.append(count)

    result_df = pd.DataFrame({
        CHANGE_POINT_APPROACH_COL: agg_df[CHANGE_POINT_APPROACH_COL],
        f"{metric_col}_mean": agg_df[f"{metric_col}_mean"],
        f"{ET_TOTAL_COL}_mean": agg_df[f"{ET_TOTAL_COL}_mean"],
        'dominated_count': dominated_counts
    })

    result_df['rank'] = result_df['dominated_count'].rank(method='min')
    return result_df.sort_values(['dominated_count', f"{metric_col}_mean"])


# =============================================================================
# STATISTICAL ANALYSIS FUNCTIONS
# =============================================================================

def get_friedman_significance_overall(dfs_dict: dict, col: str) -> None:
    """Perform Friedman test across all datasets.

    Args:
        dfs_dict: Dictionary mapping dataset names to DataFrames
        col: Column name to test
    """
    results = None
    for dataset_name, df in dfs_dict.items():
        df = df.copy()
        df = df.rename(columns={col: f"{dataset_name}_{col}"})
        df_subset = df[[CHANGE_POINT_APPROACH_COL, f"{dataset_name}_{col}"]]
        if results is None:
            results = df_subset
        else:
            results = pd.merge(results, df_subset, on=CHANGE_POINT_APPROACH_COL, how='inner')

    data_cols = [c for c in results.columns if c != CHANGE_POINT_APPROACH_COL]
    rows_with_nan = results[data_cols].isna().any(axis=1).sum()

    if rows_with_nan > 0:
        print(f"  WARNING: {rows_with_nan} approach(es) with missing data removed from analysis")
        results = results.dropna()

    if len(results) < 3:
        print(f"  ERROR: Not enough complete approaches for Friedman test (need >= 3, have {len(results)})")
        print("  Friedman Test: stat=N/A, p=N/A")
        return

    data = results.drop(columns=[CHANGE_POINT_APPROACH_COL]).T
    friedman_stat, p_value = stats.friedmanchisquare(*data.values.T)
    print(f"  Friedman Test: stat={friedman_stat:.4f}, p={p_value:.6f} (using {len(results)} approaches)")
    print(f"  {'Significant difference detected.' if p_value < 0.05 else 'No significant difference.'}")


# =============================================================================
# LATEX TABLE GENERATION FUNCTIONS
# =============================================================================

def get_cpd_cuts_latex_table(
    individual_metrics_dict: dict,
    output_folder: str,
    file_name: str = "cpd_cuts_table"
) -> None:
    """Generate LaTeX table for Change Point Detection cuts across all datasets.

    Args:
        individual_metrics_dict: Dictionary mapping dataset names to results DataFrames
        output_folder: Directory to save the table
        file_name: Base filename for the output
    """
    datasets = DATASETS_NAMES
    datasets_part1 = datasets[:3]
    datasets_part2 = datasets[3:]

    def format_header(name):
        return DATASETS_DISPLAY_NAMES.get(name, name)

    all_approaches = individual_metrics_dict[datasets[0]][CHANGE_POINT_APPROACH_COL].tolist()
    cpd_approaches = [a for a in all_approaches if not a.startswith('Fixed Cut')]
    if 'No CPD' in cpd_approaches:
        cpd_approaches.remove('No CPD')
        cpd_approaches = ['No CPD'] + cpd_approaches

    def format_value(val, width=5):
        return f"{val:0{width}.2f}"

    def get_approach_group(approach):
        """Get the group prefix for an approach."""
        if approach == 'No CPD':
            return 'No CPD'
        for prefix in ['BinSeg', 'BottomUp', 'Window']:
            if approach.startswith(prefix):
                return prefix
        return 'Other'

    def build_subtable(dataset_list):
        num_datasets = len(dataset_list)
        latex_rows = []
        prev_group = None

        for approach in cpd_approaches:
            # Add midrule between different CPD method groups
            current_group = get_approach_group(approach)
            if prev_group is not None and current_group != prev_group:
                latex_rows.append(r"            \midrule")
            prev_group = current_group

            values = []
            for dataset in dataset_list:
                df = individual_metrics_dict[dataset]
                row = df[df[CHANGE_POINT_APPROACH_COL] == approach]
                if len(row) > 0:
                    cp_perc = row[f"{CHANGE_POINT_PERC_COL}_seed_{SEEDS[0]}"].iloc[0]
                    et_cpd_mean = row[f"{ET_CPD_COL}_mean"].iloc[0]
                    et_cpd_std = row[f"{ET_CPD_COL}_std"].iloc[0]
                    values.append(f"{format_value(cp_perc)}\\%")
                    values.append(f"{format_value(et_cpd_mean)} $\\pm$ {format_value(et_cpd_std)}")
                else:
                    values.append("--")
                    values.append("--")
            latex_rows.append(f"            {approach} & " + " & ".join(values) + r" \\")

        dataset_headers = " & ".join([
            f"\\multicolumn{{2}}{{c}}{{\\textbf{{{format_header(ds)}}}}}"
            for ds in dataset_list
        ])
        sub_headers = " & ".join(["Cut \\%" + " & " + "$\\text{ET}_{\\text{CPD}}$" for _ in dataset_list])
        col_spec = "l" + "cc" * num_datasets
        cmidrules = " ".join([f"\\cmidrule(lr){{{2 + i*2}-{3 + i*2}}}" for i in range(num_datasets)])

        return f"""        \\begin{{tabular}}{{{col_spec}}}
            \\toprule
            & {dataset_headers} \\\\
            {cmidrules}
            & {sub_headers} \\\\
            \\midrule
{chr(10).join(latex_rows)}
            \\bottomrule
        \\end{{tabular}}"""

    subtable1 = build_subtable(datasets_part1)
    subtable2 = build_subtable(datasets_part2) if datasets_part2 else ""

    latex_table = f"""\\begin{{table}}[!htpb]
    \\scriptsize
    \\caption{{Detected change point (Cut \\%) and CPD execution time ($\\text{{ET}}_{{\\text{{CPD}}}}$, in minutes as mean$\\pm$std) for each CPD approach.}}
    \\label{{tab:changePointPercResults}}
    \\centering
    \\vspace{{0.5em}}
{subtable1}"""

    if subtable2:
        latex_table += f"""

    \\vspace{{1em}}

{subtable2}"""

    latex_table += "\n\\end{table}"

    os.makedirs(output_folder, exist_ok=True)
    with open(f"{output_folder}/{file_name}.tex", "w") as f:
        f.write(latex_table)
    print(f"Saved to {output_folder}/{file_name}.tex")


def get_latex_tables(
    individual_metrics_dict: dict,
    col_prefix: str,
    file_name: str,
    table_caption: str,
    table_label: str,
    output_path: str,
    metric_name: str = None,
    forecaster_type: str = None,
    higher_is_better: bool = False
) -> None:
    """Generate LaTeX table for results, split into two subtables.

    Args:
        individual_metrics_dict: Dictionary mapping dataset names to results DataFrames
        col_prefix: Prefix for the metric columns (e.g., 'Avg_RMSE')
        file_name: Base filename for the output
        table_caption: Caption for the LaTeX table
        table_label: Label for the LaTeX table
        output_path: Directory to save the table
        metric_name: Optional metric name to append to caption
        forecaster_type: Optional forecaster type to append to caption
        higher_is_better: True if higher metric values are better
    """
    all_approaches = individual_metrics_dict[DATASETS_NAMES[0]][CHANGE_POINT_APPROACH_COL].tolist()
    approaches = [a for a in all_approaches if not a.startswith('Fixed Cut')]

    if 'No CPD' in approaches:
        approaches.remove('No CPD')
        approaches = ['No CPD'] + approaches

    mean_col, std_col = f"{col_prefix}_mean", f"{col_prefix}_std"

    # Get baseline (No CPD) values
    baseline_values = {}
    for ds in individual_metrics_dict:
        df = individual_metrics_dict[ds]
        baseline_row = df[df[CHANGE_POINT_APPROACH_COL] == 'No CPD']
        baseline_values[ds] = baseline_row[mean_col].iloc[0] if len(baseline_row) > 0 else None

    # Get best values
    best_values = {}
    for ds in individual_metrics_dict:
        df = individual_metrics_dict[ds]
        filtered_df = df[df[CHANGE_POINT_APPROACH_COL].isin(approaches)]
        if higher_is_better:
            best_values[ds] = filtered_df[mean_col].max()
        else:
            best_values[ds] = filtered_df[mean_col].min()

    datasets_part1 = DATASETS_NAMES[:3]
    datasets_part2 = DATASETS_NAMES[3:]

    def format_header(name):
        display_name = DATASETS_DISPLAY_NAMES.get(name, name)
        parts = display_name.split()
        if len(parts) > 1:
            return r"\makecell{\textbf{" + parts[0] + r"}\\\textbf{" + " ".join(parts[1:]) + r"}}"
        return r"\textbf{" + display_name + r"}"

    def format_value(val, width=5):
        return f"{val:0{width}.2f}"

    def get_comparison_str(current_val, baseline_val, higher_is_better):
        if baseline_val is None or baseline_val == 0:
            return ""
        pct_change = ((current_val - baseline_val) / abs(baseline_val)) * 100
        if higher_is_better:
            if pct_change > 0:
                return f" ($\\uparrow${format_value(abs(pct_change))}\\%)"
            elif pct_change < 0:
                return f" ($\\downarrow${format_value(abs(pct_change))}\\%)"
        else:
            if pct_change < 0:
                return f" ($\\downarrow${format_value(abs(pct_change))}\\%)"
            elif pct_change > 0:
                return f" ($\\uparrow${format_value(abs(pct_change))}\\%)"
        return ""

    def get_approach_group(approach):
        """Get the group prefix for an approach."""
        if approach == 'No CPD':
            return 'No CPD'
        for prefix in ['BinSeg', 'BottomUp', 'Window']:
            if approach.startswith(prefix):
                return prefix
        return 'Other'

    def build_subtable(dataset_list):
        num_datasets = len(dataset_list)
        latex_rows = []
        prev_group = None

        for approach in approaches:
            # Add midrule between different CPD method groups
            current_group = get_approach_group(approach)
            if prev_group is not None and current_group != prev_group:
                latex_rows.append(r"        \midrule")
            prev_group = current_group

            values = []
            for dataset in dataset_list:
                df = individual_metrics_dict[dataset]
                row = df[df[CHANGE_POINT_APPROACH_COL] == approach]
                if len(row) > 0:
                    mean_val = row[mean_col].iloc[0]
                    std_val = row[std_col].iloc[0]
                    value_str = f"{format_value(mean_val)} $\\pm$ {format_value(std_val)}"

                    if approach != 'No CPD':
                        comparison = get_comparison_str(mean_val, baseline_values[dataset], higher_is_better)
                        value_str = value_str + comparison

                    if abs(mean_val - best_values[dataset]) < 1e-9:
                        value_str = f"\\textbf{{{value_str}}}"
                    values.append(value_str)
                else:
                    values.append("--")
            latex_rows.append(f"        {approach} & " + " & ".join(values) + r" \\")

        headers = "\n        & ".join([format_header(ds) for ds in dataset_list])
        col_spec = "l" + "c" * num_datasets

        return f"""    \\begin{{tabular}}{{{col_spec}}}
        \\toprule
        & {headers}\\\\
        \\midrule
{chr(10).join(latex_rows)}
        \\bottomrule
    \\end{{tabular}}"""

    subtable1 = build_subtable(datasets_part1)
    subtable2 = build_subtable(datasets_part2) if datasets_part2 else ""

    caption = table_caption
    label = table_label
    if forecaster_type:
        caption = f"{caption} ({forecaster_type})"
        label = f"{label}_{forecaster_type}"
    if metric_name:
        caption = f"{caption} - {metric_name}"
        label = f"{label}_{metric_name}"

    latex_table = f"""\\begin{{table}}[!htpb]
    \\scriptsize
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\centering
    \\vspace{{0.5em}}
{subtable1}"""

    if subtable2:
        latex_table += f"""

    \\vspace{{1em}}

{subtable2}"""

    latex_table += "\n\\end{table}"

    os.makedirs(output_path, exist_ok=True)
    full_path = f"{output_path}/{file_name}.tex"
    with open(full_path, "w") as f:
        f.write(latex_table)
    print(f"Saved to {full_path}")


def get_fixed_cut_latex_tables(
    individual_metrics_dict: dict,
    col_prefix: str,
    file_name: str,
    table_caption: str,
    table_label: str,
    output_path: str,
    forecaster_type: str = None,
    higher_is_better: bool = False
) -> None:
    """Generate LaTeX table for Fixed Cut baseline results.

    Args:
        individual_metrics_dict: Dictionary mapping dataset names to results DataFrames
        col_prefix: Prefix for the metric columns (e.g., 'Avg_RMSE')
        file_name: Base filename for the output
        table_caption: Caption for the LaTeX table
        table_label: Label for the LaTeX table
        output_path: Directory to save the table
        forecaster_type: Optional forecaster type to append to caption
        higher_is_better: True if higher metric values are better
    """
    all_approaches = individual_metrics_dict[DATASETS_NAMES[0]][CHANGE_POINT_APPROACH_COL].tolist()

    # Get Fixed Cut approaches and sort by percentage
    fixed_cut_approaches = [a for a in all_approaches if a.startswith('Fixed Cut')]
    fixed_cut_approaches = sorted(fixed_cut_approaches, key=lambda x: float(x.replace('Fixed Cut ', '').replace('%', '')))

    # Include No CPD as baseline
    approaches = ['No CPD'] + fixed_cut_approaches

    mean_col, std_col = f"{col_prefix}_mean", f"{col_prefix}_std"

    # Get baseline (No CPD) values
    baseline_values = {}
    for ds in individual_metrics_dict:
        df = individual_metrics_dict[ds]
        baseline_row = df[df[CHANGE_POINT_APPROACH_COL] == 'No CPD']
        baseline_values[ds] = baseline_row[mean_col].iloc[0] if len(baseline_row) > 0 else None

    # Get best values among Fixed Cut approaches only
    best_values = {}
    for ds in individual_metrics_dict:
        df = individual_metrics_dict[ds]
        filtered_df = df[df[CHANGE_POINT_APPROACH_COL].isin(approaches)]
        if higher_is_better:
            best_values[ds] = filtered_df[mean_col].max()
        else:
            best_values[ds] = filtered_df[mean_col].min()

    datasets_part1 = DATASETS_NAMES[:3]
    datasets_part2 = DATASETS_NAMES[3:]

    def format_header(name):
        display_name = DATASETS_DISPLAY_NAMES.get(name, name)
        parts = display_name.split()
        if len(parts) > 1:
            return r"\makecell{\textbf{" + parts[0] + r"}\\\textbf{" + " ".join(parts[1:]) + r"}}"
        return r"\textbf{" + display_name + r"}"

    def format_value(val, width=5):
        return f"{val:0{width}.2f}"

    def get_comparison_str(current_val, baseline_val, higher_is_better):
        if baseline_val is None or baseline_val == 0:
            return ""
        pct_change = ((current_val - baseline_val) / abs(baseline_val)) * 100
        if higher_is_better:
            if pct_change > 0:
                return f" ($\\uparrow${format_value(abs(pct_change))}\\%)"
            elif pct_change < 0:
                return f" ($\\downarrow${format_value(abs(pct_change))}\\%)"
        else:
            if pct_change < 0:
                return f" ($\\downarrow${format_value(abs(pct_change))}\\%)"
            elif pct_change > 0:
                return f" ($\\uparrow${format_value(abs(pct_change))}\\%)"
        return ""

    def build_subtable(dataset_list):
        num_datasets = len(dataset_list)
        latex_rows = []

        for approach in approaches:
            values = []
            for dataset in dataset_list:
                df = individual_metrics_dict[dataset]
                row = df[df[CHANGE_POINT_APPROACH_COL] == approach]
                if len(row) > 0:
                    mean_val = row[mean_col].iloc[0]
                    std_val = row[std_col].iloc[0]
                    value_str = f"{format_value(mean_val)} $\\pm$ {format_value(std_val)}"

                    if approach != 'No CPD':
                        comparison = get_comparison_str(mean_val, baseline_values[dataset], higher_is_better)
                        value_str = value_str + comparison

                    if abs(mean_val - best_values[dataset]) < 1e-9:
                        value_str = f"\\textbf{{{value_str}}}"
                    values.append(value_str)
                else:
                    values.append("--")
            # Escape % in approach name for LaTeX
            approach_latex = approach.replace('%', '\\%')
            latex_rows.append(f"        {approach_latex} & " + " & ".join(values) + r" \\")
            # Add midrule after No CPD to separate from Fixed Cut approaches
            if approach == 'No CPD':
                latex_rows.append(r"        \midrule")

        headers = "\n        & ".join([format_header(ds) for ds in dataset_list])
        col_spec = "l" + "c" * num_datasets

        return f"""    \\begin{{tabular}}{{{col_spec}}}
        \\toprule
        & {headers}\\\\
        \\midrule
{chr(10).join(latex_rows)}
        \\bottomrule
    \\end{{tabular}}"""

    subtable1 = build_subtable(datasets_part1)
    subtable2 = build_subtable(datasets_part2) if datasets_part2 else ""

    caption = table_caption
    label = table_label
    if forecaster_type:
        caption = f"{caption} ({forecaster_type})"
        label = f"{label}_{forecaster_type}"

    latex_table = f"""\\begin{{table}}[!htpb]
    \\scriptsize
    \\caption{{{caption}}}
    \\label{{{label}}}
    \\centering
    \\vspace{{0.5em}}
{subtable1}"""

    if subtable2:
        latex_table += f"""

    \\vspace{{1em}}

{subtable2}"""

    latex_table += "\n\\end{table}"

    os.makedirs(output_path, exist_ok=True)
    full_path = f"{output_path}/{file_name}.tex"
    with open(full_path, "w") as f:
        f.write(latex_table)
    print(f"Saved to {full_path}")


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def display_with_latex(df: pd.DataFrame, caption: str = "", label: str = "", **kwargs):
    """Display DataFrame in Jupyter notebook.

    Args:
        df: DataFrame to display
        caption: Table caption (unused, kept for API compatibility)
        label: Table label (unused, kept for API compatibility)
        **kwargs: Additional arguments (unused, kept for API compatibility)
    """
    from IPython.display import display
    display(df)


def prepare_comparison_df(
    individual_metrics_dict: dict,
    metric_col: str = RMSE_COL
) -> pd.DataFrame:
    """Prepare a comparison DataFrame combining all datasets.

    Args:
        individual_metrics_dict: Dictionary mapping dataset names to results DataFrames
        metric_col: Column name for the primary metric

    Returns:
        Combined DataFrame with all datasets
    """
    comp_df = pd.concat([
        individual_metrics_dict[ds][[
            CHANGE_POINT_APPROACH_COL,
            f"{CHANGE_POINT_PERC_COL}_seed_{SEEDS[0]}",
            f"{metric_col}_mean",
            f"{ET_TOTAL_COL}_mean"
        ]].assign(**{DATASET_NAME_COL: ds})
        for ds in DATASETS_NAMES
    ], axis=0)
    return comp_df


def save_results_to_excel(
    df: pd.DataFrame,
    output_folder: str,
    filename: str
) -> None:
    """Save DataFrame to Excel file.

    Args:
        df: DataFrame to save
        output_folder: Directory to save the file
        filename: Filename (without extension)
    """
    os.makedirs(output_folder, exist_ok=True)
    filepath = f"{output_folder}/{filename}.xlsx"
    df.to_excel(filepath, index=False)
    print(f"Saved to {filepath}")
