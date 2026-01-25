"""
Configuration constants for metrics analysis notebooks.
"""
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.constants import TRAIN_PERC

# =============================================================================
# PATHS
# =============================================================================
# Base project directory (parent of metrics_for_paper)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Input folder with the reports
INPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs/report")

# Output folder for saving results
OUTPUT_FOLDER = os.path.join(PROJECT_ROOT, "outputs/paper_v2")

# Datasets folder
DATASETS_FOLDER = os.path.join(PROJECT_ROOT, "datasets")

# =============================================================================
# RANDOM SEEDS
# =============================================================================
SEEDS = [0, 42, 52, 101, 214, 565, 600, 713, 999, 1001]

# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
# Unified dataset configuration (ordered for paper tables)
DATASETS_CONFIG = {
    "UCI_AIR_QUALITY": {
        "database": "UCI",
        "dataset": "AIR_QUALITY",
        "display_name": "Air Quality",
        "csv_path": os.path.join(DATASETS_FOLDER, "uci/air_quality.csv"),
        "keep_cols": ["ds", "CO(GT)", "C6H6(GT)", "NOx(GT)", "NO2(GT)", "T", "RH"],
    },
    "UCI_APPLIANCES_ENERGY": {
        "database": "UCI",
        "dataset": "APPLIANCES_ENERGY",
        "display_name": "Appliances Energy",
        "csv_path": os.path.join(DATASETS_FOLDER, "uci/appliances_energy.csv"),
        "keep_cols": ["ds", "T_out", "Press_mm_hg", "RH_out", "Windspeed", "Visibility", "Tdewpoint"],
    },
    "UCI_PRSA_BEIJING": {
        "database": "UCI",
        "dataset": "PRSA_BEIJING",
        "display_name": "Beijing PM2.5",
        "csv_path": os.path.join(DATASETS_FOLDER, "uci/prsa_beijing.csv"),
        "keep_cols": ["ds", "pm2_5", "DEWP", "TEMP", "PRES", "Iws", "Is", "Ir"],
    },
    "INMET_SAOPAULO_SP": {
        "database": "INMET",
        "dataset": "SAOPAULO_SP",
        "display_name": "INMET São Paulo",
        "csv_path": os.path.join(DATASETS_FOLDER, "inmet/A701_SaoPaulo_SP.csv"),
        "keep_cols": ["ds", "P", "PrA", "T", "UR", "VV"],
    },
    "UCI_METRO_TRAFFIC": {
        "database": "UCI",
        "dataset": "METRO_TRAFFIC",
        "display_name": "Metro Traffic",
        "csv_path": os.path.join(DATASETS_FOLDER, "uci/metro_traffic.csv"),
        "keep_cols": ["ds", "temp", "rain_1h", "clouds_all"],
    },
    "AUTOFORMER_WEATHER": {
        "database": "AUTOFORMER",
        "dataset": "WEATHER",
        "display_name": "Weather",
        "csv_path": os.path.join(DATASETS_FOLDER, "autoformer/weather.csv"),
        "keep_cols": ["p (mbar)", "T (degC)", "rh (%)", "VPact (mbar)", "rho (g/m**3)", "wv (m/s)", "rain (mm)", "SWDR (W/m**2)"],
    },
}
DATASETS_NAMES = list(DATASETS_CONFIG.keys())

# Backward compatibility aliases (derived from unified config)
DATASETS_DISPLAY_NAMES = {k: v["display_name"] for k, v in DATASETS_CONFIG.items()}
DATASET_CSV_CONFIG = {k: {"csv_path": v["csv_path"], "keep_cols": v["keep_cols"]} for k, v in DATASETS_CONFIG.items()}

# =============================================================================
# COLUMN NAMES
# =============================================================================
# Error metrics
RMSE_COL = "Avg_RMSE"
R2_COL = "Avg_R2"
CHANGE_POINT_PERC_COL = "change_point_perc"
CHANGE_POINT_APPROACH_COL = "change_point_approach"
MODEL_TYPE_COL = "forecaster_type"
DATASET_NAME_COL = "dataset_name"

# Execution time columns (all in minutes)
ET_CPD_COL = "ET_CPD"  # detect_change_point_perf_duration
ET_HPO_COL = "ET_HPO"  # tuner_process_duration
ET_RETRAIN_COL = "ET_Retrain"  # retrain_perf_duration
ET_TOTAL_COL = "ET_Total"  # sum of CPD + HPO + Retrain

# =============================================================================
# ANALYSIS CONFIGURATION
# =============================================================================
# Approaches to exclude from ranking analysis
FIXED_APPROACHES = [f'Fixed Cut {i}0%' for i in range(1, 10)]

# =============================================================================
# METRIC CONFIGURATION FOR MULTI-OBJECTIVE OPTIMIZATION
# =============================================================================
# Metric properties configuration
# - "minimize": True if lower values are better (e.g., RMSE, MAE)
# - "minimize": False if higher values are better (e.g., R2, Accuracy)
METRIC_CONFIG = {
    RMSE_COL: {"minimize": True, "display_name": "RMSE", "format": ".4f"},
    R2_COL: {"minimize": False, "display_name": "R2", "format": ".4f"},
    ET_TOTAL_COL: {"minimize": True, "display_name": "Execution Time (min)", "format": ".2f"},
}

# Re-export TRAIN_PERC for convenience
__all__ = [
    'PROJECT_ROOT', 'INPUT_FOLDER', 'OUTPUT_FOLDER', 'DATASETS_FOLDER',
    'SEEDS', 'DATASETS_CONFIG', 'DATASETS_NAMES', 'DATASETS_DISPLAY_NAMES',
    'DATASET_CSV_CONFIG', 'RMSE_COL', 'R2_COL', 'CHANGE_POINT_PERC_COL',
    'CHANGE_POINT_APPROACH_COL', 'MODEL_TYPE_COL', 'DATASET_NAME_COL',
    'ET_CPD_COL', 'ET_HPO_COL', 'ET_RETRAIN_COL', 'ET_TOTAL_COL',
    'FIXED_APPROACHES', 'METRIC_CONFIG', 'TRAIN_PERC'
]
