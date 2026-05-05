#!/bin/bash

# Rolling window validation experiment script for Pareto and Weighted Sum approaches

LOG_DIR="outputs/rolling_experiment_logs"
mkdir -p "$LOG_DIR"

# Activate poetry virtualenv
echo "Activating poetry virtual environment..."
VENV_PATH=$(poetry env info --path)
source "$VENV_PATH/bin/activate"
echo "Using Python: $(which python)"
echo ""

# Single seed - configurable via argument or default to 42
SEED=${1:-42}
N_SPLITS=${2:-5}

DATASETS=("INMET SAOPAULO_SP" "UCI AIR_QUALITY" "UCI PRSA_BEIJING" "UCI APPLIANCES_ENERGY" "UCI METRO_TRAFFIC" "AUTOFORMER WEATHER")
FORECASTER_TYPES=("LSTM" "TCN")

# Define the Pareto & Weighted Sum approaches (Method|CostFunction)
PARETO_CONFIGS=(
  "Fixed_Perc|Fixed_Cut_0.0"
  "Fixed_Perc|Fixed_Cut_0.1"
  "Fixed_Perc|Fixed_Cut_0.4"
  "Fixed_Perc|Fixed_Cut_0.5"
  "Fixed_Perc|Fixed_Cut_0.6"
  "Fixed_Perc|Fixed_Cut_0.8"
  "Fixed_Perc|Fixed_Cut_0.9"
  "Window|Normal"
  "Window|RBF"
  "Bin_Seg|Normal"
  "Bottom_Up|L2"
)

total_jobs=0
skipped_jobs=0

# Function to check if experiment already completed
is_completed() {
  local seed="$1"
  local dataset_full="$2"
  local forecaster_type="$3"
  local cpd_method="$4"
  local cost_function="$5"
  local n_splits="$6"
  read -r dataset_domain dataset_name <<< "$dataset_full"

  local base_path="seed=$seed/dataset_domain=$dataset_domain/dataset_name=$dataset_name/cpd_method=$cpd_method/cpd_cost_function=$cost_function/forecaster_type=$forecaster_type"
  base_path=${base_path// /_}

  local report_dir="outputs/rolling_report/$base_path"
  
  for ts_dir in "$report_dir"/timestamp=*; do
    [ -d "$ts_dir" ] || continue
    local report_file="$ts_dir/report.json"
    if [ -f "$report_file" ] && grep -q '"aggregated_metrics_real_only"' "$report_file" 2>/dev/null; then
      local all_folds_exist=true
      for (( i=1; i<=n_splits; i++ )); do
        if [ ! -f "$ts_dir/fold_${i}_report.json" ]; then
          all_folds_exist=false
          break
        fi
      done
      if [ "$all_folds_exist" = true ]; then
        return 0
      fi
    fi
  done
  return 1
}

run_job() {
  local seed="$1"
  local dataset_full="$2"
  local forecaster_type="$3"
  local cpd_method="$4"
  local cost_function="$5"
  local n_splits="$6"
  read -r dataset_domain dataset_name <<< "$dataset_full"

  local log_dir="$LOG_DIR/seed=$seed/dataset_domain=$dataset_domain/dataset_name=$dataset_name/cpd_method=$cpd_method/cpd_cost_function=$cost_function/forecaster_type=$forecaster_type"
  log_dir=${log_dir// /_}
  local log_file="$log_dir/log.log"
  mkdir -p "$log_dir"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $dataset_domain $dataset_name | $cpd_method | $cost_function | $forecaster_type"

  {
    echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "Command: python main_rolling.py $dataset_domain $dataset_name $cpd_method $cost_function $forecaster_type $seed $n_splits"
    
    nice -n 10 \
        python -u main_rolling.py \
        "$dataset_domain" "$dataset_name" "$cpd_method" "$cost_function" "$forecaster_type" "$seed" "$n_splits"

    exit_code=$?
    echo "=== Job finished at $(date '+%Y-%m-%d %H:%M:%S') with exit code: $exit_code ==="
  } > "$log_file" 2>&1

  if [ $exit_code -eq 0 ]; then
    echo "  -> COMPLETED"
  else
    echo "  -> FAILED (see $log_file)"
  fi
}

echo "=========================================="
echo "Rolling Window Pareto & Weighted Sum"
echo "Seed: $SEED | Splits: $N_SPLITS"
echo "=========================================="
echo ""

for forecaster_type in "${FORECASTER_TYPES[@]}"; do
  for config in "${PARETO_CONFIGS[@]}"; do
    IFS="|" read -r cpd_method cost_function <<< "$config"
    for dataset in "${DATASETS[@]}"; do
      
      if is_completed "$SEED" "$dataset" "$forecaster_type" "$cpd_method" "$cost_function" "$N_SPLITS"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] SKIPPED: $dataset | $cpd_method | $cost_function | $forecaster_type"
        ((skipped_jobs++))
        continue
      fi

      run_job "$SEED" "$dataset" "$forecaster_type" "$cpd_method" "$cost_function" "$N_SPLITS"
      ((total_jobs++))
    done
  done
done

echo ""
echo "=========================================="
echo "Finished $total_jobs experiments"
echo "Skipped $skipped_jobs experiments"
echo "=========================================="
