#!/bin/bash

export OMP_NUM_THREADS=4
export TF_NUM_INTRAOP_THREADS=4
export TF_NUM_INTEROP_THREADS=1
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

MAX_JOBS=4
LOG_DIR="outputs/experiment_logs"
mkdir -p "$LOG_DIR"

# Activate poetry virtualenv once at the start to avoid lock contention
echo "Activating poetry virtual environment..."
VENV_PATH=$(poetry env info --path)
source "$VENV_PATH/bin/activate"
echo "Using Python: $(which python)"
echo ""

SEEDS=(0 42 52 101 214 565 600 713 999 1001)
DATASETS=("INMET SAOPAULO_SP" "UCI AIR_QUALITY" "UCI PRSA_BEIJING" "UCI APPLIANCES_ENERGY" "UCI METRO_TRAFFIC" "AUTOFORMER WEATHER")
CPD_METHODS=("Window" "Bin_Seg" "Bottom_Up")
CPD_COST_FUNCTIONS=("L1" "L2" "Normal" "Linear" "Rank" "RBF" "AR")
CPD_FIXED_CUTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
FORECASTER_TYPES=("LSTM" "TCN" "ARIMA")

total_jobs=0
skipped_jobs=0
declare -a job_pids=()

# Function to check if experiment already completed
is_completed() {
  local seed="$1"
  local dataset_full="$2"
  local forecaster_type="$3"
  local cpd_method="$4"
  local cost_function="$5"
  read -r dataset_domain dataset_name <<< "$dataset_full"

  local log_dir="$LOG_DIR/seed=$seed/dataset_domain=$dataset_domain/dataset_name=$dataset_name/cpd_method=$cpd_method/cpd_cost_function=$cost_function/forecaster_type=$forecaster_type"
  log_dir=${log_dir// /_}
  local log_file="$log_dir/log.log"

  # Check if log file exists and contains COMPLETED marker
  if [ -f "$log_file" ] && grep -q "COMPLETED" "$log_file" 2>/dev/null; then
    return 0  # True - already completed
  else
    return 1  # False - not completed
  fi
}

run_job() {
  local seed="$1"
  local dataset_full="$2"
  local forecaster_type="$3"
  local cpd_method="$4"
  local cost_function="$5"
  read -r dataset_domain dataset_name <<< "$dataset_full"

  mkdir -p "$LOG_DIR"

  local log_dir="$LOG_DIR/seed=$seed/dataset_domain=$dataset_domain/dataset_name=$dataset_name/cpd_method=$cpd_method/cpd_cost_function=$cost_function/forecaster_type=$forecaster_type"
  log_dir=${log_dir// /_}
  local log_file="$log_dir/log.log"
  mkdir -p "$log_dir"

  # Clean existing log file before starting
  rm -f "$log_file"

  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting [seed=$seed]: \
  $dataset_domain $dataset_name | $cpd_method | $cost_function | $forecaster_type" \
  | tee -a "$log_file"
  # Run in background with explicit redirection
  {
    echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "Seed: $seed"
    echo "Dataset: $dataset_domain $dataset_name"
    echo "Method: $cpd_method | Cost: $cost_function | Forecaster: $forecaster_type"
    echo "Command: python main.py $dataset_domain $dataset_name $cpd_method $cost_function $forecaster_type $seed"
    echo "=== Output ==="

    nice -n -10 python main.py \
      "$dataset_domain" "$dataset_name" "$cpd_method" "$cost_function" "$forecaster_type" "$seed"

    exit_code=$?
    echo "=== Job finished at $(date '+%Y-%m-%d %H:%M:%S') with exit code: $exit_code ==="

    if [ $exit_code -eq 0 ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ COMPLETED [seed=$seed]: $dataset_domain $dataset_name | $cpd_method | $cost_function | $forecaster_type"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ FAILED (exit code $exit_code) [seed=$seed]: $dataset_domain $dataset_name | $cpd_method | $cost_function | $forecaster_type"
    fi

    # Keep only the last 10 lines of the log file (after job completes)
    tail -10 "$log_file" > "$log_file.tmp" && mv "$log_file.tmp" "$log_file"
  } >> "$log_file" 2>&1 &

  # Capture PID of the backgrounded job block immediately (not of later short-lived commands)
  local pid=$!
  job_pids+=($pid)
  echo "  → Backgrounded as PID $pid (log: $log_file)"
}

# Function to count running jobs by checking PIDs
count_running_jobs() {
  local count=0
  for pid in "${job_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      ((count++))
    fi
  done
  echo $count
}

# Function to clean up finished PIDs from array
cleanup_finished_jobs() {
  local new_pids=()
  for pid in "${job_pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      new_pids+=($pid)
    fi
  done
  job_pids=("${new_pids[@]}")
}

for forecaster_type in "${FORECASTER_TYPES[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for cpd_fixed_cut in "${CPD_FIXED_CUTS[@]}"; do
      for seed in "${SEEDS[@]}"; do
        # Check if already completed
        if is_completed "$seed" "$dataset" "$forecaster_type" "Fixed_Perc" "Fixed_Cut_$cpd_fixed_cut"; then
          echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⊙ SKIPPED (already completed) [seed=$seed]: $dataset | Fixed_Perc | Fixed_Cut_$cpd_fixed_cut | $forecaster_type"
          ((skipped_jobs++))
          continue
        fi

        # Wait for an available slot before launching
        while [[ $(count_running_jobs) -ge $MAX_JOBS ]]; do
          sleep 0.5
          cleanup_finished_jobs
        done

        run_job "$seed" "$dataset" "$forecaster_type" "Fixed_Perc" "Fixed_Cut_$cpd_fixed_cut"
        ((total_jobs++))

        # Show current job count after launching
        current_jobs=$(count_running_jobs)
        echo "  → Active jobs: $current_jobs / $MAX_JOBS"
      done
    done
  done
done

for forecaster_type in "${FORECASTER_TYPES[@]}"; do
  for cpd_method in "${CPD_METHODS[@]}"; do
    for cost_function in "${CPD_COST_FUNCTIONS[@]}"; do
      for dataset in "${DATASETS[@]}"; do
        for seed in "${SEEDS[@]}"; do
          # Check if already completed
          if is_completed "$seed" "$dataset" "$forecaster_type" "$cpd_method" "$cost_function"; then
            echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⊙ SKIPPED (already completed) [seed=$seed]: $dataset | $cpd_method | $cost_function | $forecaster_type"
            ((skipped_jobs++))
            continue
          fi

          # Wait for an available slot before launching
          while [[ $(count_running_jobs) -ge $MAX_JOBS ]]; do
            sleep 0.5
            cleanup_finished_jobs
          done

          run_job "$seed" "$dataset" "$forecaster_type" "$cpd_method" "$cost_function"
          ((total_jobs++))

          # Show current job count after launching
          current_jobs=$(count_running_jobs)
          echo "  → Active jobs: $current_jobs / $MAX_JOBS"
        done
      done
    done
  done
done

echo ""
echo "=========================================="
echo "Launched $total_jobs experiments"
echo "Running with MAX_JOBS=$MAX_JOBS in parallel"
echo "Logs saved to: $LOG_DIR/"
echo "=========================================="
echo ""

wait
echo ""
echo "All experiments finished!"
echo "Total: $total_jobs experiments"