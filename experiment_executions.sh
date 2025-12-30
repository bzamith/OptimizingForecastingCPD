#!/bin/bash

export OMP_NUM_THREADS=1
export TF_NUM_INTRAOP_THREADS=1
export TF_NUM_INTEROP_THREADS=1

MAX_JOBS=4
LOG_DIR="outputs/experiment_logs"
mkdir -p "$LOG_DIR"

# Activate poetry virtualenv once at the start to avoid lock contention
echo "Activating poetry virtual environment..."
VENV_PATH=$(poetry env info --path)
source "$VENV_PATH/bin/activate"
echo "Using Python: $(which python)"
echo ""

# SEEDS=(0 42 52 101 214 565 600 713 999 1001)
DATASETS=("UCI AIR_QUALITY" "UCI PRSA_BEIJING" "UCI APPLIANCES_ENERGY" "UCI METRO_TRAFFIC" \
          "INMET SAOPAULO_SP" "AUTOFORMER WEATHER")
CPD_METHODS=("Window" "Bin_Seg" "Bottom_Up")
CPD_COST_FUNCTIONS=("L1" "L2" "Normal" "Linear" "Rank" "RBF" "AR")
CPD_FIXED_CUTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
FORECASTER_TYPES=("LSTM" "TCN" "GRU")

total_jobs=0
declare -a job_pids=()

run_job() {
  local dataset_full="$1"
  local forecaster_type="$2"
  local cpd_method="$3"
  local cost_function="$4"
  read -r dataset_domain dataset_name <<< "$dataset_full"
  
  local log_name="${dataset_domain}_${dataset_name}_${cpd_method}_${cost_function}_${forecaster_type}"
  log_name=$(echo "$log_name" | tr ' ' '_' | tr '/' '_')
  local log_file="$LOG_DIR/${log_name}.log"
  
  # Write start message directly to terminal AND log
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] Starting: $dataset_domain $dataset_name | $cpd_method | $cost_function | $forecaster_type" | tee -a "$log_file"
  
  # Run in background with explicit redirection
  {
    echo "=== Job started at $(date '+%Y-%m-%d %H:%M:%S') ==="
    echo "Dataset: $dataset_domain $dataset_name"
    echo "Method: $cpd_method | Cost: $cost_function | Forecaster: $forecaster_type"
    echo "Command: python main.py $dataset_domain $dataset_name $cpd_method $cost_function $forecaster_type"
    echo "=== Output ==="
    
    nice -n -10 python main.py \
      "$dataset_domain" "$dataset_name" "$cpd_method" "$cost_function" "$forecaster_type"
    
    exit_code=$?
    echo "=== Job finished at $(date '+%Y-%m-%d %H:%M:%S') with exit code: $exit_code ==="
    
    if [ $exit_code -eq 0 ]; then
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✓ COMPLETED: $dataset_domain $dataset_name | $cpd_method | $cost_function | $forecaster_type"
    else
      echo "[$(date '+%Y-%m-%d %H:%M:%S')] ✗ FAILED (exit code $exit_code): $dataset_domain $dataset_name | $cpd_method | $cost_function | $forecaster_type"
    fi
  } >> "$log_file" 2>&1 &
  
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

for dataset in "${DATASETS[@]}"; do
  for forecaster_type in "${FORECASTER_TYPES[@]}"; do
    for cpd_method in "${CPD_METHODS[@]}"; do
      for cost_function in "${CPD_COST_FUNCTIONS[@]}"; do
        # Wait for an available slot before launching
        while [[ $(count_running_jobs) -ge $MAX_JOBS ]]; do
          sleep 0.5
          cleanup_finished_jobs
        done

        run_job "$dataset" "$forecaster_type" "$cpd_method" "$cost_function"
        ((total_jobs++))

        # Show current job count after launching
        current_jobs=$(count_running_jobs)
        echo "  → Active jobs: $current_jobs / $MAX_JOBS"
      done
    done

    for cpd_cut in "${CPD_FIXED_CUTS[@]}"; do
      # Wait for an available slot before launching
      while [[ $(count_running_jobs) -ge $MAX_JOBS ]]; do
        sleep 0.5
        cleanup_finished_jobs
      done

      run_job "$dataset" "$forecaster_type" "Fixed_Perc" "Fixed_Cut_$cpd_cut"
      ((total_jobs++))

      # Show current job count after launching
      current_jobs=$(count_running_jobs)
      echo "  → Active jobs: $current_jobs / $MAX_JOBS"
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
echo "Monitor progress with:"
echo "  watch -n 5 'ls $LOG_DIR/*.log | wc -l'"
echo "  tail -f $LOG_DIR/*.log"
echo "  grep -r \"COMPLETED\\|FAILED\" $LOG_DIR/"
echo ""

wait
echo ""
echo "All experiments finished!"
echo "Total: $total_jobs experiments"