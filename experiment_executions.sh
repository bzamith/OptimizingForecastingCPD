#!/bin/bash

# Constants
CONFIG_FILE="config/constants.py"
DATASETS=("UCI AIR_QUALITY" "UCI APPLIANCES_ENERGY" "UCI METRO_TRAFFIC" "UCI PRSA_BEIJING")
SEEDS=(0 42 1001)
METHODS=("Window" "Bin_Seg" "Bottom_Up")
METRICS=("L1" "L2" "Normal" "RBF" "Linear" "Rank" "AR")

# Ensure config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "File $CONFIG_FILE does not exist."
  exit 1
fi

# Function to execute commands
run_experiments() {
  local dataset=$1
  local seed=$2

  # Update SEED value in config file
  sed -i '' "1s/.*/SEED = $seed/" "$CONFIG_FILE"

  # Run experiments for each method and metric
  for method in "${METHODS[@]}"; do
    for metric in "${METRICS[@]}"; do
      nice -n -10 ./run.sh execute "$dataset" "$method" "$metric"
    done
  done

  # Run experiments for fixed method
  nice -n -10 ./run.sh execute "$dataset" Fixed_Perc Fixed_Cut_0.0
}

# Run experiments for each dataset and seed
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_experiments "$dataset" "$seed"
  done
done
