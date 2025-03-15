#!/bin/bash

# Constants
CONFIG_FILE="config/constants.py"
RUN_SCRIPT="./run.sh"
SEEDS=(0 42 52 713 1001)
DATASETS=("UCI AIR_QUALITY" "UCI APPLIANCES_ENERGY" "UCI METRO_TRAFFIC" "UCI PRSA_BEIJING")
METHODS=("Window" "Bin_Seg" "Bottom_Up")
COST_FUNCTIONS=("L1" "L2" "Normal" "RBF" "Linear" "Rank" "AR")
FIXED_CUTS=(0.0 0.1 0.2 0.3 0.4 0.6 0.7 0.8 0.9)


# Ensure config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "File $CONFIG_FILE does not exist."
  exit 1
fi

# Ensure run script exists
if [[ ! -x "$RUN_SCRIPT" ]]; then
  printf "Error: Script %s not found or not executable.\n" "$RUN_SCRIPT"
  exit 1
fi

# Function to execute commands
run_experiments() {
  local dataset="$1"
  local seed="$2"

  # Update SEED value in config file
  sed -i '' "1s/.*/SEED = $seed/" "$CONFIG_FILE"

  printf "Running experiments for dataset: %s, seed: %s\n" "$dataset" "$seed"

  # Run experiments for each method and cost function
  for method in "${METHODS[@]}"; do
    for cost_function in "${COST_FUNCTIONS[@]}"; do
      nice -n -10 "$RUN_SCRIPT" execute "$dataset" "$method" "$cost_function"
    done
  done

  # Run experiments for fixed method
  for cut in "${FIXED_CUTS[@]}"; do
    nice -n -10 "$RUN_SCRIPT" execute "$dataset" Fixed_Perc "Fixed_Cut_$cut"
  done
}

# Run experiments for each dataset and seed
for dataset in "${DATASETS[@]}"; do
  for seed in "${SEEDS[@]}"; do
    run_experiments "$dataset" "$seed"
  done
done
