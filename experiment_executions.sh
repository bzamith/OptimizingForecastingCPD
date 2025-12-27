#!/bin/bash

# Constants
CONFIG_FILE="config/constants.py"
SEEDS=(0 42 52 101 214 565 600 713 999 1001)
DATASETS=("AUTOFORMER WEATHER" "INMET BRASILIA_DF" "INMET VITORIA_ES" "INMET PORTOALEGRE_RS" "INMET SAOPAULO_SP" "UCI AIR_QUALITY" "UCI PRSA_BEIJING" "UCI APPLIANCES_ENERGY" "UCI METRO_TRAFFIC")
CPD_METHODS=("Window" "Bin_Seg" "Bottom_Up")
CPD_COST_FUNCTIONS=("L1" "L2" "Normal" "Linear" "Rank" "RBF" "AR")
CPD_FIXED_CUTS=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)
FORECASTER_TYPES=("LSTM" "Transformer" "SSM")

# Ensure config file exists
if [[ ! -f "$CONFIG_FILE" ]]; then
  echo "File $CONFIG_FILE does not exist."
  exit 1
fi

run_experiments() {
  local seed="$1"
  local dataset_full="$2"
  local forecaster_type="$3"

  # Split dataset into domain and dataset name
  read -r dataset_domain dataset_name <<< "$dataset_full"

  # Update SEED value in config file
  sed -i '' "1s/.*/SEED = $seed/" "$CONFIG_FILE"

  printf "Running experiments for dataset: %s %s, forecaster: %s, seed: %s\n" \
    "$dataset_domain" "$dataset_name" "$forecaster_type" "$seed"

  # Run experiments for each method and cost function
  for cpd_method in "${CPD_METHODS[@]}"; do
    for cost_function in "${CPD_COST_FUNCTIONS[@]}"; do
      printf "  Running: %s %s %s %s %s\n" \
        "$dataset_domain" "$dataset_name" "$cpd_method" "$cost_function" "$forecaster_type"
      nice -n -10 poetry run python main.py \
        "$dataset_domain" "$dataset_name" "$cpd_method" "$cost_function" "$forecaster_type"
    done
  done

  # Run experiments for fixed cut method
  for cpd_cut in "${CPD_FIXED_CUTS[@]}"; do
    printf "  Running: %s %s Fixed_Perc Fixed_Cut_%s %s\n" \
      "$dataset_domain" "$dataset_name" "$cpd_cut" "$forecaster_type"
    nice -n -10 poetry run python main.py \
      "$dataset_domain" "$dataset_name" "Fixed_Perc" "Fixed_Cut_$cpd_cut" "$forecaster_type"
  done
}

# Run experiments for each seed, dataset, and forecaster type
for seed in "${SEEDS[@]}"; do
  for dataset in "${DATASETS[@]}"; do
    for forecaster_type in "${FORECASTER_TYPES[@]}"; do
      run_experiments "$seed" "$dataset" "$forecaster_type"
    done
  done
done
