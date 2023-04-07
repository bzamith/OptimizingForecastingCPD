#!/bin/bash
set -e

BASE_DIR=$(dirname $0)
FLAKE8_CONFIG="${BASE_DIR}/config/.flake8"
PYTHONPATH="${BASE_DIR}"

run_flake8()
{
  flake8 . --config=$FLAKE8_CONFIG
}

if [ "$1" == "build" ]; then
  mkdir -p outputs/
  echo ">>>>>>>>>>>> [1/1] Running flake8"
  flake8 . --config=$FLAKE8_CONFIG
elif [ "$1" == "execute" ]; then
  mkdir -p outputs/
  echo ">>>>>>>>>>>> [1/1] Executing"
  TF_CPP_MIN_LOG_LEVEL=3 python3 main.py $2 $3
elif [ "$1" == "all" ]; then
  mkdir -p outputs/
  echo ">>>>>>>>>>>> [1/2] Running flake8"
  flake8 . --config=$FLAKE8_CONFIG
  echo ">>>>>>>>>>>> [2/2] Executing"
  TF_CPP_MIN_LOG_LEVEL=3 python3 main.py $2 $3
else
  echo ">>>>>>>>>>>> Option \"$1\" not found. Please try again with one of the following options:"
  echo "- build - execute - all"
fi