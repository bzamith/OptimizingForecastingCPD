#!/bin/bash
set -e

BASE_DIR=$(dirname $0)
FLAKE8_CONFIG="${BASE_DIR}/config/.flake8"
PYTHONPATH="${BASE_DIR}"

run_directories_check()
{
  mkdir -p outputs/
  mkdir -p outputs/EMBRAPA/
  mkdir -p outputs/INMET/
  mkdir -p outputs/UCI/
  mkdir -p outputs/TCPD/
}

run_flake8()
{
  flake8 . --config=$FLAKE8_CONFIG
}

if [ "$1" == "build" ]; then
  echo ">>>>>>>>>>>> [1/2] Running directories check"
  run_directories_check
  echo ">>>>>>>>>>>> [2/2] Running flake8"
  run_flake8
elif [ "$1" == "execute" ]; then
  echo ">>>>>>>>>>>> [1/2] Running directories check"
  run_directories_check
  echo ">>>>>>>>>>>> [2/2] Executing"
  TF_CPP_MIN_LOG_LEVEL=3 python3 main.py $2 $3 $4
elif [ "$1" == "all" ]; then
  echo ">>>>>>>>>>>> [1/3] Running directories check"
  run_directories_check
  echo ">>>>>>>>>>>> [2/3] Running flake8"
  run_flake8
  echo ">>>>>>>>>>>> [3/3] Executing"
  TF_CPP_MIN_LOG_LEVEL=3 python3 main.py $2 $3 $4
else
  echo ">>>>>>>>>>>> Option \"$1\" not found. Please try again with one of the following options:"
  echo "- build - execute - all"
fi