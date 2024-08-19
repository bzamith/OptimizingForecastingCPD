# Time Series Forecasting Subsets Study

## About

- Authors: Bruna Zamith Santos, Maira Farias de Andrade Lira
- Supervisors: Ricardo Cerri, Ricardo Prudêncio

## Install

```
python3 -m virtualenv -p /usr/bin/python3 env
source env/bin/activate
pip3 install -r requirements.txt
deactivate
```

## Data

You may check the pre-define dataset domains in `/src/dataset/`
In case you want to add a custom one, you'll have to update that file

All datasets must be time series (uni or multivariate), and contain the `Date` column.

## Run

```
source env/bin/activate
chmod +x run.sh

# Don't forget to activate the virtualenv, if you are using one!
# First, build the code:
./run.sh build

# Then, to run for Apple dataset (TCPD domain) and Window L1 cut point approach:
./run.sh execute TCPD APPLE Window L1
```
