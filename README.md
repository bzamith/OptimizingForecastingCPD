# Optimizing Climate Forecasting via Reduced Time Series from Change Point Detection Methods

**Authors:** Bruna Zamith Santos, Maira Farias de Andrade Lira
**Supervisors:** Ricardo Cerri, Ricardo Prudêncio

## Installation

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry install --with dev
```

## Usage

```bash
# Basic execution
poetry run python main.py <DATASET_DOMAIN> <DATASET> <CPD_METHOD> <COST_FUNCTION> <MODEL>

# Example
poetry run python main.py TCPD APPLE Window L1 lstm
```

**Available models:** `LSTM`, `Transformer`, `TCN`
**CPD methods:** `Window`, `Bin_Seg`, `Bottom_Up`, `Fixed_Perc`
**Dataset domains:** See [src/data_reader/](src/data_reader/)

## Development Commands

```bash
make install-dev    # Install dependencies
make format         # Format code
make lint           # Run linter
make test           # Run tests
make build          # Format, lint and test
make run            # Run example
make clean          # Clean artifacts
```
