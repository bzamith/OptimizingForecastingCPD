DATE_COLUMN = 'ds'

TRAIN_PERC = 0.8
NB_TRIALS = 20

FORECASTER_LOSS = 'mean_squared_error'
FORECAST_HORIZON = 7
OBSERVATION_WINDOW = 14
EARLY_STOPPING_PATIENCE = 10
NB_EPOCHS = 150

# Common hyperparameter search spaces
HP_DROPOUT_RATES = [0.1, 0.2, 0.3]
HP_LEARNING_RATES = [1e-3, 5e-4, 1e-4]
HP_MODEL_DIMS = [32, 64, 128]
HP_L2_REG = [1e-4, 1e-3]

# Rolling window validation settings
ROLLING_WINDOW_N_SPLITS = 5
ROLLING_WINDOW_TEST_SIZE = 0.1  # Each test fold is 10% of total data