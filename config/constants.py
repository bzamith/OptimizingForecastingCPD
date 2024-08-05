SEED = 42

FIXED_CUTS_PERCS = [round(i * 0.1, 1) for i in range(8)]

CUT_POINT_METHODS = [
    "L1",
    "L2",
    "Normal",
    "RBF",
    "Cosine",
    "Linear",
    "Clinear",
    "Rank",
    "Mahalanobis",
    "AR"
]
DATE_COLUMN = "Date"

FORECASTER_OBJECTIVE = 'val_loss'
MODEL_TYPE = 'LSTM'
NB_TRIALS = 20
OBSERVATION_WINDOW = 14
TRAIN_PERC = 0.8
