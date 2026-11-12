import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(r"c:\Users\bhuky\Desktop\vegetable-price-prediction")
DATA_RAW_DIR = BASE_DIR / "Data"
DATA_PROCESSED_DIR = BASE_DIR / "Data" / "processed"
PAPERS_DIR = BASE_DIR / "papers"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"
RESULTS_DIR = BASE_DIR / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
PREDICTIONS_DIR = RESULTS_DIR / "predictions"
TABLES_DIR = RESULTS_DIR / "tables"
PLOTS_DIR = RESULTS_DIR / "plots"
EDA_PLOTS_DIR = PLOTS_DIR / "eda"
HMM_PLOTS_DIR = PLOTS_DIR / "hmm"
PRED_PLOTS_DIR = PLOTS_DIR / "predictions"
COMP_PLOTS_DIR = PLOTS_DIR / "comparison"
ABLATION_PLOTS_DIR = PLOTS_DIR / "ablation"

# Commodities & Files
COMMODITY_FILES = {
    "Tomato": "Tomato_weekly_price_weather.xlsx",
    "Onion": "Onion_weekly_price_weather.xlsx",
    "Potato": "Potato_weekly_price_weather.xlsx"
}

# Reproducibility
RANDOM_SEED = 42

# Forecast Horizons (weeks ahead)
HORIZONS = [1, 4, 8, 12]

# Chronological Data Split Ratios
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Sequence Model Settings
SEQUENCE_LENGTH = 12  # Lookback window: 12 weeks (~3 months)
BATCH_SIZE = 32
EPOCHS = 80
LEARNING_RATE = 0.001
EARLY_STOPPING_PATIENCE = 12

# Feature Engineering Definitions
PRICE_LAGS = [1, 2, 3, 4, 8, 12, 24, 52]
WEATHER_TEMP_LAGS = [1, 2, 4, 8]
WEATHER_PRECIP_LAGS = [1, 2, 4, 8]
WEATHER_ROLLING_WINDOWS = [4, 8]

# HMM States to test
HMM_CANDIDATE_STATES = [2, 3, 4, 5, 6]

# Ablation Levels
ABLATION_LEVELS = {
    "A": "Price Only + GRU",
    "B": "Price + Weather + GRU",
    "C": "Price + Technical Indicators + GRU",
    "D": "Price + Weather + Technical Indicators + GRU",
    "E": "Price + Weather + Technical Indicators + HMM + GRU"
}
