"""
Configuration file for Day-Ahead Electricity Price Forecasting System

This file contains all configuration parameters for the EPF system.
Modify these settings according to your specific requirements.
"""

from datetime import datetime, timedelta
from pathlib import Path

# =============================================================================
# SYSTEM CONFIGURATION
# =============================================================================

# Directory paths
DATA_DIR = "data"
MODELS_DIR = "models" 
EVALUATION_DIR = "evaluation"
OUTPUT_DIR = "output"

# =============================================================================
# FORECASTING CONFIGURATION
# =============================================================================

# Zones to forecast (bidding zones)
ZONES = ['DE']

# Date configuration
STARTING_DATE = None  # Format: 'dd/mm/yyyy' or None for current date
DAYS_AHEAD = 1        # Number of days ahead to forecast (1 = next day)

# Model configuration
TRAINING_VERSION = 'v1'
TABLE_PRED = 'Prices'
START_DATE_TS = '2022-06-09'  # Start date for time series models
DAYS_BACK = 20                # Days back for evaluation

# Model types to include
INCLUDE_NAIVE_MODEL = False
INCLUDE_TIME_SERIES = True
INCLUDE_ML_MODELS = True
INCLUDE_DEEP_LEARNING = True

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================

# Machine Learning Models
ML_MODELS = [
    'LinearRegression', 'Lasso', 'Ridge', 'kNN', 'XGB', 'RandomForest',
    'GAM', 'SVM', 'GBM', 'AdaBoost', 'CatBoost', 'EXT', 'MLP'
]

# Deep Learning Models (Neural Networks)
DEEP_LEARNING_MODELS = [
    'LSTM', 'BILSTM', 'GRU', 'LSTM1', 'LSTM2', 'LSTM3', 'LSTM4',
    'GRU1', 'GRU2', 'GRU3', 'BILSTM1', 'BILSTM2', 'BILSTM3'
]

# Time Series Models
TIME_SERIES_MODELS = [
    'SARIMA_TS', 'ETS_TS', 'GBLA_TS', 'LSTM_TS', 'Ridge_TS', 'Lasso_TS',
    'GBT_TS', 'kNN_TS', 'MLP_TS', 'MLR_TS', 'RF_TS', 'SVR_TS', 'XBD_TS', 'LXBD_TS'
]

# Custom Ensemble Models
ENSEMBLE_MODELS = [
    'Best_Hourly', 'Best_Daily', 'Best_811', 'Best_721', 'Best_631', 'Best_532', 'Best_Average'
]

# =============================================================================
# DATA CONFIGURATION
# =============================================================================

# Country/Zone specific data configuration
COUNTRIES_CONFIG = {
    'v1': {
        'DE': {
            'Fundamentals_forecast': ['Load_DE', 'Wind_DE', 'Solar_DE', 'Hydro_DE', 'Nuc_DE'],
            'GasSpot': ['THE'],
            'EUA': ['EUA_DecLastPrice'],
            'Prices': ['DE_LU']
        }
        # Add more zones/countries here
        # 'FR': { ... },
        # 'HU': { ... }
    }
}

# Data granularity specification
DAILY_GRANULARITY_TABLES = ['GasSpot', 'EUA', 'GasFutures']

# =============================================================================
# EVALUATION CONFIGURATION
# =============================================================================

# Evaluation metrics to calculate
EVALUATION_METRICS = ['MAE', 'RMSE', 'MAPE', 'sMAPE', 'NegPos']

# Plotting configuration
CREATE_PLOTS = False        # Set to True to generate evaluation plots
PLOT_DPI = 100             # Resolution for saved plots
PLOT_FORMAT = 'png'        # Format for saved plots

# Plot themes (requires jupyterthemes)
PLOT_THEMES = {
    'hourly': 'chesterish',
    'table': 'gruvboxl', 
    'evaluation': 'chesterish',
    'residuals': 'grade3'
}

# =============================================================================
# TEAMS INTEGRATION
# =============================================================================

# Teams notification configuration
TEAMS_NOTIFICATION = False
TEAMS_WEBHOOK_URL = None  # Add your Teams webhook URL here

# Teams message configuration
TEAMS_MESSAGE_COLOR = "#bfbf9f"
TEAMS_MESSAGE_TITLE_TEMPLATE = "{date}, {days} {day_word} ahead price predictions are ready!"

# =============================================================================
# ADVANCED CONFIGURATION
# =============================================================================

# Neural network specific settings
NN_CONFIG = {
    'time_steps': 1,
    'batch_size': 32,
    'epochs': 20,
    'validation_split': 0.2,
    'verbose': 0
}

# Time series model parameters
TS_CONFIG = {
    'SARIMA': {
        'order': (3, 0, 3),
        'seasonal_order': (2, 1, 0, 7),
        'enforce_stationarity': False,
        'enforce_invertibility': False
    },
    'ETS': {
        'trend': 'additive',
        'seasonal': 'additive', 
        'seasonal_periods': 12,
        'damped': True
    },
    'LSTM_TS': {
        'lags': 24,
        'lstm_layer_sizes': (72, 72, 72, 72),
        'dropout': (0, 0, 0, 0),
        'activation': 'tanh',
        'optimizer': 'Adam',
        'learning_rate': 0.001
    }
}

# Ensemble weights for custom models
ENSEMBLE_WEIGHTS = {
    'Best_811': [0.8, 0.1, 0.1],
    'Best_721': [0.7, 0.2, 0.1], 
    'Best_631': [0.6, 0.3, 0.1],
    'Best_532': [0.5, 0.3, 0.2]
}

# =============================================================================
# FILE NAMING CONVENTIONS
# =============================================================================

# Input file names
INPUT_FILES = {
    'prices': 'DE_LU_DAM_prices.csv',
    'actual': 'actual_data.csv',
    'evaluation': 'evaluation_data.csv',
    'historical': 'historical_predictions.csv'
}

# Output file naming
OUTPUT_NAMING = {
    'forecasts': 'forecasts_{date}.csv',
    'evaluation': 'Evaluation_Results.xlsx',
    'summary': 'Summary_results.txt',
    'best_hourly': 'Best_Hourly_Table.pkl'
}

# =============================================================================
# LOGGING CONFIGURATION
# =============================================================================

# Logging settings
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = 'epf_system.log'

# Progress reporting
VERBOSE = True
SHOW_PROGRESS = True
PRINT_INTERMEDIATE_RESULTS = True

# =============================================================================
# VALIDATION CONFIGURATION
# =============================================================================

# Data validation settings
VALIDATE_DATA_RANGES = True
MIN_PRICE_THRESHOLD = -500    # Minimum realistic price (EUR/MWh)
MAX_PRICE_THRESHOLD = 3000    # Maximum realistic price (EUR/MWh)
MIN_DATA_COMPLETENESS = 0.8   # Minimum data completeness ratio

# Model validation
MIN_MODELS_REQUIRED = 3       # Minimum number of models for ensemble
MAX_FORECAST_HORIZON = 7      # Maximum days ahead to forecast

# =============================================================================
# PERFORMANCE CONFIGURATION
# =============================================================================

# Memory management
CHUNK_SIZE = 1000            # For processing large datasets
MAX_MEMORY_USAGE = '2GB'     # Maximum memory usage

# Parallel processing (if implemented)
N_JOBS = -1                  # Number of parallel jobs (-1 = all cores)
USE_MULTIPROCESSING = False  # Enable multiprocessing for model training

# =============================================================================
# EXPORT FUNCTIONS
# =============================================================================

def get_model_config():
    """Get the complete model configuration."""
    return {
        'ml_models': ML_MODELS if INCLUDE_ML_MODELS else [],
        'deep_learning_models': DEEP_LEARNING_MODELS if INCLUDE_DEEP_LEARNING else [],
        'time_series_models': TIME_SERIES_MODELS if INCLUDE_TIME_SERIES else [],
        'ensemble_models': ENSEMBLE_MODELS
    }

def get_data_config():
    """Get the data configuration."""
    return {
        'countries': COUNTRIES_CONFIG,
        'daily_granularity': DAILY_GRANULARITY_TABLES,
        'input_files': INPUT_FILES
    }

def get_forecast_config():
    """Get the forecasting configuration."""
    return {
        'zones': ZONES,
        'starting_date': STARTING_DATE,
        'days_ahead': DAYS_AHEAD,
        'training_version': TRAINING_VERSION,
        'table_pred': TABLE_PRED,
        'start_date_ts': START_DATE_TS,
        'days_back': DAYS_BACK,
        'include_naive': INCLUDE_NAIVE_MODEL,
        'include_ts': INCLUDE_TIME_SERIES
    }

def get_teams_config():
    """Get the Teams integration configuration."""
    return {
        'enabled': TEAMS_NOTIFICATION,
        'webhook_url': TEAMS_WEBHOOK_URL,
        'color': TEAMS_MESSAGE_COLOR,
        'title_template': TEAMS_MESSAGE_TITLE_TEMPLATE
    }

def validate_config():
    """Validate configuration settings."""
    errors = []
    
    # Check required directories
    for dir_path in [DATA_DIR, MODELS_DIR]:
        if not Path(dir_path).exists():
            errors.append(f"Directory does not exist: {dir_path}")
    
    # Check required files
    data_path = Path(DATA_DIR)
    for file_key, filename in INPUT_FILES.items():
        if not (data_path / filename).exists():
            errors.append(f"Required data file missing: {filename}")
    
    # Check Teams configuration
    if TEAMS_NOTIFICATION and not TEAMS_WEBHOOK_URL:
        errors.append("Teams notification enabled but webhook URL not provided")
    
    # Check date ranges
    if DAYS_AHEAD < 1 or DAYS_AHEAD > MAX_FORECAST_HORIZON:
        errors.append(f"DAYS_AHEAD must be between 1 and {MAX_FORECAST_HORIZON}")
    
    if DAYS_BACK < 1:
        errors.append("DAYS_BACK must be at least 1")
    
    return errors

if __name__ == "__main__":
    # Validate configuration when run directly
    errors = validate_config()
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
    else:
        print("Configuration validation passed!")
        
    # Print current configuration
    print("\nCurrent Configuration:")
    print(f"  Zones: {ZONES}")
    print(f"  Days ahead: {DAYS_AHEAD}")
    print(f"  Training version: {TRAINING_VERSION}")
    print(f"  Include time series: {INCLUDE_TIME_SERIES}")
    print(f"  Create plots: {CREATE_PLOTS}")
    print(f"  Teams notification: {TEAMS_NOTIFICATION}")
