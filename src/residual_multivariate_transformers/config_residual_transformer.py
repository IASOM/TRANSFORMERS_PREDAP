"""
Configuration Module for Residual Multivariate Transformers
===========================================================

This module contains all configuration parameters and settings for residual
multivariate transformer models.
"""

# Data Processing Parameters
DEFAULT_SPLIT_RATIO = 0.8
DEFAULT_INIT_DATE = '2010-01-01'

# Categorical Variables for Time Series Features
DEFAULT_CATEGORICAL_VARS = [
    "Day_of_Week", 
    "Month", 
    "Season", 
    "Holiday", 
    "School_Vacation"
]

# Model Architecture Parameters
DEFAULT_TRANSFORMER_PARAMS = {
    'head_size': 2,
    'num_heads': 2,
    'ff_dim': 8,
    'dropout': 0.2
}

DEFAULT_LSTM_PARAMS = {
    'units_1': 64,
    'units_2': 32,
    'dropout': 0.2,
    'return_sequences': True
}

# Training Parameters
DEFAULT_TRAINING_PARAMS = {
    'batch_size': 32,
    'epochs': 100,
    'validation_split': 0.1,
    'shuffle': False,
    'patience': 25
}

# Learning Rate Schedule Parameters
DEFAULT_LR_SCHEDULE_PARAMS = {
    'initial_lr': 1e-4,
    'max_lr': 1e-3,
    'min_lr': 1e-5,
    'warmup_steps': 20,
    'total_steps': 50
}

# Default Forecast and Lookback Settings
DEFAULT_FORECAST = 30
DEFAULT_LOOKBACK = 30

# Default Learning Rate
DEFAULT_LEARNING_RATE = 0.001

# Model Saving Parameters
DEFAULT_SAVE_PARAMS = {
    'save_history': False,
    'save_model': True,
    'save_memory': False
}

# Pandemic Waves Configuration
PANDEMIC_WAVES = {
    "Primera Onada": ("2020-03", "2020-06"),
    "Segona Onada": ("2020-10", "2020-12"),
    "Tercera Onada": ("2021-01", "2021-03"),
    "Quarta Onada": ("2021-04", "2021-06"),
}

# File Extensions
MODEL_EXTENSION = '.keras'
HISTORY_EXTENSION = '_history.pkl'
MEMORY_LOG_FILE = 'memory.csv'

# Directory Paths
DEFAULT_MODEL_DIR = 'models'
DEFAULT_DATA_PATH = "J:/longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv"