"""
Utility Functions for Residual Multivariate Transformers
========================================================

This module contains utility functions for data processing, train-test splitting,
and covariate learning for residual multivariate transformer models.
"""

import pandas as pd
import numpy as np
import sys
import os

# Add parent directory to path to access data_preparation module
sys.path.append('..')
import data_preparation

from .config_residual_transformer import (
    DEFAULT_SPLIT_RATIO, 
    DEFAULT_INIT_DATE, 
    DEFAULT_CATEGORICAL_VARS,
    PANDEMIC_WAVES
)


def split_train_test(df, split_ratio=None, init_date=None):
    """
    Splits a dataframe into train and test sets using the given split ratio.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The input dataframe to split
    split_ratio : float, optional
        The fraction of data to be used for training (default from config)
    init_date : str, optional
        Initial date to filter data from (default from config)
        
    Returns:
    --------
    tuple
        (train_df, test_df) - Training and testing datasets
        
    Raises:
    -------
    KeyError
        If 'timestamp' column is not found in the DataFrame
    """
    # Use default parameters if not provided
    if split_ratio is None:
        split_ratio = DEFAULT_SPLIT_RATIO
    if init_date is None:
        init_date = DEFAULT_INIT_DATE
    
    # Keep only rows STRICTLY after init_date using the 'timestamp' column
    if 'timestamp' not in df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")
    
    df = df.copy()  # Create a copy to avoid modifying the original
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    cutoff = pd.Timestamp(init_date)
    df = df[df['timestamp'] > cutoff].reset_index(drop=True)
    
    print(f"Data filtered from {init_date}. Remaining records: {len(df)}")
    
    # Process each column (except timestamp)
    for code in df.columns:
        if code != 'timestamp':
            # Convert to numeric
            df[code] = pd.to_numeric(df[code], errors='coerce')
            
            # Remove rows with NaN values for this column
            df = df.dropna(subset=[code]).reset_index(drop=True)

            # Min-max scale the target column
            cmin, cmax = df[code].min(), df[code].max()
            if pd.isna(cmin) or pd.isna(cmax) or cmax == cmin:
                print(f"Warning: Column {code} has constant values or NaN. Setting to 0.0")
                df[code] = 0.0
            else:
                df[code] = (df[code] - cmin) / (cmax - cmin)
                print(f"Column {code} normalized. Range: [{cmin:.4f}, {cmax:.4f}]")

    # Split the data
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    print(f"Data split - Train: {len(train_df)} records, Test: {len(test_df)} records")
    
    return train_df, test_df


def learn_covariates(train_split, categorical_vars=None):
    """
    Process time series data to extract and prepare covariates.
    
    Parameters:
    -----------
    train_split : pd.DataFrame
        Training data split containing timestamp and target variables
    categorical_vars : list, optional
        List of categorical variables to create (default from config)
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame with time series features and covariates
    """
    # Use default categorical variables if not provided
    if categorical_vars is None:
        categorical_vars = DEFAULT_CATEGORICAL_VARS.copy()
    
    print(f"Learning covariates with variables: {categorical_vars}")
    
    df = train_split.copy()

    # Call the function from data_preparation module
    df_processed = data_preparation.prepare_time_series_features(df, categorical_vars)

    # Display information about the processed DataFrame
    print("Processed DataFrame info:")
    print(f"  Shape: {df_processed.shape}")
    print(f"  Columns: {list(df_processed.columns)}")
    print(f"  Date range: {df_processed['timestamp'].min()} to {df_processed['timestamp'].max()}")
    
    # Check for any remaining NaN values
    nan_counts = df_processed.isnull().sum()
    if nan_counts.any():
        print("NaN values found in columns:")
        for col, count in nan_counts[nan_counts > 0].items():
            print(f"  {col}: {count}")
    
    return df_processed


def prepare_residual_data(original_predictions, actual_values):
    """
    Compute residuals for residual learning.
    
    Parameters:
    -----------
    original_predictions : np.ndarray
        Predictions from the original model
    actual_values : np.ndarray
        Actual target values
        
    Returns:
    --------
    np.ndarray
        Residuals (actual - predicted)
    """
    residuals = actual_values - original_predictions
    
    print(f"Residual statistics:")
    print(f"  Mean: {np.mean(residuals):.6f}")
    print(f"  Std: {np.std(residuals):.6f}")
    print(f"  Min: {np.min(residuals):.6f}")
    print(f"  Max: {np.max(residuals):.6f}")
    
    return residuals


def create_pandemic_waves_df():
    """
    Create a DataFrame with pandemic wave information.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with pandemic wave periods
    """
    # Convert waves dictionary to DataFrame
    df_waves = pd.DataFrame(PANDEMIC_WAVES).T.reset_index()
    df_waves.columns = ["Onada", "Inici", "Final"]
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])
    
    print("Pandemic waves DataFrame created:")
    print(df_waves.to_string(index=False))
    
    return df_waves


def validate_data_shapes(X, Y, expected_sequence_length=None, expected_features=None):
    """
    Validate the shapes of input data arrays.
    
    Parameters:
    -----------
    X : np.ndarray
        Input features array
    Y : np.ndarray
        Target values array
    expected_sequence_length : int, optional
        Expected sequence length
    expected_features : int, optional
        Expected number of features
        
    Returns:
    --------
    bool
        True if all validations pass
        
    Raises:
    -------
    ValueError
        If validation fails
    """
    print(f"Validating data shapes:")
    print(f"  X shape: {X.shape}")
    print(f"  Y shape: {Y.shape}")
    
    # Check that X and Y have the same number of samples
    if X.shape[0] != Y.shape[0]:
        raise ValueError(f"Mismatch in number of samples: X has {X.shape[0]}, Y has {Y.shape[0]}")
    
    # Check expected sequence length
    if expected_sequence_length is not None and len(X.shape) > 1:
        if X.shape[1] != expected_sequence_length:
            raise ValueError(f"Expected sequence length {expected_sequence_length}, got {X.shape[1]}")
    
    # Check expected number of features
    if expected_features is not None and len(X.shape) > 2:
        if X.shape[2] != expected_features:
            raise ValueError(f"Expected {expected_features} features, got {X.shape[2]}")
    
    print("✅ Data shape validation passed")
    return True


def load_and_preprocess_data(data_path, target_code=None, split_ratio=None):
    """
    Load and preprocess data from CSV file.
    
    Parameters:
    -----------
    data_path : str
        Path to the CSV data file
    target_code : str, optional
        Target variable code to focus on
    split_ratio : float, optional
        Train-test split ratio (default from config)
        
    Returns:
    --------
    tuple
        (train_df, test_df) - Preprocessed training and testing data
    """
    print(f"Loading data from: {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Load the data
    df = pd.read_csv(data_path)
    print(f"Loaded data shape: {df.shape}")
    
    if target_code:
        print(f"Filtering for target code: {target_code}")
        # Keep only timestamp and target code columns
        if target_code in df.columns and 'timestamp' in df.columns:
            df = df[['timestamp', target_code]].copy()
        else:
            available_cols = list(df.columns)
            raise ValueError(f"Target code '{target_code}' not found. Available columns: {available_cols}")
    
    # Split the data
    train_df, test_df = split_train_test(df, split_ratio=split_ratio)
    
    return train_df, test_df


def extract_model_params_from_filename(model_filename):
    """
    Extract model parameters from filename.
    
    Parameters:
    -----------
    model_filename : str
        Model filename containing parameters
        
    Returns:
    --------
    dict
        Dictionary with extracted parameters (code, forecast, ff_dim, lookback, lr)
    """
    import re
    
    # Pattern to extract parameters from filename
    # Expected format: {code}_*_{forecast}fh_{ff_dim}ff_{lookback}lb_{lr}initlr.keras
    pattern = r'(\w+)_.*_(\d+)fh_(\d+)ff_(\d+)lb_([\d.]+)initlr'
    
    match = re.search(pattern, model_filename)
    if match:
        return {
            'code': match.group(1),
            'forecast': int(match.group(2)),
            'ff_dim': int(match.group(3)),
            'lookback': int(match.group(4)),
            'learning_rate': float(match.group(5))
        }
    else:
        print(f"Warning: Could not extract parameters from filename: {model_filename}")
        return None