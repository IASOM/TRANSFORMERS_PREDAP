# ==========================================================
# DATA PREPARATION
# Module to prepare data for trainig-testing purposes
# ==========================================================

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from dateutil.easter import easter


def prepare_data(csv_file,code, lookback, forecast, debug=False, univariate=True):
    # Load CSV
    df = pd.read_csv(csv_file)
    # Keep only rows STRICTLY after 2010-01-01 using the 'timestamp' column
    if 'timestamp' not in df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    cutoff = pd.Timestamp('2010-01-01')
    df = df[df['timestamp'] > cutoff].reset_index(drop=True)  # Subset the DataFrame
    
    # Convert timestamp to datetime (optional)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    

    df[code] = pd.to_numeric(df[code], errors='coerce')
    df = df.dropna(subset=[code]).reset_index(drop=True)

    # Min-max scale ONLY the target column (univariate)
    cmin, cmax = df[code].min(), df[code].max()
    if pd.isna(cmin) or pd.isna(cmax) or cmax == cmin:
        df[code] = 0.0
    else:
        df[code] = (df[code] - cmin) / (cmax - cmin)
        
    if univariate:
        # univariate scenario ...................................................
        # Only use the target column as input (Univariate Forecasting)
        #feature_col = df.columns[-1]   # Use the target itself as input
        #target_col = df.columns[-1]    # The future target to predict
        idx_code = df.columns.get_loc(code)
        feature_cols = df.columns[idx_code]  # Ignore timestamp, exclude target
        target_col = df.columns[idx_code]  # Target is the last column

        # Convert to numpy arrays
        X_raw = df[feature_cols].values.reshape(-1, 1)  # Ensure shape is (rows, 1)
        Y_raw = df[target_col].values    # Target values
        print(X_raw.shape, Y_raw.shape)
    else: 
        # multivariate scenario ..................................................
        # Select feature columns (exclude timestamp & target)
        feature_cols = df.columns[1:-1]  # Ignore timestamp, exclude target
        target_col = df.columns[-1]  # Target is the last column
    
        # Convert DataFrame to numpy arrays
        X_raw = df[feature_cols].values  # Shape: (200, 3)
        Y_raw = df[target_col].values    # Shape: (200,)

    # Generate rolling sequences
    X, Y = [], []
    for i in range(len(X_raw) - lookback - forecast + 1):
        X.append(X_raw[i : i + lookback])  # Create `lookback` sequence
        Y.append(Y_raw[i + lookback : i + lookback + forecast])  # Future `forecast` values

    # Convert to numpy arrays
    X, Y = np.array(X), np.array(Y)

    if debug:
        print(f"Processed Data Shapes: X={X.shape}, Y={Y.shape}")  

    return X, Y


def extract_dates(csv_file,code, lookback, forecast):
    """
    Extracts the 'date' column from the CSV file to align with the test dataset for plotting.

    Args:
        csv_file (str): Path to the CSV file.
        lookback (int): Number of past timesteps used for input sequences.
        forecast (int): Number of timesteps predicted.

    Returns:
        list: A list of datetime values corresponding to the test predictions.
    """
    df = pd.read_csv(csv_file)
    #df = df[code]

    if 'timestamp' not in df.columns:
        raise ValueError("The dataset must contain a 'timestamp' column for plotting.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    cutoff = pd.Timestamp('2010-01-01')
    df = df[df['timestamp'] > cutoff].reset_index(drop=True)  # Subset the DataFrame
    # Convert to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Align dates with the first prediction in each sequence
    date_list = df['timestamp'].iloc[lookback : len(df) - forecast + 1].reset_index(drop=True)

    return date_list.tolist()


def prepare_time_series_features(df, categorical_vars):
    """
    Prepares a time series dataset by adding date-related features (holidays, school vacations, etc.)
    and dummifying categorical variables.

    Parameters:
    - df (pd.DataFrame): Input dataset containing a 'timestamp' column.
    - categorical_vars (list): List of categorical variables to dummify.

    Returns:
    - df_final (pd.DataFrame): Processed DataFrame with additional features.
    """

    # Ensure 'timestamp' column is in datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # Define fixed public holidays
    fixed_holidays = {
        "New Year's Day": "01-01",
        "Epiphany": "01-06",
        "Labour Day": "05-01",
        "Feast of St. John the Baptist": "06-24",
        "Assumption of the Virgin": "08-15",
        "National Day of Catalonia": "09-11",
        "Hispanic Day": "10-12",
        "All Saints' Day": "11-01",
        "Constitution Day": "12-06",
        "Immaculate Conception Day": "12-08",
        "Christmas Day": "12-25",
        "St. Stephen's Day": "12-26"
    }

    # Function to determine movable holidays
    def get_movable_holidays(year):
        good_friday = easter(year) - pd.Timedelta(days=2)
        easter_monday = easter(year) + pd.Timedelta(days=1)
        return {"Good Friday": good_friday, "Easter Monday": easter_monday}

    # Generate a list of public holidays
    public_holidays = []
    for year in range(df['timestamp'].min().year, df['timestamp'].max().year + 1):
        for holiday_name, date_str in fixed_holidays.items():
            holiday_date = pd.Timestamp(f"{year}-{date_str}")
            public_holidays.append((holiday_date, holiday_name))
        for holiday_name, holiday_date in get_movable_holidays(year).items():
            public_holidays.append((holiday_date, holiday_name))

    public_holidays_df = pd.DataFrame(public_holidays, columns=["timestamp", "Holiday"])

    # Generate date range
    date_range = pd.date_range(start=df['timestamp'].min(), end=df['timestamp'].max(), freq='D')
    
    df_dates = pd.DataFrame({
        'timestamp': date_range,
        'Day_of_Week': date_range.day_name(),
        'Month': date_range.month_name(),
        'Season': date_range.month.map(lambda m: "Winter" if m in [12, 1, 2] else 
                                                    "Spring" if m in [3, 4, 5] else
                                                    "Summer" if m in [6, 7, 8] else "Autumn")
    })

    # Merge with holidays
    df_dates = df_dates.merge(public_holidays_df, on="timestamp", how="left")
    df_dates["Holiday"] = df_dates["Holiday"].fillna("No Holiday")

    # Define school vacation periods
    school_vacations = [
        ("Christmas Break", "12-23", "01-07"),  
        ("Easter Break", "03-23", "04-06"),  
        ("Summer Break", "06-22", "09-10")  
    ]

    # Function to determine school vacations
    def get_school_vacation(date):
        year = date.year
        for vacation_name, start_date, end_date in school_vacations:
            if start_date.startswith("12") and end_date.startswith("01"):
                start_dec = pd.Timestamp(f"{year-1}-{start_date}")
                end_jan = pd.Timestamp(f"{year}-{end_date}")
                if start_dec <= date <= end_jan:
                    return vacation_name
            else:
                start = pd.Timestamp(f"{year}-{start_date}")
                end = pd.Timestamp(f"{year}-{end_date}")
                if start <= date <= end:
                    return vacation_name
        return "No Vacation"

    df_dates["School_Vacation"] = df_dates["timestamp"].apply(get_school_vacation)
    
    # Simplify categories
    df_dates['Holiday'] = np.where(df_dates['Holiday'] != "No Holiday", 'Holiday', 'No_Holiday')
    df_dates['School_Vacation'] = np.where(df_dates['School_Vacation'] != "No Vacation", 'Vacation', 'No_Vacation')

    # Merge with original df on timestamp
    #df_final = df.merge(df_dates, on="timestamp", how="left")

    # Dummify categorical variables
    df_final = pd.get_dummies(df_dates, columns=categorical_vars, drop_first=False)

    return df_final

def generate_rolling_sequences_covariates(df_processed, lookback, forecast, predictions_train=None):
    """
    Generates rolling sequences for multivariate time series forecasting.
    
    Parameters:
    - df_processed (pd.DataFrame): Processed DataFrame (should exclude timestamp & target).
    - lookback (int): Number of past timesteps to include in each sequence.
    - forecast (int): Number of future timesteps to predict.
    - predictions_train (np.array, optional): Model predictions to include as a feature. 
      Expected shape: (num_samples, forecast, 1).
    
    Returns:
    - X_train_covs (np.array): Rolling sequences of covariates with shape (samples, forecast, features).
    """

    # Select feature columns (exclude timestamp)
    feature_cols = df_processed.columns[1:]  # Ignore timestamp column
    X_raw = df_processed[feature_cols].values  # Convert DataFrame to NumPy array

    # Generate rolling sequences
    X = [X_raw[i + lookback : i + lookback + forecast] for i in range(len(X_raw) - lookback - forecast + 1)]
    
    # Convert to NumPy array
    X_train_covs = np.array(X)

    print(f"Processed covariate data Shapes: X={X_train_covs.shape}")

    # If predictions are provided, concatenate as an extra feature
    if predictions_train is not None:
        # Ensure predictions are correctly shaped
        predictions_train = predictions_train.reshape(X_train_covs.shape[0], X_train_covs.shape[1], 1)
        X_train_covs = np.concatenate([X_train_covs, predictions_train], axis=-1)  # Add as extra feature

        print(f"Processed covariate Shapes + predictions: X={X_train_covs.shape}")

    return X_train_covs

def shift_covariates(df, forecast):
    """
    Shifts all covariates back in time by the forecast range to prevent future leakage.

    Parameters:
    - df (pd.DataFrame): DataFrame containing timestamp (first column) and covariates.
    - forecast (int): Number of future timesteps to predict.

    Returns:
    - pd.DataFrame: DataFrame with shifted covariates (NaNs at the start).
    """

    df_shifted = df.copy()

    # Shift all covariate columns (except the timestamp) back by 'forecast' steps
    df_shifted.loc[:,:] = df_shifted.loc[:, :].shift(forecast)

    return df_shifted

def prepare_time_series_covariates(df, df_covs):
    """
    Ensures that 'timestamp' exists in both df and df_covs, converts it to datetime,
    and subsets df_covs based on df's date range.
    """
    
    # Ensure 'timestamp' exists and is datetime in df
    df['timestamp'] = pd.to_datetime(df.get('timestamp'), errors='coerce')
    if df['timestamp'].isna().all():
        raise KeyError("Valid 'timestamp' column not found in df.")
    
    # Handle 'timestamp' in df_covs
    if 'timestamp' not in df_covs.columns:
        df_covs = df_covs.reset_index() if df_covs.index.name == 'timestamp' else df_covs
        df_covs.rename(columns={col: 'timestamp' for col in df_covs.columns if 'date' in col.lower() or 'time' in col.lower()}, inplace=True)
    
    df_covs['timestamp'] = pd.to_datetime(df_covs.get('timestamp'), errors='coerce')
    df_covs.dropna(subset=['timestamp'], inplace=True)
    
    # Subset df_covs based on df's date range
    return df_covs[df_covs['timestamp'].between(df['timestamp'].min(), df['timestamp'].max())]

# Define function to subset df_covs based on the date range in df
def subset_df_covs_by_index(df, df_covs):
    """
    Subsets df_covs to only include rows where the index date is between 
    the min and max dates in df.

    Parameters:
    - df (pd.DataFrame): DataFrame containing the reference date range.
    - df_covs (pd.DataFrame): DataFrame with covariates where the index is in datetime format.

    Returns:
    - pd.DataFrame: Subset of df_covs with index dates within df's date range.
    """
    
    # Ensure df timestamp is in datetime format
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    if df['timestamp'].isna().all():
        raise KeyError("Valid 'timestamp' column not found in df.")

    # Ensure df_covs index is in datetime format
    df_covs = df_covs.copy()
    if not isinstance(df_covs.index, pd.DatetimeIndex):
        raise ValueError("df_covs index must be a DatetimeIndex.")

    # Subset df_covs based on min/max dates in df
    df_covs_subset = df_covs.loc[df['timestamp'].min():df['timestamp'].max()]

    return df_covs_subset
