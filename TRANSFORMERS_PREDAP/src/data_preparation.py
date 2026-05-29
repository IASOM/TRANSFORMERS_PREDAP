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
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler 

MAX_DATE = '2021-06-30'
# Define function for train-test split
def split_train_test(df, split_ratio=0.8):
    """
    Splits a dataframe into train and test sets using the given split ratio.

    Parameters:
    - df (pd.DataFrame): The input dataframe to split.
    - split_ratio (float): The fraction of data to be used for training (default is 0.8).

    Returns:
    - train_df (pd.DataFrame): Training dataset.
    - test_df (pd.DataFrame): Testing dataset.
    """
    split_idx = int(len(df) * split_ratio)  # Compute split index
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, test_df


def normalize_dataframe(train_df,test_df, csv_file = None, save_data = False,  target_code=None, scaler=None):
    '''
    Normalizes the train and test dataframes using the provided scaler. 
    If no scaler is provided, a new MinMaxScaler is used by default.
    Parameters:
    - train_df (pd.DataFrame): Training dataframe.
    - test_df (pd.DataFrame): Testing dataframe.
    - csv_file (str): Path to the original CSV file (for saving normalized data).
    - save_data (bool): Whether to save the normalized dataframes to CSV files.
    - target_code (str): The column name of the target variable to scale separately.
    - scaler: Pre-fitted scaler to use for normalization (optional).
    Returns:
    - train_df (pd.DataFrame): Normalized training dataframe.
    - test_df (pd.DataFrame): Normalized testing dataframe.
    '''
    if 'timestamp' not in train_df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")
    train_df['timestamp'] = pd.to_datetime(train_df['timestamp'], errors='coerce')
    test_df['timestamp'] = pd.to_datetime(test_df['timestamp'], errors='coerce')
    
    if scaler is None:
        scaler = MinMaxScaler()
        scaler_target = MinMaxScaler()

    else:
        scaler_target = scaler
        
    codes = [code for code in train_df.columns if (code != 'timestamp' and code != target_code)]
    scaler.fit(train_df[codes].values)
    train_df[codes] = scaler.transform(train_df[codes].values)  
    test_df[codes] = scaler.transform(test_df[codes].values)
    if target_code is not None:
        # Min-max scale ONLY the target column (univariate)
        scaler_target = scaler_target.fit(train_df[[target_code]].values)
        test_df[[target_code]] = scaler_target.transform(test_df[[target_code]].values)
        train_df[[target_code]] = scaler_target.transform(train_df[[target_code]].values)

    if save_data and csv_file is not None:
        input_path = Path(csv_file)
        parent_dir = input_path.parent if input_path.parent != Path('.') else AssertionError("Input path must have a parent directory.")
        output_file = parent_dir / f"train_df_normalized_{input_path.name}"
        train_df.to_csv(output_file, index=False)
        print(f"Normalized data saved to: {output_file}")
    
    return train_df, test_df


def cut_dataframe(df:pd.DataFrame, date_cutoff: str = '2010-01-01', max_date: str = MAX_DATE, csv_file: str = None, save_data: bool = False)-> pd.DataFrame:
    # Keep only rows STRICTLY after the cutoff date using the 'timestamp' column
    '''
    Cuts the dataframe to only include rows after the specified cutoff date and up to the max date.
    Parameters:
    - df (pd.DataFrame): The input dataframe to cut.
    - date_cutoff (str): The cutoff date in 'YYYY-MM-DD' format.
    - max_date (str): The maximum date in 'YYYY-MM-DD' format.
    - csv_file (str): Path to the original CSV file (for saving cut data).
    - save_data (bool): Whether to save the cut dataframe to a CSV file.
    Returns:
    - pd.DataFrame: The cut dataframe.
    '''
    if 'timestamp' not in df.columns:
        raise KeyError("Expected a 'timestamp' column in the CSV.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    cutoff = pd.Timestamp(date_cutoff)
    max_dt = pd.Timestamp(max_date)
    df = df[(df['timestamp'] > cutoff) & (df['timestamp'] <= max_dt)].reset_index(drop=True) 

    if save_data and csv_file is not None:
        input_path = Path(csv_file)
        parent_dir = input_path.parent if input_path.parent != Path('.') else AssertionError("Input path must have a parent directory.")
        output_file = parent_dir / f"date_{date_cutoff}_{input_path.name}"
        df.to_csv(output_file, index=False)
        print(f"Cut data saved to: {output_file}")

    return df

def eliminate_covid_dates(df:pd.DataFrame, covid_periods:list) -> pd.DataFrame:
    """
    Subtracts the days specified in the covid periods list from the dataframe provided.
    input:
        - df (pd.DataFrame): DataFrame containing a 'timestamp' column.
        - covid_periods (list): List of tuples with start and end dates of covid periods.
    output:
        - pd.DataFrame: DataFrame with covid periods removed.
    """
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    for start_date, end_date in covid_periods:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
        df = df[~((df['timestamp'] >= start) & (df['timestamp'] <= end))].reset_index(drop=True)
    return df

def inverse_transform_predictions(predictions, original_scale_df, code, lookback, forecast, cutoff_date='2010-01-01', max_date='2021-06-30', scaler = None, eliminate_covid_data=False, covid_dates=None):

    """
    Inverses the min-max scaling of predictions to the original scale.

    Parameters:
    - predictions (np.array): Scaled predictions to inverse transform.
    - original_scale_df (pd.DataFrame): Original dataframe used for scaling.
    - code (str): The column name of the target variable.

    Returns:
    - np.array: Predictions in the original scale.
    """

    # Extract the original values for the target code
    #original_values = original_scale_df[code].values.reshape(-1, 1)
    code = code.replace("#", ":")
    if eliminate_covid_data:
        assert covid_dates is not None
        original_scale_df = eliminate_covid_dates(original_scale_df, covid_dates)


    # Fit scaler on original values
    original_scale_df['timestamp'] = pd.to_datetime(original_scale_df['timestamp'], errors='coerce')
    cutoff = pd.Timestamp(cutoff_date)
    max_date = pd.Timestamp(max_date)
    original_scale_df = original_scale_df[(original_scale_df['timestamp'] > cutoff)&(original_scale_df['timestamp'] <= max_date)].reset_index(drop=True)  # Subset the DataFrame
    
    train_df, test_df = split_train_test(original_scale_df)
    
    train_df_seq = []
    
    for i in range(len(train_df[[code]]) - lookback - forecast + 1):
        train_df_seq.append(train_df[[code]].values[i + lookback : i + lookback + forecast])  # Future `forecast` values
    train_df_seq = np.array(train_df_seq).squeeze()
    # Min-max scale ONLY the target column (univariate)
    if scaler is None:
        scaler_target = MinMaxScaler()
    else:
        scaler_target = scaler
    if len(train_df_seq.shape) < 2:
        train_df_seq = train_df_seq.reshape(-1,1)
    scaler_target.fit(train_df_seq)
    pred_original_scale = scaler_target.inverse_transform(predictions)

    #pred_original_scale = pred_original_scale.reshape(predictions.shape)

    return pred_original_scale


def inverse_causal_transform_predictions(predictions, original_scale_df, code, lookback, forecast, cutoff_date='2010-01-01', max_dt='2021-06-30', scaler = None):

    """
    Inverses the min-max scaling of predictions to the original scale.

    Parameters:
    - predictions (np.array): Scaled predictions to inverse transform.
    - original_scale_df (pd.DataFrame): Original dataframe used for scaling.
    - code (str): The column name of the target variable.

    Returns:
    - np.array: Predictions in the original scale.
    """

    # Extract the original values for the target code
    #original_values = original_scale_df[code].values.reshape(-1, 1)
    
    # Fit scaler on original values
    original_scale_df['timestamp'] = pd.to_datetime(original_scale_df['timestamp'], errors='coerce')
    cutoff = pd.Timestamp(cutoff_date)
    original_scale_df = original_scale_df[(original_scale_df['timestamp'] > cutoff)&(original_scale_df['timestamp'] <= max_dt)].reset_index(drop=True)  # Subset the DataFrame
    
    train_df, test_df = split_train_test(original_scale_df)
    '''codes = [code for code in train_df.columns if code != 'timestamp']
    scaler = MinMaxScaler()
    scaler.fit(train_df[code].values.reshape(-1, 1))

    #pred_original_array = np.zeros_like(predictions)
    # Inverse transform predictions
    pred_flat_pred = predictions.reshape(-1,1)
    pred_original_array = scaler.inverse_transform(pred_flat_pred)
    pred_original_array = pred_original_array.reshape(predictions.shape)
    for i, pred in enumerate(predictions):
        pred = pred.reshape(-1, 1)
        pred_orig = scaler.inverse_transform(pred)
        pred_original_array[i] = pred_orig.flatten()'''
    
    train_df_seq = []
    
    # Min-max scale ONLY the target column (univariate)
    if scaler is None:
        scaler_target = MinMaxScaler()
    else:
        scaler_target = scaler

    scaler_target.fit(train_df[[code]].values)
    if len(predictions.shape) > 2:
        predictions = np.squeeze(predictions, axis = 2)
    pred_original_scale = scaler_target.inverse_transform(predictions)

    #pred_original_scale = pred_original_scale.reshape(predictions.shape)

    return pred_original_scale

def add_covid_token(df):
    """
    Add a column to the dataframe indicating COVID-19 period. The covid token is 1 during the pandemic period and 0 otherwise."""

    waves = {
        "Primera Onada": ("2020-03-01", "2020-06-30"),
        "Segona Onada": ("2020-10-01", "2020-12-31"),
        "Tercera Onada": ("2021-01-01", "2021-03-31"),
        "Quarta Onada": ("2021-04-01", "2021-06-30"),
        "Cinquena Onada": ("2021-07-01", "2021-09-30")
    }
    
    for wave, (start, end) in waves.items():
        start_date = pd.to_datetime(start)
        end_date = pd.to_datetime(end)
        df['covid_token'] = df['timestamp'].apply(
            lambda x: 1 if start_date <= x <= end_date else 0
        )

    return df

def prepare_data(csv_file,code, lookback, forecast, cutoff_date = '2010-01-01', max_date = '2021-06-30', covid_token = False, relevant_feature_cols = None,train = True, debug=False, univariate=True, scaler = None, eliminate_covid_data = False, covid_dates = None):
    code = code.replace("#", ":")
    start_time =time.time()
    # Load CSV
    df = pd.read_csv(csv_file)
    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    df = cut_dataframe(df, cutoff_date,max_date, csv_file)
    train_df, test_df = split_train_test(df)
    train_df, test_df = normalize_dataframe(train_df,test_df, csv_file, target_code=code, scaler = scaler)

    if train:
        df = train_df
    else:
        df = test_df

    if univariate:
        # univariate scenario ...................................................
        # Only use the target column as input (Univariate Forecasting)
        #feature_col = df.columns[-1]   # Use the target itself as input
        #target_col = df.columns[-1]    # The future target to predict
        idx_code = df.columns.get_loc(code)
        feature_cols = df.columns[idx_code]  
        target_col = df.columns[idx_code]  # Get the target column 

        # Convert to numpy arrays
        X_raw = df[feature_cols].values.reshape(-1, 1)  # Ensure shape is (rows, 1)
        Y_raw = df[target_col].values # Target values
        print(X_raw.shape, Y_raw.shape)
    else: 
        # multivariate scenario ..................................................
        # Select feature columns (exclude timestamp & target)
        idx_code = df.columns.get_loc(code)
        
        df_features = df.drop(columns = ['timestamp'])

        
        target_col = df.columns[idx_code]  # Target code column
    
        # Convert DataFrame to numpy arrays
        if relevant_feature_cols is not None:
            X_raw = df_features[relevant_feature_cols].values  
        else:
            X_raw = df_features.values
        Y_raw = df[target_col].values
    if covid_token:
        df_covid = add_covid_token(df)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
        
        X_raw = np.hstack((X_raw, covid_feature))
    # Generate rolling sequences
    X, Y = [], []
    for i in range(len(X_raw) - lookback - forecast + 1):
        X.append(X_raw[i : i + lookback])  # Create `lookback` sequence
        Y.append(Y_raw[i + lookback : i + lookback + forecast])  # Future `forecast` values

    # Convert to numpy arrays
    X, Y = np.array(X), np.array(Y)

    if debug:
        print(f"Processed Data Shapes: X={X.shape}, Y={Y.shape}")  
    end_time = time.time()
    print(f"Data preparation took {end_time - start_time:.2f} seconds")
    return X, Y


def prepare_causal_data(csv_file,code, lookback, forecast, cutoff_date = '2010-01-01',covid_token = False, relevant_feature_cols = None,train = True, debug=False, univariate=True):
    start_time =time.time()
    # Load CSV
    df = pd.read_csv(csv_file)
    
    df = cut_dataframe(df, cutoff_date, csv_file)
    train_df, test_df = split_train_test(df)
    train_df, test_df = normalize_dataframe(train_df,test_df, csv_file, target_code=code)

    if train:
        df = train_df
    else:
        df = test_df

    if univariate:
        # univariate scenario ...................................................
        # Only use the target column as input (Univariate Forecasting)
        #feature_col = df.columns[-1]   # Use the target itself as input
        #target_col = df.columns[-1]    # The future target to predict
        idx_code = df.columns.get_loc(code)
        feature_cols = df.columns[idx_code]  
        target_col = df.columns[idx_code]  # Get the target column 

        # Convert to numpy arrays
        X_raw = df[feature_cols].values.reshape(-1, 1)  # Ensure shape is (rows, 1)
        Y_raw = df[target_col].values # Target values
        print(X_raw.shape, Y_raw.shape)
    else: 
        # multivariate scenario ..................................................
        # Select feature columns (exclude timestamp & target)
        idx_code = df.columns.get_loc(code)
        
        df_features = df.drop(columns = ['timestamp'])

        
        target_col = df.columns[idx_code]  # Target code column
    
        # Convert DataFrame to numpy arrays
        if relevant_feature_cols is not None:
            X_raw = df_features[relevant_feature_cols].values  
        else:
            X_raw = df_features.values
        Y_raw = df[target_col].values
    if covid_token:
        df_covid = add_covid_token(df)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
        
        X_raw = np.hstack((X_raw, covid_feature))
    # Convert to numpy arrays
    X = X_raw[:lookback]
    Y = Y_raw[lookback:lookback + forecast]
    X, Y = np.array(X), np.array(Y)

    if debug:
        print(f"Processed Data Shapes: X={X.shape}, Y={Y.shape}")  
    end_time = time.time()
    print(f"Data preparation took {end_time - start_time:.2f} seconds")
    return X, Y


def prepare_data_not_normalized(csv_file,code, lookback, forecast, cutoff_date = '2010-01-01', max_date = '2021-06-30', covid_token = False, relevant_feature_cols = None,train = True, debug=False, univariate=True, eliminate_covid_data=False, covid_dates=None):
    # Load CSV
    code = code.replace("#", ":")
    df = pd.read_csv(csv_file)
    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    df = cut_dataframe(df, cutoff_date, max_date, csv_file)
    train_df, test_df = split_train_test(df)
    
    if train:
        df = train_df
    else:
        df = test_df

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
        idx_code = df.columns.get_loc(code)
        
        df_features = df.drop(columns = ['timestamp'])

        
        target_col = df.columns[idx_code]  # Target code column
    
        # Convert DataFrame to numpy arrays
        if relevant_feature_cols is not None:
            X_raw = df_features[relevant_feature_cols].values  
        else:
            X_raw = df_features.values
        Y_raw = df[target_col].values
    if covid_token:
        df_covid = add_covid_token(df)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
        X_raw = np.hstack((X_raw, covid_feature))
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

def prepare_causal_data_not_normalized(csv_file,code, lookback, forecast, cutoff_date = '2010-01-01',covid_token = False, relevant_feature_cols = None,train = True, debug=False, univariate=True):
    # Load CSV
    df = pd.read_csv(csv_file)

    df = cut_dataframe(df, cutoff_date, csv_file)
    train_df, test_df = split_train_test(df)
    
    if train:
        df = train_df
    else:
        df = test_df

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
        idx_code = df.columns.get_loc(code)
        
        df_features = df.drop(columns = ['timestamp'])

        
        target_col = df.columns[idx_code]  # Target code column
    
        # Convert DataFrame to numpy arrays
        if relevant_feature_cols is not None:
            X_raw = df_features[relevant_feature_cols].values  
        else:
            X_raw = df_features.values
        Y_raw = df[target_col].values
    if covid_token:
        df_covid = add_covid_token(df)
        covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
        X_raw = np.hstack((X_raw, covid_feature))
    
    X = X_raw
    Y = Y_raw
    # Convert to numpy arrays
    X, Y = np.array(X), np.array(Y)

    if debug:
        print(f"Processed Data Shapes: X={X.shape}, Y={Y.shape}")  

    return X, Y


def extract_dates(csv_file,code,lookback, forecast, train = True, cutoff_date = '2010-01-01', max_date = MAX_DATE, eliminate_covid_data=False, covid_dates=None):
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

    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)

    if 'timestamp' not in df.columns:
        raise ValueError("The dataset must contain a 'timestamp' column for plotting.")
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    cutoff = pd.Timestamp(cutoff_date)
    max_dt = pd.Timestamp(max_date)
    df = df[(df['timestamp'] > cutoff) & (df['timestamp'] <= max_dt)].reset_index(drop=True) 

    # Convert to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_train, df_test = split_train_test(df)
    if train:
        df = df_train
    else:
        df = df_test

    # Align dates with the first prediction in each sequence
    date_list = df['timestamp'].iloc[lookback : len(df) - forecast + 1].reset_index(drop=True)

    return date_list.tolist()


def extract_causal_dates(csv_file,code,lookback, forecast, train = True, cutoff_date = '2010-01-01', max_date = MAX_DATE):
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
    cutoff = pd.Timestamp(cutoff_date)
    max_dt = pd.Timestamp(max_date)
    df = df[(df['timestamp'] > cutoff) & (df['timestamp'] <= max_dt)].reset_index(drop=True) 

    # Convert to datetime format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df_train, df_test = split_train_test(df)
    if train:
        df = df_train
    else:
        df = df_test

    # Align dates with the first prediction in each sequence
    date_list = df['timestamp'].reset_index(drop=True)

    return date_list.tolist()

def prepare_time_series_features(df, categorical_vars, cutoff_date = '2010-01-01', max_date = MAX_DATE, scaler = None, eliminate_covid_data=False, covid_dates=None):
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
    
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')

    if eliminate_covid_data:
        assert covid_dates is not None
        df = eliminate_covid_dates(df, covid_dates)
    cutoff = pd.Timestamp(cutoff_date)
    max_dt = pd.Timestamp(max_date)
    df = df[(df['timestamp'] > cutoff) & (df['timestamp'] <= max_dt)].reset_index(drop=True) 
    
    # Convert timestamp to datetime (optional)
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    if scaler is None:
        scaler = MinMaxScaler()
    
    codes = [code for code in df.columns if code != 'timestamp']
    #train_df, test_df = split_train_test(df)

    scaler.fit(df[codes])
    df[codes] = scaler.transform(df[codes])


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

    # Ensure same datetime type
    public_holidays_df['timestamp'] = pd.to_datetime(public_holidays_df['timestamp'])

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
    df_dates["Is_Weekend"] = (df_dates["timestamp"].dt.dayofweek >= 5).astype(bool)
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
    if eliminate_covid_data:
        df_final = eliminate_covid_dates(df_final, covid_dates)

    return df_final

def generate_rolling_sequences_covariates(df_processed, lookback, forecast, predictions_train=None, generate_y = False):
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
    X = [X_raw[i+lookback : i + lookback + forecast] for i in range(len(X_raw) - lookback - forecast + 1)]
    
    
    # Convert to NumPy array
    X_train_covs = np.array(X)

    print(f"Processed covariate data Shapes: X={X_train_covs.shape}")

    # If predictions are provided, concatenate as an extra feature
    if predictions_train is not None:
        # Ensure predictions are correctly shaped
        predictions_train = predictions_train.reshape(X_train_covs.shape[0],forecast, 1)
        
        #repeated_predictions_train = np.repeat(predictions_train, lookback, axis=1)  # Repeat to match lookback length
        X_train_covs = np.concatenate([X_train_covs, predictions_train], axis=-1)  # Add as extra feature

        print(f"Processed covariate Shapes + predictions: X={X_train_covs.shape}")
    

    if generate_y == True:
        Y = [X_raw[i : i + lookback ] for i in range(len(X_raw) - lookback - forecast + 1)]
        Y_train_covs = np.array(Y)
        print(f"Processed covariate data Shapes: Y={Y_train_covs.shape}")
        return Y_train_covs
    else:
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


def compute_dynamic_batch_size(lookback, forecast):
    """
    Computes an appropriate batch size based on lookback and forecast parameters.

    Parameters:
    - lookback (int): Number of past timesteps used for input sequences.
    - forecast (int): Number of timesteps predicted.

    Returns:
    - int: Computed batch size.
    """
    gpus = tf.config.list_physical_devices('GPU')

    
    
    if lookback <= 30 and forecast <= 30:
        batch_size = 512
    elif (30 <= lookback <= 60) and forecast <= 60:
        batch_size = 256
    elif 60 <= lookback <= 128 and forecast<= 128:
        batch_size = 64
    elif 128 < lookback <= 365 and forecast <=365:
        batch_size = 32
    else:
        batch_size = 64
    

    if len(gpus) == 0:
        batch_size = 32
        
    return batch_size
