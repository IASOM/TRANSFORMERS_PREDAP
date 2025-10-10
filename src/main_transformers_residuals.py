#!/usr/bin/env python
# coding: utf-8
import importlib
import evaluation_plot_utils
import data_preparation
importlib.reload(data_preparation)
import time
import pandas as pd
import os
import re
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed
from tensorflow.keras.layers import Dropout
from tensorflow import keras
from tensorflow.keras import layers
import math
import pickle


def learn_covariates(csvFile_tr, categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation"]):
    df = pd.read_csv(csvFile_tr)

    # Call the function
    df_processed = data_preparation.prepare_time_series_features(df, categorical_vars)

    # Display the processed DataFrame
    print(df_processed.info())

    return df_processed

if __name__ == "__main__":
    code = "J00_NOPAND"
    forecast = 30
    lookback = 30
    ff_dim = 8
    learning_rate = 0.001

    model_name = f'{code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{learning_rate}initlr'

    start_time = time.perf_counter()
    input_directory =  "J:/longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv"

    
    # prepare X and Y arrays
    X_test, Y_test = data_preparation.prepare_data(input_directory, lookback, forecast, debug=True, univariate =True)
    date_list_test = data_preparation.extract_dates(input_directory, lookback, forecast)

    X_train, Y_train = data_preparation.prepare_data(input_directory, lookback, forecast, debug=True, univariate =True)
    date_list_train = data_preparation.extract_dates(input_directory, lookback, forecast)

    finish_preparing = time.perf_counter()
    time_data_preparation = finish_preparing - start_time
    print("finished preparing data, time spent:",time_data_preparation)
    print("finished preparing timesteps test, time spent:",len(date_list_test))
    print("finished preparing timesteps train, time spent:",len(date_list_train))


    model_name = f'{code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{learning_rate}initlr'
    print(model_name)
    base_path = "/models"

    # Get available folders
    available_folders = [f for f in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, f))]
    print("Available model folders:", available_folders)

    # Automatically select the correct one
    for folder in available_folders:
        if model_name in folder:  # Check the pattern
            full_path = os.path.join(base_path, folder)
            print(f"✅ Using detected model folder: {folder}")
            break
    else:
        print("❌ No matching folder found!")
        full_path = None

    # Load the model if a valid path was found
    if full_path:
        
        model = tf.keras.models.load_model(full_path)
        model.summary()

#GET PREDICTIONS FROM TRANSFORMER UNIVARIATE MODEL
# Get predictions test
predictions_test = model.predict(X_test)
print("Predicted test values:", predictions_test.shape)
print("Real test values:", Y_test.shape)

# Get predictions train
predictions_train = model.predict(X_train)
print("Predicted train values:", predictions_train.shape)
print("Real train values:", Y_train.shape)

#COMPUTE RESIDUALS FOR TRAIN AND TEST 

# Create new dataset for residual model
Y_train_residual = Y_train - predictions_train
Y_test_residual = Y_test - predictions_test

#Y_train_residual = np.expand_dims(Y_train_residual, axis=-1)
#Y_test_residual = np.expand_dims(Y_test_residual, axis=-1)

print("Residuals shape train:", Y_train_residual.shape)
print("Residuals shape test:", Y_test_residual.shape)