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

# Define function for train-test split
def split_train_test(df, split_ratio=0.8):
    """
    Splits a dataframe into train and test sets using the given split ratio.

    Parameters:
    df (pd.DataFrame): The input dataframe to split.
    split_ratio (float): The fraction of data to be used for training (default is 0.8).

    Returns:
    train_df (pd.DataFrame): Training dataset.
    test_df (pd.DataFrame): Testing dataset.
    """
    split_idx = int(len(df) * split_ratio)  # Compute split index
    train_df = df.iloc[:split_idx].reset_index(drop=True)
    test_df = df.iloc[split_idx:].reset_index(drop=True)
    
    return train_df, test_df


def learn_covariates(train_split, categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation"]):
    df = train_split

    # Call the function
    df_processed = data_preparation.prepare_time_series_features(df, categorical_vars)

    # Display the processed DataFrame
    print(df_processed.info())

    return df_processed

def train_given_model_and_data(model, X, Y, batch_size=1024, model_name=None, epochs=100, save_history=False, save_model=True, save_memory=True, shuffle=False, callbacks=None):
    if save_memory: #configure GPU memory growth
        # prepare for measuring memory
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError as e:
                print(e)
    if callbacks is None: # define callbacks (If no callbacks are provided, it automatically enables Early Stopping)
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=25, restore_best_weights=True)
        callbacks = [early_stop]

    if not os.path.exists(f'{model_name}'): #check if model exist and run if not
        history = model.fit(x=X, 
                            y=Y, 
                            batch_size=batch_size,  # batch gradient descent (batch size 1024)
                            epochs=epochs, 
                            shuffle=shuffle,        # Allows shuffling
                            validation_split=0.1,   # 30% data for validation
                            callbacks=callbacks)
    else:
        print(f"model {model_name} already exists")
        return

    if model_name is None:
        model_name = "testing"

    if save_history: # save training history
        with open(f'{model_name}_history.pkl', 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
    
    if save_model and epochs > 1: # save model
        model.save(model_name)
        
    if save_memory: # log memory usage (optional)
        # save memory usave
        # Get memory information
        memory_info = tf.config.experimental.get_memory_info('GPU:0')
        with open('memory.csv', 'a') as resultcsv:
            resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
        print(f"Current memory usage: {memory_info['current'] / (batch_size**2)} MB")
        print(f"Peak memory usage: {memory_info['peak'] / (batch_size**2)} MB")

# Transformer Encoder Block
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.2):
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)                  
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)  
    x = layers.Dropout(dropout)(x) 
    res = x + inputs  

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)                       
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(x) 
    x = layers.Dropout(dropout)(x)                                         
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)          
    return x + res    


def hybrid_lstm_transformer_model(input_shape, forecast):
    input_layer = keras.Input(shape=input_shape)

    # LSTM Block
    x = layers.LSTM(64, return_sequences=True)(input_layer)
    x = layers.Dropout(0.2)(x)  # Dropout to reduce overfitting
    x = layers.LSTM(32, return_sequences=True)(x)
    x = layers.Dropout(0.2)(x)

    # Transformer Block
    x = transformer_encoder(x, head_size=2, num_heads=2, ff_dim=8, dropout=0.2)

    # Output Layer
    outputs = layers.TimeDistributed(Dense(1))(x)

    # Build Model
    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile Model
    model.compile(optimizer='adam', loss='mse')

    return model



# defaults to 50 epochs total and 20 warmup steps
class CustomCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, initial_lr=1e-4, max_lr=1e-3, min_lr=1e-5, warmup_steps=20, total_steps=50):
        self.initial_lr = initial_lr  # Starting learning rate (0.01)
        self.max_lr = max_lr          # Maximum learning rate (0.1)
        self.min_lr = min_lr          # Final learning rate (0.0001)
        self.warmup_steps = warmup_steps  # Number of warmup steps to reach max_lr
        self.total_steps = total_steps    # Total number of steps (decay after warmup)

    def __call__(self, step):
        # Warm-up phase: linearly increase to max_lr
        if step < self.warmup_steps:
            return self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.warmup_steps)

        # Cosine decay phase after warmup
        decay_steps = self.total_steps - self.warmup_steps
        step_after_warmup = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_after_warmup / decay_steps))
        decayed = (self.max_lr - self.min_lr) * cosine_decay + self.min_lr
        return decayed
    

if __name__ == "__main__":
    code = "T14"
    forecast = 30
    lookback = 1
    ff_dim = 8
    learning_rate = 0.001

    model_name = f'{code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{learning_rate}initlr'

    start_time = time.perf_counter()
    input_directory =  "J:/longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv"

    
    # prepare X and Y arrays
    X_test, Y_test = data_preparation.prepare_data(input_directory, code,lookback, forecast, debug=True, univariate =True)
    date_list_test = data_preparation.extract_dates(input_directory,code, lookback, forecast)

    X_train, Y_train = data_preparation.prepare_data(input_directory,code, lookback, forecast, debug=True, univariate =True)
    date_list_train = data_preparation.extract_dates(input_directory,code, lookback, forecast)

    finish_preparing = time.perf_counter()
    time_data_preparation = finish_preparing - start_time
    print("finished preparing data, time spent:",time_data_preparation)
    print("finished preparing timesteps test, time spent:",len(date_list_test))
    print("finished preparing timesteps train, time spent:",len(date_list_train))


    model_name = f'{code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{learning_rate}initlr'
    print(model_name)
    base_path = "models"

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

    train_split, test_split = split_train_test(pd.read_csv(input_directory))
    df_processed = learn_covariates(train_split)

    # Generate rolling sequences
    X_train_covs = data_preparation.generate_rolling_sequences_covariates(df_processed, lookback, forecast, predictions_train)

    # Verify shape
    print("Final Shape:", X_train_covs.shape)  # Should be (num_samples, forecast, num_features)
    print(f"Residuals Shapes: X={Y_train_residual.shape}")  


    # Define Hybrid LSTM + Transformer Model
    residual_model = hybrid_lstm_transformer_model((lookback, X_train_covs.shape[2]), forecast)
    residual_model.summary()

    model_name = f'{code}_SEASONAL_RESIDUALS_LEARNING_{forecast}fh_{ff_dim}ff_{lookback}lb_{learning_rate}initlr'

    train_given_model_and_data(residual_model, X_train_covs, Y_train_residual, batch_size=32, model_name=model_name, 
                               epochs=100, save_model=True, save_memory=False, callbacks=None)

    #MODEL TESTING 

    
    # List of categorical variables to dummify
    categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation"]

    # Call the function
    df_test_processed = data_preparation.prepare_time_series_features(test_split, categorical_vars)

    # Display the processed DataFrame
    print(df_test_processed.info())

    # Generate rolling sequences
    X_test_covs = data_preparation.generate_rolling_sequences_covariates(df_test_processed, lookback, forecast, predictions_test)

    # Verify shape
    print("Final Shape:", X_test_covs.shape)  # Should be (num_samples, forecast, num_features)
    print(f"Residuals Shapes: X={Y_test_residual.shape}")  

    # Predict residuals for the test set
    predicted_residuals = model.predict(X_test_covs)  # Shape: (num_samples, 14, 1)
    predicted_residuals = np.squeeze(predicted_residuals, axis=-1)

    # Correct the original forecast
    corrected_forecast = predictions_test + predicted_residuals

    evaluation_plot_utils.plot_stepwise_errors(Y_test, predictions_test)
    evaluation_plot_utils.plot_stepwise_errors(Y_test, corrected_forecast)

    date_list = date_list_test
    waves = {
        "Primera Onada": ("2020-03", "2020-06"),
        "Segona Onada": ("2020-10", "2020-12"),
        "Tercera Onada": ("2021-01", "2021-03"),
        "Quarta Onada": ("2021-04", "2021-06"),
        #"Cinquena Onada": ("2021-07", "2021-09")
    }

    # Convertir a DataFrame per facilitar la representació
    df_waves = pd.DataFrame(waves).T.reset_index()
    df_waves.columns = ["Onada", "Inici", "Final"]
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])

    predictions_to_plot = predictions_test.mean(axis=1) 
    Y_test_to_plot = Y_test.mean(axis=1)

    # Call the function with formatted dates
    evaluation_plot_utils.plot_predictions_with_waves(Y_test_to_plot, predictions_to_plot, date_list, df_waves, waves=False)

    evaluation_plot_utils.plot_errors_over_time_with_waves(Y_test_to_plot, predictions_to_plot, date_list, df_waves, waves=False)

    evaluation_plot_utils.evaluate_error_significance_pandemic_waves(Y_test_to_plot, predictions_to_plot, date_list, df_waves)