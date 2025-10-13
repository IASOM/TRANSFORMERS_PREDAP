
"""
Main Transformers Utils - Refactored to use Univariate Transformer Module
=========================================================================

This script has been refactored to use functions from the univariate_transformer
module instead of having duplicate implementations. All core functions like
build_model, train_given_model_and_data, transformer_encoder, etc. are now
imported from the modular structure.

The remaining functions in this file are either:
- Utility functions specific to this script
- Modified versions needed for compatibility
- Functions not yet moved to the modular structure
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
import math
import os 
import time
from tensorflow import keras
import pickle
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Lambda
import re
import os

import data_preparation

# Import functions from univariate_transformer module
from univariate_transformer import (
    build_model,
    transformer_encoder, 
    CustomCosineDecay,
    train_given_model_and_data,
    evaluate_model_sliding_window,
    plt_model,
    plot_predictions_with_waves,
    plot_example,
    extract_model_params,
    setup_gpu_memory,
    create_model_directories,
    create_pandemic_waves_df,
    load_and_preprocess_data,
    default_config
)

# plot_example function is now imported from univariate_transformer module


def evaluate_model(model, X_test, Y_test, sliding_window=10, plt_results=True):
    """
    Evaluate the model using a sliding window approach.

    Args:
        model: Trained Keras model.
        X_test: Test input data (shape: (samples, lookback, 1)).
        Y_test: True future values (shape: (samples, n_pred)).
        sliding_window: Number of test samples to slide over.
    
    Returns:
        Plots the MSE, MAE, and Loss over time.
    """

    mse_list = []
    mae_list = []
    loss_list = []

    num_samples = len(X_test) - sliding_window + 1  # Number of sliding steps

    for i in range(num_samples):
        X_window = X_test[i : i + sliding_window]  # Get sliding window input
        Y_window_true = Y_test[i : i + sliding_window]  # True values

        # Predict using the model
        Y_window_pred = model.predict(X_window, verbose=0)

        # Compute metrics
        mse = mean_squared_error(Y_window_true, Y_window_pred)
        mae = mean_absolute_error(Y_window_true, Y_window_pred)
        loss = np.mean(np.abs(Y_window_true - Y_window_pred))  # Approximate loss (MAE)

        # Store metrics
        mse_list.append(mse)
        mae_list.append(mae)
        loss_list.append(loss)

    # Plot results
    if plt_results:
        plot_evaluations(mse_list, mae_list, loss_list)

    return mse_list, mae_list, loss_list



def plot_evaluations(mse_list, mae_list, loss_list):
    """
    Plots the MSE, MAE, and Loss over time.

    Args:
        mse_list: List of Mean Squared Error values.
        mae_list: List of Mean Absolute Error values.
        loss_list: List of Loss values.
    """

    plt.figure(figsize=(15, 5))

    # Plot MSE
    plt.subplot(3, 1, 1)
    plt.plot(mse_list, label="MSE")
    plt.xlabel("Time")
    plt.ylabel("MSE")
    plt.title("Mean Squared Error (MSE)")
    plt.legend()

    # Plot MAE
    plt.subplot(3, 1, 2)
    plt.plot(mae_list, label="MAE", color="orange")
    plt.xlabel("Time")
    plt.ylabel("MAE")
    plt.title("Mean Absolute Error (MAE)")
    plt.legend()

    # Plot Loss
    plt.subplot(3, 1, 3)
    plt.plot(loss_list, label="Loss", color="red")
    plt.xlabel("Time")
    plt.ylabel("Loss")
    plt.title("Loss Over Time")
    plt.legend()

    plt.tight_layout()
    plt.show()

# transformer_encoder function is now imported from univariate_transformer module

# build_model function is now imported from univariate_transformer module

# CustomCosineDecay class is now imported from univariate_transformer module

# train_given_model_and_data function is now imported from univariate_transformer module



# plot_predictions_with_waves function is now imported from univariate_transformer module
    
def evaluate_model(model,model_name, X_test, Y_test, date_list, df_waves, sliding_window=10):
    """
    Evaluate the model using a sliding window approach.

    Args:
        model: Trained Keras model.
        X_test: Test input data (shape: (samples, lookback, 1)).
        Y_test: True future values (shape: (samples, n_pred)).
        sliding_window: Number of test samples to slide over.
    
    Returns:
        Plots the MSE, MAE, and Loss over time.
    """

    mse_list = []
    mae_list = []
    loss_list = []

    num_samples = len(X_test) - sliding_window + 1  # Number of sliding steps

    for i in range(num_samples):
        X_window = X_test[i : i + sliding_window]  # Get sliding window input
        Y_window_true = Y_test[i : i + sliding_window]  # True values

        # Predict using the model
        Y_window_pred = model.predict(X_window, verbose=0)

        # Compute metrics
        mse = mean_squared_error(Y_window_true, Y_window_pred)
        mae = mean_absolute_error(Y_window_true, Y_window_pred)
        loss = np.mean(np.abs(Y_window_true - Y_window_pred))  # Approximate loss (MAE)

        # Store metrics
        mse_list.append(mse)
        mae_list.append(mae)
        loss_list.append(loss)
    
    # Ensure date_list length matches metric lists
    trimmed_dates = date_list[-len(mse_list):]  # Take only the last elements

    plt.figure(figsize=(12, 10))

    # Plot MSE
    ax1 = plt.subplot(3, 1, 1)
    for i, row in df_waves.iterrows():
        ax1.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)
    ax1.plot(trimmed_dates, mse_list, label="MSE", color="blue")
    ax1.set_xlabel("Data")
    ax1.set_ylabel("MSE")
    ax1.set_title("Mean Squared Error (MSE) Over Time")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)
    #ax1.set_ylim(0, 1)

    # Plot MAE
    ax2 = plt.subplot(3, 1, 2)
    for i, row in df_waves.iterrows():
        ax2.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)
    ax2.plot(trimmed_dates, mae_list, label="MAE", color="orange")
    ax2.set_xlabel("Data")
    ax2.set_ylabel("MAE")
    ax2.set_title("Mean Absolute Error (MAE) Over Time")
    ax2.legend()
    ax2.tick_params(axis="x", rotation=45)
    #ax2.set_ylim(0, 1)

    # Plot Loss
    ax3 = plt.subplot(3, 1, 3)
    for i, row in df_waves.iterrows():
        ax3.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)
    ax3.plot(trimmed_dates, loss_list, label="Loss", color="red")
    ax3.set_xlabel("Data")
    ax3.set_ylabel("Loss")
    ax3.set_title("Loss Over Time")
    ax3.legend()
    ax3.tick_params(axis="x", rotation=45)
    #ax3.set_ylim(0, 1)

    plt.tight_layout()

    plt.savefig(f"plots/evaluation_metrics_over_time_{model_name}.png")  # Save the figure
    plt.show()

# plt_model function is now imported from univariate_transformer module

# extract_model_params function is now imported from univariate_transformer module


if __name__ == "__main__":
    # Use configuration from univariate_transformer module
    config = default_config
    
    # Set default values from configuration
    FORECAST = config.FORECAST
    LOOKBACK = config.LOOKBACK
    
    # Transformer model parameters
    HEAD_SIZE = config.HEAD_SIZE
    NUM_HEADS = config.NUM_HEADS
    NUM_TRANSFORMER_BLOCKS = config.NUM_TRANSFORMER_BLOCKS
    FF_DIM = config.FF_DIM
    
    # Multi-Layer Perceptron (MLP) Parameters
    MLP_UNITS = config.MLP_UNITS
    MLP_DROPOUT = config.MLP_DROPOUT
    
    # General Regularization Parameters
    DROPOUT = config.DROPOUT
    
    # Optimization and Training Parameters
    LEARNING_RATE = config.LEARNING_RATE
    EPOCHS = config.EPOCHS
    EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
    
    # Setup GPU memory and create directories
    setup_gpu_memory()
    create_model_directories()

    # RAW DATA --------------------------------------------------------------
    data_path = "J:/longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv"
    df = pd.read_csv(data_path, index_col=0)
    df['COV-19'] = df["B34"]+df["U07"]   # join cov19 codes
    df = df.drop(['B34', 'U07'], axis=1)

    df.index = pd.to_datetime(df.index)
    d = pd.to_datetime('2000-01-01')
    mask = df.index < d
    df = df.loc[~mask]

    d = pd.to_datetime('2010-01-01')
    mask = df.index < d
    df = df.loc[~mask]

    df["Overall"] = df.iloc[:, 1:].sum(axis=1)
    print(df)


    plot_example(df,"RAW DATA (example 10 diags) COVID")

    code = "T14"

    '''# Create a new DataFrame with timestamp and target columns
    transformed_df = df[["date", code]].rename(columns={"date": "timestamp", code: code})
    transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
    transformed_df.reset_index(drop=True, inplace=True)

    # Apply Min-Max Scaling (0 to 1) only to the "target" column
    scaler = MinMaxScaler(feature_range=(0, 1))
    transformed_df[code] = scaler.fit_transform(transformed_df[[code]]) 

    # Apply function to split data
    train_df, test_df = split_train_test(transformed_df)
    print(train_df.info())
    print(test_df.info())'''


    start_time = time.perf_counter()
    input_directory = f"J:/longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv" # define data dir
    batchSize=16               # how many samples processing in parallel during training
    shuffle=False              # randomize order of training data??


    start_time = time.perf_counter()

    csvFile = f'{input_directory}/{code}_train_example.csv'
    #
    # prepare X and Y arrays
    X, Y = data_preparation.prepare_data(input_directory, code, LOOKBACK, FORECAST, debug=True, univariate=True)

    finish_preparing = time.perf_counter()
    time_data_preparation = finish_preparing - start_time
    print("finished preparing data, time spent:",time_data_preparation)

    
    # declare model
    model = build_model(
        (LOOKBACK,1),
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_units=[MLP_UNITS],
        mlp_dropout=MLP_DROPOUT,
        dropout=DROPOUT,
        n_pred=FORECAST#+1
    )

    model.summary()


    # MODEL TRAINING --------------------------------------------------------------

    # Create the custom learning rate schedule
    LR_init=LEARNING_RATE
    LR_min=LR_init*10
    LR_max=LR_init*10*10
    scheduler = CustomCosineDecay(initial_lr=LR_init, max_lr=LR_max, min_lr=LR_min, warmup_steps=EPOCHS/5, total_steps=EPOCHS)

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(
            scheduler),
        tf.keras.callbacks.EarlyStopping(
            patience=EARLY_STOP_PATIENCE,
            monitor='val_loss',
            mode='min',
            restore_best_weights=True)]


    model.compile(loss='MSE', metrics=['mae', 'mse'], optimizer=Adam())

    # train model
    MODEL_NAME = f'models/{code}_example_transformer_{FORECAST}fh_{FF_DIM}ff_{LOOKBACK}lb_{LEARNING_RATE}initlr.keras'
    callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]
    train_given_model_and_data(model, X, Y, model_name=MODEL_NAME, epochs=EPOCHS, save_model=True, save_memory=False, callbacks=callbacks)

    LOOKBACK_LIST = [7]#[1,7,14,30,60, 182,365]
    FORECAST_LIST = [7]#[1,7,14,30,60, 182,365]

    # train different models for different lookback and forecast horizons
    for lb in LOOKBACK_LIST:

        csvFile = f'{input_directory}/{code}_train_example.csv'

        for fh in FORECAST_LIST:

            # prepare X and Y arrays
            X, Y = data_preparation.prepare_data(input_directory, code, lb, fh, debug=True, univariate =True)
            
            start_time = time.perf_counter()
            # declare model
            model = build_model(
                (lb,1),
                head_size=HEAD_SIZE,
                num_heads=NUM_HEADS,
                ff_dim=FF_DIM,
                num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
                mlp_units=[MLP_UNITS],
                mlp_dropout=MLP_DROPOUT,
                dropout=DROPOUT,
                n_pred=fh#+1
            )
            
            # Create the custom learning rate schedule
            scheduler = CustomCosineDecay(initial_lr=LR_init, max_lr=LR_max, min_lr=LR_min, warmup_steps=EPOCHS/5, total_steps=EPOCHS)

            callbacks = [
                tf.keras.callbacks.LearningRateScheduler(scheduler),
                tf.keras.callbacks.EarlyStopping(
                    patience=EARLY_STOP_PATIENCE,
                    monitor='val_loss',
                    mode='min',
                    restore_best_weights=True)]


            model.compile(loss='MSE', metrics=['mae', 'mse'], optimizer=Adam())
            
            # train model
            MODEL_NAME = f'models/{code}_example_transformer_{fh}fh_{FF_DIM}ff_{lb}lb_{LEARNING_RATE}initlr.keras'
            callbacks = [tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=EARLY_STOP_PATIENCE, restore_best_weights=True)]
            train_given_model_and_data(model, X, Y, model_name=MODEL_NAME, epochs=EPOCHS, save_model=True, save_memory=False, callbacks=callbacks)

            finish_preparing = time.perf_counter()
            time_data_preparation = finish_preparing - start_time
            print("finished modelling, time spent:",time_data_preparation)


    # EXPLORE AND TEST MODELS --------------------------------------------------------------
    MODEL_NAME = f'models/{code}_example_transformer_{FORECAST}fh_{FF_DIM}ff_{LOOKBACK}lb_{LEARNING_RATE}initlr.keras'
    print(MODEL_NAME)

    # MODEL EVALUATION 
    start_time = time.perf_counter()
    input_directory = "J:/longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv"

    csvFile = f'{input_directory}/{code}_test_example.csv'

    # prepare X and Y arrays
    X_test, Y_test = data_preparation.prepare_data(input_directory, code, LOOKBACK, FORECAST, debug=True, univariate=True)
    date_list = data_preparation.extract_dates(input_directory, code, LOOKBACK, FORECAST)

    finish_preparing = time.perf_counter()
    time_data_preparation = finish_preparing - start_time
    print("finished preparing data, time spent:",time_data_preparation)
    print("finished preparing timesteps, time spent:",len(date_list))

    MODEL_FOLDER = 'models'
    print("Files in model folder:", os.listdir(MODEL_FOLDER))
    trained_models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.keras')]
    print("Trained models:", trained_models)

    for model_name in trained_models:
        model_path = os.path.join(MODEL_FOLDER, model_name)
        model = tf.keras.models.load_model(model_path, compile=True)
        # Print model architecture
        model.summary()

        # Create pandemic waves DataFrame using imported function
        df_waves = create_pandemic_waves_df()

        lookback, forecast = extract_model_params(model_name)
        X_test, Y_test = data_preparation.prepare_data(input_directory, code, lookback, forecast,train = False, debug=True, univariate=True)
        date_list = data_preparation.extract_dates(input_directory, code, lookback, forecast)
        # Assuming X_test and Y_test are prepared
        loss, mae, mse = model.evaluate(X_test, Y_test)

        print(f"Test Loss: {loss}")
        print(f"Test MAE: {mae}")   # Mean Absolute Error
        print(f"Test MSE: {mse}")   # Mean Squared Error

        # Get predictions
        predictions = model.predict(X_test)
        print("Predicted values:", predictions.shape)

        plt_model(Y_test, predictions, model_name=model_name, col_idx=0, show_plt=True)
        # Call the function with formatted dates
        plot_predictions_with_waves(Y_test, predictions, date_list, df_waves)

        # Use the imported sliding window evaluation function
        evaluate_model_sliding_window(model, model_name, X_test, Y_test, date_list, df_waves, sliding_window=forecast)

        