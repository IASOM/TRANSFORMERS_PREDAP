
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

# plot example 10 diags -------------------------
def plot_example(df,title):
    # PLOT RAW DATA (example 10 diags)
    dff = df
    dff["date"] = dff.index
    dff = dff[['J00','COV-19','date']]
    sns.set_theme(rc={'figure.figsize':(20,8)})
    sns.lineplot(data=dff.replace('nan', float('nan')).melt(id_vars=['date']),x='date', y='value', hue='variable').set(title=title)


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

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):

    
    d_model = max(head_size * num_heads, 8)
    x = layers.Dense(d_model, activation="relu")(inputs)
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)                  # to imputs to stabilize training
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x,x)   # self attention to normalized input 
    x = layers.Dropout(dropout)(x) # (dropout to reduce overfitting)
    res = x + inputs               # attention output added to original inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)                       # again after resid connection
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(x) # point-wise convol.: Expands feature dim to ff_dim using tanh activ
    x = layers.Dropout(dropout)(x)                                         # dropout again
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)          # reduces feature dim back to match input size
    x = x + res
    
    return x

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, n_pred=1):
    inputs = keras.Input(shape=input_shape) #defines input tensor
    
    x = inputs                            #initial input
    #
    for _ in range(num_transformer_blocks): #apply num_transformer_blocks transformer encoder layers seq.
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout) #uses previous defined trans_encoder layer

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x) # reduces seq dimension (timesteps) averaging for each fature channel
    for dim in mlp_units:                                              # multi layer perceptron (dropout to avoid overfitting)
        x = layers.Dense(dim, activation="tanh")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_pred)(x)
    return keras.Model(inputs, outputs)

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
        early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', mode='min', patience=5, restore_best_weights=True)
        callbacks = [early_stop]

    if not os.path.exists(f'{model_name}'): #check if model exist and run if not
        history = model.fit(x=X, 
                            y=Y, 
                            batch_size=batch_size,  # batch gradient descent (batch size 1024)
                            epochs=epochs, 
                            shuffle=shuffle,        # Allows shuffling
                            validation_split=0.3,   # 30% data for validation
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
        print(f"Current memory usage: {memory_info['current'] / (1024**2)} MB")
        print(f"Peak memory usage: {memory_info['peak'] / (1024**2)} MB")



def plot_predictions_with_waves(Y_test, predictions, date_list, df_waves):
    """
    Plots actual vs. predicted values with dates as x-labels, showing only 15 evenly spaced date labels.
    Also highlights COVID-19 pandemic waves with a red background.

    Args:
        Y_test (array): Actual target values.
        predictions (array): Predicted values from the model.
        date_list (list): List of datetime values for the x-axis.
        df_waves (DataFrame): Contains pandemic waves' start and end dates.
    """
    # Convert timestamps to string format (YYYY-MM-DD)
    date_labels = [date.strftime('%Y-%m-%d') for date in date_list]

    # Select 15 evenly spaced indices for x-axis labels
    num_labels = 15
    indices = np.linspace(0, len(date_list) - 1, num_labels, dtype=int)

    plt.figure(figsize=(20, 5))

    # Highlight pandemic waves with a red background
    for i, row in df_waves.iterrows():
        plt.axvspan(row["Inici"], row["Final"], color="red", alpha=0.2)

    # Plot actual and predicted values
    plt.plot(date_list, Y_test, label="Actual Values (Y-test)", marker='o', linestyle='-', alpha=0.7)
    plt.plot(date_list, predictions, label="Predicted Values", marker='x', linestyle='--', alpha=0.7)

    plt.xlabel("Date")
    plt.ylabel("Target Value (J00)")
    plt.title("Predictions vs. Actual Values (J00 - Y-test) with Pandemic Waves")
    #plt.ylim(0, 1)
    plt.legend()

    # Apply only 15 labels to the x-axis
    plt.xticks([date_list[i] for i in indices], [date_labels[i] for i in indices], rotation=45)

    plt.grid()
    plt.show()
    
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

def plt_model(y_test_inverse, yhat_inverse, model_name, col_idx=None, show_plt = False):
    """
    Plot model results comparing true vs predicted values.
    
    Parameters:
    -----------
    y_test_inverse : np.ndarray
        True values (inverse transformed)
    yhat_inverse : np.ndarray
        Predicted values (inverse transformed)
    model_name : str
        Name of the model for the plot title
    col_idx : int, optional
        Column index to plot (defaults to global col_idx)
        
    Example:
    --------
    >>> plt_model(y_true, y_pred, "LSTM", col_idx=0)
    """
    try:
        # Use global col_idx if not provided
        if col_idx is None:
            col_idx = globals().get('col_idx', 0)
            
        fig, ax = plt.subplots(figsize=(20, 10))
        ax.plot(pd.DataFrame(y_test_inverse)[[col_idx]], label='True Values')
        ax.plot(pd.DataFrame(yhat_inverse)[[col_idx]], label='Predicted Values')
        ax.set_xlabel('Date', fontweight='bold', fontsize=12)
        ax.set_ylabel('Value', fontweight='bold', fontsize=12)
        ax.set_title(f'Real vs. Predicted Values // MODEL: {model_name}')
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"plots/model_results_{model_name}.png")
        if show_plt:
            plt.show()

    except Exception as e:
        print(f"Error plotting model results: {str(e)}")

def extract_model_params(model_name):
    """
    Extract lookback and forecast parameters from model filename.
    
    Expected format: {code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{lr}initlr.keras
    
    Args:
        model_name (str): Model filename
        
    Returns:
        tuple: (lookback, forecast) or (None, None) if not found
    """
    # Pattern to match the model name format
    pattern = r'(\d+)fh_\d+ff_(\d+)lb_'
    
    match = re.search(pattern, model_name)
    if match:
        forecast = int(match.group(1))  # First group is forecast
        lookback = int(match.group(2))  # Second group is lookback
        return lookback, forecast
    else:
        print(f"Could not extract parameters from: {model_name}")
        return None, None


if __name__ == "__main__":
    # Set default values ...............................................................
    FORECAST=7     # number of future time steps the model will predict (predict horizon)
    LOOKBACK=7     # how many past time steps the model uses as input  (input seq length)

    #Transformer model .................................................................
    HEAD_SIZE=2              # dimensions of each attention head(If you have num_heads=2, 
                            #     attention 2 * head_size = 4 dimensions.)
    NUM_HEADS=2              # number attention head (More headsimprove ability to focus 
                            #     on different aspects but increase computational cost.)
    NUM_TRANSFORMER_BLOCKS=2 # n transformer layers (More blocks can improve the model's 
                            #     capacity but may lead to overfitting.)
    FF_DIM=8                 # dimensionality feed-forward layer  (more increases capacity but adds cost)

    # Multi-Layer Perceptron (MLP) Parameters ...........................................
    MLP_UNITS=32     # n neurons fully connected feed forward network (MLP)
    MLP_DROPOUT=0.25 # dropout rate MLP layers to prevent overfitting (% neurons randomly dropper durinhg training)

    # General Regularization Parameters .................................................
    DROPOUT=0.5

    # Optimization and Training Parameters ..............................................
    LEARNING_RATE = 0.001  # step size for gradient updates during training.( more speeds up but cause instabiliy)
    EPOCHS=100             # max number of times to train
    EARLY_STOP_PATIENCE=15 # if not improvement


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

        waves = {
        "Primera Onada": ("2020-03", "2020-06"),
        "Segona Onada": ("2020-10", "2020-12"),
        "Tercera Onada": ("2021-01", "2021-03"),
        "Quarta Onada": ("2021-04", "2021-06"),
        "Cinquena Onada": ("2021-07", "2021-09")
        }

        # Convertir a DataFrame per facilitar la representació
        df_waves = pd.DataFrame(waves).T.reset_index()
        df_waves.columns = ["Onada", "Inici", "Final"]
        df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
        df_waves["Final"] = pd.to_datetime(df_waves["Final"])

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

        plt_model(Y_test, predictions, model_name=MODEL_NAME, col_idx=0, show_plt=True)
        # Call the function with formatted dates
        plot_predictions_with_waves(Y_test, predictions, date_list, df_waves)

        #evaluate_model(model,MODEL_NAME, X_test, Y_test, date_list, df_waves, sliding_window=FORECAST)

        