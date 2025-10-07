
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
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)                  # to imputs to stabilize training
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)   # self attention to normalized input 
    x = layers.Dropout(dropout)(x) # (dropout to reduce overfitting)
    res = x + inputs               # attention output added to original inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)                       # again after resid connection
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(x) # point-wise convol.: Expands feature dim to ff_dim using tanh activ
    x = layers.Dropout(dropout)(x)                                         # dropout again
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)          # reduces feature dim back to match input size
    return x + res

def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, n_pred=1):
    inputs = keras.Input(shape=input_shape) #defines input tensor
    x = inputs 
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

if __name__ == "__main__":
    # Set default values ...............................................................
    FORECAST=7      # number of future time steps the model will predict (predict horizon)
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
    data_path = "C:/Users/34648/OneDrive - Generalitat de Catalunya/Escriptori/waikato/input data/longitudinalitat_DIAGNOSTICS_GROUPED.csv"
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

    # Create a new DataFrame with timestamp and target columns
    transformed_df = df[["date", code]].rename(columns={"date": "timestamp", code: code})
    transformed_df["timestamp"] = pd.to_datetime(transformed_df["timestamp"])
    transformed_df.reset_index(drop=True, inplace=True)

    # Apply Min-Max Scaling (0 to 1) only to the "target" column
    scaler = MinMaxScaler(feature_range=(0, 1))
    transformed_df[code] = scaler.fit_transform(transformed_df[[code]]) 

    # Apply function to split data
    train_df, test_df = split_train_test(transformed_df)
    print(train_df.info())
    print(test_df.info())


    start_time = time.perf_counter()
    input_directory = f'C:/Users/34648/OneDrive - Generalitat de Catalunya/Escriptori/waikato/diagnostic_data' # define data dir
    batchSize=16               # how many samples processing in parallel during training
    shuffle=False              # randomize order of training data??


    start_time = time.perf_counter()

    csvFile = f'{input_directory}/{code}_train_example.csv'
    #
    # prepare X and Y arrays
    X, Y = data_preparation.prepare_data(f'{input_directory}/{code}_train_example.csv', LOOKBACK, FORECAST, debug=True, univariate =True)

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


