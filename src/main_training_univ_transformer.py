"""
Main Training and Evaluation Script for Univariate Transformer
============================================================
This script orchestrates the entire training and evaluation pipeline.
"""

import pandas as pd
import numpy as np
import tensorflow as tf
import time
import os
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import Huber


from univariate_transformer import (
    build_model, CustomCosineDecay,
    train_given_model_and_data,
    evaluate_model_sliding_window,
    plt_model,  plot_predictions_with_waves, plot_example,
    extract_model_params, load_and_evaluate_models, setup_gpu_memory,create_model_directories, create_pandemic_waves_df, load_and_preprocess_data, evaluate_univ_transformer,
    default_config, create_config
)
    


# Import data preparation module
import sys
import os
# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'src' else current_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

import data_preparation


def main_univ_transformer(lookback, forecast, code,
                          activation_function = None,
                          covid_token = None, cutoff_date='2010-01-01', 
                          config = None, evaluate_model = False,
                          head_size = default_config.HEAD_SIZE, num_heads = default_config.NUM_HEADS,
                          ff_dim = default_config.FF_DIM, num_transformer_blocks = default_config.NUM_TRANSFORMER_BLOCKS,
                          mlp_units = default_config.MLP_UNITS,
                          positional_encoding = False,
                          dropout = default_config.DROPOUT,
                          learning_rate = default_config.LEARNING_RATE,
                          data_path = default_config.DATA_PATH
                          ):
    """Main function that orchestrates the training and evaluation pipeline."""
    
    # Use provided config or default configuration
    if config is None:
        config = default_config
    
    # Print configuration for transparency
    config.print_config(head_size=head_size, num_heads=num_heads, num_transformer_blocks=num_transformer_blocks,
                        ff_dim=ff_dim, lookback=lookback, forecast=forecast, learning_rate=learning_rate,
                        epochs=config.EPOCHS, batch_size=config.BATCH_SIZE, early_stop_patience=config.EARLY_STOP_PATIENCE,
                        target_code=code, date_cutoff=cutoff_date)
    
    # Extract configuration values for easier access
    FORECAST = forecast
    LOOKBACK = lookback
    

    HEAD_SIZE = head_size
    NUM_HEADS = num_heads
    NUM_TRANSFORMER_BLOCKS = num_transformer_blocks
    FF_DIM = ff_dim
    MLP_UNITS = mlp_units
    MLP_DROPOUT = dropout
    DROPOUT = dropout
    LEARNING_RATE = learning_rate
    EPOCHS = config.EPOCHS
    EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
    BATCH_SIZE = config.BATCH_SIZE
    SHUFFLE = config.SHUFFLE_DATA

    DATA_PATH = data_path
    MODEL_FOLDER = '../' + config.MODEL_DIR
    PLOTS_DIR = config.PLOTS_DIR
    if covid_token is None:
        COVID_TOKEN = config.COVID_TOKEN
    else:
        COVID_TOKEN = covid_token
    SAVE_TRAIN_HISTORY = config.SAVE_TRAIN_HISTORY
    if activation_function is None:
        ACTIVATION_FUNCTION = config.ACTIVATION_FUNCTION
    else:
        ACTIVATION_FUNCTION = activation_function

    # Setup
    setup_gpu_memory()
    create_model_directories()

    # Load and preprocess data using configuration
    
    '''df = load_and_preprocess_data(DATA_PATH, target_code=code)
    
    print("Loaded data shape:", df.shape)
    print("Date range:", df.index.min(), "to", df.index.max())

    # Plot example data
    plot_example(df, f"RAW DATA (example 10 diags) - {code}")'''

    
    input_directory = DATA_PATH
    batch_size = BATCH_SIZE
    shuffle = SHUFFLE   

    # TRAINING PHASE
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    start_time = time.perf_counter()

    # Prepare data for initial training
    X, Y = data_preparation.prepare_data(input_directory, code, LOOKBACK, FORECAST,covid_token=COVID_TOKEN, cutoff_date=cutoff_date,train = True, debug=True, univariate=True)
    
    finish_preparing = time.perf_counter()
    time_data_preparation = finish_preparing - start_time
    print(f"Data preparation time: {time_data_preparation:.2f} seconds")

    # Build model
    model = build_model(
        (LOOKBACK, X.shape[-1]),  # Input shape
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_units=[MLP_UNITS],
        mlp_dropout=MLP_DROPOUT,
        dropout=DROPOUT,
        n_pred=FORECAST,
        activation_function=ACTIVATION_FUNCTION,
        pos_encoding=positional_encoding,
    )

    model.summary()

    # Create learning rate scheduler using configuration
    lr_params = config.get_lr_schedule_params(learning_rate=LEARNING_RATE)
    scheduler = CustomCosineDecay(**lr_params)

    # Setup callbacks
    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(scheduler),
        tf.keras.callbacks.EarlyStopping(
            patience=EARLY_STOP_PATIENCE,
            monitor='val_loss',
            mode='min',
            restore_best_weights=True
        )
    ]

    # Compile model
    model.compile(loss='MAE', metrics=['mae', 'mse'], optimizer=Adam(clipnorm = 2.0, learning_rate=LEARNING_RATE, weight_decay=1e-4))

    # Train initial model
    MODEL_NAME = f'{code}_example_transformer_{FORECAST}fh_{FF_DIM}ff_{LOOKBACK}lb_{LEARNING_RATE}initlr.keras'
    train_given_model_and_data(
        model, X, Y, 
        batch_size=batch_size,
        model_name=MODEL_NAME, 
        epochs=EPOCHS, 
        save_model=True, 
        save_memory=False, 
        callbacks=callbacks, 
        save_history=SAVE_TRAIN_HISTORY
    )

    # EVALUATION PHASE
    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)
    
    # Evaluate all trained models using configuration
    
    print("Files in model folder:", os.listdir(MODEL_FOLDER))
    
    trained_models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.keras')]
    print("Trained models:", trained_models)

    # Create pandemic waves DataFrame
    df_waves = create_pandemic_waves_df()
    loss, mae, mse = None, None, None
    if evaluate_model:
        loss, mae, mse = evaluate_univ_transformer(MODEL_NAME, input_directory, code, cutoff_date=cutoff_date,covid_token=COVID_TOKEN, MODEL_FOLDER=MODEL_FOLDER, df_waves=df_waves)
        
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)

    return model, MODEL_NAME, loss, mae, mse


if __name__ == "__main__":
    # Option 1: Use default configuration
    main_univ_transformer(lookback=default_config.LOOKBACK, forecast=default_config.FORECAST, code=default_config.TARGET_CODE)