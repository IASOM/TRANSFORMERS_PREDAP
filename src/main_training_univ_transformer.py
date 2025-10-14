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


def main_univ_transformer(lookback, forecast, code, config = None, evaluate_model = False):
    """Main function that orchestrates the training and evaluation pipeline."""
    
    # Use provided config or default configuration
    if config is None:
        config = default_config
    
    # Print configuration for transparency
    config.print_config()
    
    # Extract configuration values for easier access
    FORECAST = forecast
    LOOKBACK = lookback
    LOOKBACK_LIST = config.LOOKBACK_LIST
    FORECAST_LIST = config.FORECAST_LIST

    HEAD_SIZE = config.HEAD_SIZE
    NUM_HEADS = config.NUM_HEADS
    NUM_TRANSFORMER_BLOCKS = config.NUM_TRANSFORMER_BLOCKS
    FF_DIM = config.FF_DIM
    MLP_UNITS = config.MLP_UNITS
    MLP_DROPOUT = config.MLP_DROPOUT
    DROPOUT = config.DROPOUT
    LEARNING_RATE = config.LEARNING_RATE
    EPOCHS = config.EPOCHS
    EARLY_STOP_PATIENCE = config.EARLY_STOP_PATIENCE
    BATCH_SIZE = config.BATCH_SIZE
    SHUFFLE = config.SHUFFLE_DATA

    DATA_PATH = config.DATA_PATH
    TARGET_CODE = config.TARGET_CODE
    MODEL_FOLDER = config.MODEL_DIR
    PLOTS_DIR = config.PLOTS_DIR

    # Setup
    setup_gpu_memory()
    create_model_directories()

    # Load and preprocess data using configuration
    
    df = load_and_preprocess_data(DATA_PATH, target_code=TARGET_CODE)
    
    print("Loaded data shape:", df.shape)
    print("Date range:", df.index.min(), "to", df.index.max())

    # Plot example data
    plot_example(df, f"RAW DATA (example 10 diags) - {TARGET_CODE}")

    code = TARGET_CODE
    input_directory = DATA_PATH
    batch_size = BATCH_SIZE
    shuffle = SHUFFLE   

    # TRAINING PHASE
    print("\n" + "="*50)
    print("TRAINING PHASE")
    print("="*50)
    
    start_time = time.perf_counter()

    # Prepare data for initial training
    X, Y = data_preparation.prepare_data(input_directory, code, LOOKBACK, FORECAST, debug=True, univariate=True)
    
    finish_preparing = time.perf_counter()
    time_data_preparation = finish_preparing - start_time
    print(f"Data preparation time: {time_data_preparation:.2f} seconds")

    # Build model
    model = build_model(
        (LOOKBACK, 1),
        head_size=HEAD_SIZE,
        num_heads=NUM_HEADS,
        ff_dim=FF_DIM,
        num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
        mlp_units=[MLP_UNITS],
        mlp_dropout=MLP_DROPOUT,
        dropout=DROPOUT,
        n_pred=FORECAST
    )

    model.summary()

    # Create learning rate scheduler using configuration
    lr_params = config.get_lr_schedule_params()
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
    model.compile(loss='MSE', metrics=['mae', 'mse'], optimizer=Adam())

    # Train initial model
    MODEL_NAME = f'models/{code}_example_transformer_{FORECAST}fh_{FF_DIM}ff_{LOOKBACK}lb_{LEARNING_RATE}initlr.keras'
    train_given_model_and_data(
        model, X, Y, 
        batch_size=batch_size,
        model_name=MODEL_NAME, 
        epochs=EPOCHS, 
        save_model=True, 
        save_memory=False, 
        callbacks=callbacks
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

    if evaluate_model:
        evaluate_univ_transformer(MODEL_NAME, input_directory, code, MODEL_FOLDER=MODEL_FOLDER, df_waves=df_waves)
        
    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)


if __name__ == "__main__":
    # Option 1: Use default configuration
    main_univ_transformer(lookback=default_config.LOOKBACK, forecast=default_config.FORECAST, code=default_config.TARGET_CODE)