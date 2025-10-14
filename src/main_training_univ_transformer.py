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
    extract_model_params, load_and_evaluate_models, setup_gpu_memory,create_model_directories, create_pandemic_waves_df, load_and_preprocess_data,
    default_config, create_config
)
    


# Import data preparation module
import sys
sys.path.append('..')  # Add parent directory to path
import data_preparation


def main_univ_transformer():
    """Main function that orchestrates the training and evaluation pipeline."""
    
    # Use provided config or default configuration
    if config is None:
        config = default_config
    
    # Print configuration for transparency
    config.print_config()
    
    # Extract configuration values for easier access
    FORECAST = config.FORECAST
    LOOKBACK = config.LOOKBACK
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
    MODEL_DIR = config.MODEL_DIR
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

    # HYPERPARAMETER SWEEP
    print("\n" + "="*50)
    print("HYPERPARAMETER SWEEP")
    print("="*50)
    
   
    # Train different models for different lookback and forecast horizons
    for lb in LOOKBACK_LIST:
        if lb <= 1:  # Skip problematic sequence lengths
            continue
            
        for fh in FORECAST_LIST:
            print(f"\nTraining model: lookback={lb}, forecast={fh}")
            
            # Prepare data
            X, Y = data_preparation.prepare_data(input_directory, code, lb, fh, debug=True, univariate=True)
            
            start_time = time.perf_counter()
            
            # Build model
            model = build_model(
                (lb, 1),
                head_size=HEAD_SIZE,
                num_heads=NUM_HEADS,
                ff_dim=FF_DIM,
                num_transformer_blocks=NUM_TRANSFORMER_BLOCKS,
                mlp_units=[MLP_UNITS],
                mlp_dropout=MLP_DROPOUT,
                dropout=DROPOUT,
                n_pred=fh
            )
            
            # Setup scheduler and callbacks using configuration
            lr_params = config.get_lr_schedule_params()
            scheduler = CustomCosineDecay(**lr_params)
            
            callbacks = [
                tf.keras.callbacks.LearningRateScheduler(scheduler),
                tf.keras.callbacks.EarlyStopping(
                    patience=EARLY_STOP_PATIENCE,
                    monitor='val_loss',
                    mode='min',
                    restore_best_weights=True
                )
            ]

            model.compile(loss='MSE', metrics=['mae', 'mse'], optimizer=Adam())
            
            # Train model
            MODEL_NAME = f'models/{code}_example_transformer_{fh}fh_{FF_DIM}ff_{lb}lb_{LEARNING_RATE}initlr.keras'
            train_given_model_and_data(
                model, X, Y, 
                batch_size=batch_size,
                model_name=MODEL_NAME, 
                epochs=EPOCHS, 
                save_model=True, 
                save_memory=False, 
                callbacks=callbacks
            )

            finish_time = time.perf_counter()
            training_time = finish_time - start_time
            print(f"Model training time: {training_time:.2f} seconds")

    # EVALUATION PHASE
    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)
    
    # Evaluate all trained models using configuration
    MODEL_FOLDER = config.MODEL_DIR
    print("Files in model folder:", os.listdir(MODEL_FOLDER))
    
    trained_models = [f for f in os.listdir(MODEL_FOLDER) if f.endswith('.keras')]
    print("Trained models:", trained_models)

    # Create pandemic waves DataFrame
    df_waves = create_pandemic_waves_df()

    for model_name in trained_models:
        print(f"\n--- Evaluating model: {model_name} ---")
        
        # Extract parameters from filename
        lookback, forecast = extract_model_params(model_name)
        
        if lookback is None or forecast is None:
            print(f"Skipping {model_name} - could not extract parameters")
            continue
            
        # Load model
        model_path = os.path.join(MODEL_FOLDER, model_name)
        model = tf.keras.models.load_model(model_path, compile=True)
        
        # Prepare test data
        X_test, Y_test = data_preparation.prepare_data(
            input_directory, code, lookback, forecast, train=False, debug=True, univariate=True
        )
        date_list = data_preparation.extract_dates(input_directory, code, lookback, forecast, train=False)
        
        # Evaluate model
        loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)
        print(f"Test Results - Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
        
        # Get predictions
        predictions = model.predict(X_test, verbose=0)
        print("Predicted values shape:", predictions.shape)

        # Generate plots
        model_display_name = model_name.replace('.keras', '')
        plt_model(Y_test, predictions, model_name=model_display_name, col_idx=0, show_plt=False)
        
        # Plot predictions with pandemic waves
        plot_predictions_with_waves(Y_test, predictions, date_list, df_waves, model_display_name)
        
        # Sliding window evaluation (optional)
        # evaluate_model_sliding_window(model, model_display_name, X_test, Y_test, date_list, df_waves, sliding_window=forecast)

    print("\n" + "="*50)
    print("EVALUATION COMPLETE")
    print("="*50)


if __name__ == "__main__":
    # Option 1: Use default configuration
    main_univ_transformer()