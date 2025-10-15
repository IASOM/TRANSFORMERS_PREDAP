"""
Main Training and Evaluation Script for Residual Multivariate diagnostic variables Transformers
==========================================================================

This script orchestrates the residual multivariate transformer training and 
evaluation pipeline by using the features extracted in the LMLR and G-causal phase.
This pipeline includes loading base models, computing residuals, training 
residual correction models, and evaluating results.
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error, mean_absolute_error


# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir) if os.path.basename(current_dir) != 'src' else current_dir
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)
import data_preparation
import evaluation_plot_utils


# Import from residual_multivariate_transformers module
from residual_multivariate_transformers import (
    # Configuration
    DEFAULT_FORECAST, DEFAULT_LOOKBACK, DEFAULT_LEARNING_RATE,
    DEFAULT_DATA_PATH, DEFAULT_MODEL_DIR,
    
    # Model Architecture
    hybrid_lstm_transformer_model, CustomCosineDecay,
    
    # Training and Evaluation
    train_given_model_and_data, setup_gpu_memory, create_model_directories,
    evaluate_model, load_trained_model, save_performance_results, load_performance_results, compare_model_performance,
    
    # Utilities
    split_train_test, learn_covariates, prepare_residual_data,
    create_pandemic_waves_df, load_and_preprocess_data,
    extract_model_params_from_filename, filter_diagnostics_covariates, prepare_base_model_data, load_base_model_transformer,
    
    # Visualization
    plot_residuals_analysis, plot_stepwise_errors_comparison,
    plot_predictions_with_pandemic_waves, plot_errors_over_time_with_waves,
    evaluate_error_significance_pandemic_waves
)


def main_train_diagnostic_residual_transformer(forecast, lookback, code = "T14", diagnostic_covariates_path = "BEST_features_NOSMOOTH.xlsx", predictions_train_corrected = None, predictions_test_corrected = None):
    """Main function that orchestrates the residual multivariate transformer pipeline."""
    
    # Configuration Parameters
    code = code
    forecast = forecast
    lookback = lookback
    ff_dim = 8
    learning_rate = DEFAULT_LEARNING_RATE
    diagnostic_covariates_path = diagnostic_covariates_path
    diagnostic_covariates_df = pd.read_excel(diagnostic_covariates_path, engine='openpyxl')
    diagnostic_covariates_list = list(diagnostic_covariates_df[diagnostic_covariates_df['LAG'] == forecast]['predictors'])[0].split(',')
    
    # Model naming
    base_model_name = f'{code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{learning_rate}initlr.keras'
    residual_model_name = f'{code}_DIAGNOSTIC_RESIDUALS_LEARNING_{forecast}fh_{ff_dim}ff_{lookback}lb_{learning_rate}initlr.keras'
    
    print("="*60)
    print("RESIDUAL MULTIVARIATE TRANSFORMER PIPELINE")
    print("="*60)
    print(f"Target Code: {code}")
    print(f"Forecast Horizon: {forecast}")
    print(f"Lookback Window: {lookback}")
    print(f"Base Model: {base_model_name}")
    print(f"Residual Model: {residual_model_name}")
    
    # Setup
    setup_gpu_memory()
    create_model_directories()
    
    # PHASE 1: LOAD BASE MODEL AND COMPUTE RESIDUALS
    print("\n" + "="*50)
    print("PHASE 1: LOADING BASE MODEL AND COMPUTING RESIDUALS")
    print("="*50)
    
    
    input_directory = DEFAULT_DATA_PATH
    
    # Load and prepare data for the base model
    print("Preparing data for base model...")
    
    Y_train, Y_test, X_train, X_test, date_list_train, date_list_test = prepare_base_model_data(
        input_directory, code, lookback, forecast
    )

    if predictions_train_corrected is None:
        predictions_train, predictions_test = load_base_model_transformer(
           X_train, X_test, DEFAULT_MODEL_DIR, base_model_name
        )
    else:
        predictions_train = predictions_train_corrected
        predictions_test = predictions_test_corrected

    
    
    print(f"Base model predictions - Test: {predictions_test.shape}, Train: {predictions_train.shape}")
    print(f"Actual values - Test: {Y_test.shape}, Train: {Y_train.shape}")
    
    # Compute residuals for train and test sets
    print("\nComputing residuals...")
    Y_train_residual = prepare_residual_data(predictions_train, Y_train)
    Y_test_residual = prepare_residual_data(predictions_test, Y_test)
    
    print(f"Residuals shape - Train: {Y_train_residual.shape}, Test: {Y_test_residual.shape}")
    
    # PHASE 2: PREPARE COVARIATE DATA FOR RESIDUAL MODEL
    print("\n" + "="*50)
    print("PHASE 2: PREPARING COVARIATE DATA FOR RESIDUAL MODEL")
    print("="*50)
    
    # Load and split the original data for covariate extraction
    train_split, test_split = split_train_test(pd.read_csv(input_directory), split_ratio=0.8, init_date='2010-01-01')
    train_split = filter_diagnostics_covariates(train_split, diagnostic_covariates_list)
    test_split = filter_diagnostics_covariates(test_split, diagnostic_covariates_list)
    # Learn covariates from training data
    df_processed = train_split
    
    # Generate rolling sequences with covariates for training
    print("Generating sequences with diagnostics covariates for training...")
    '''X_train_covs = data_preparation.generate_rolling_sequences_covariates(
        df_processed, lookback, forecast, predictions_train
    )'''
    X_train_covs, _ = data_preparation.prepare_data(input_directory, code, lookback, forecast,relevant_feature_cols=diagnostic_covariates_list, train = True, univariate = False)
    
    print(f"Training covariates shape: {X_train_covs.shape}")
    print(f"Expected shape: (num_samples, {lookback}, num_features)")
    
    # PHASE 3: TRAIN RESIDUAL CORRECTION MODEL
    print("\n" + "="*50)
    print("PHASE 3: TRAINING RESIDUAL CORRECTION MODEL")
    print("="*50)
    
    # Define Hybrid LSTM + Transformer Model for residuals
    print("Building residual correction model...")
    residual_model = hybrid_lstm_transformer_model(
        (lookback, X_train_covs.shape[2]), 
        forecast
    )
    residual_model.summary()
    
    # Train the residual model
    print("Training residual correction model...")
    train_given_model_and_data(
        residual_model, X_train_covs, Y_train_residual,
        batch_size=32,
        model_name=residual_model_name,
        epochs=100,
        save_model=True,
        save_memory=False,
        callbacks=None
    )

    predicted_residuals_train = residual_model.predict(X_train_covs, verbose=1)
    predicted_residuals_train = np.squeeze(predicted_residuals_train, axis=-1)
    
    predictions_train_corrected = predictions_train + predicted_residuals_train
    
    # PHASE 4: EVALUATE RESIDUAL CORRECTION MODEL
    print("\n" + "="*50)
    print("PHASE 4: EVALUATING RESIDUAL CORRECTION MODEL")
    print("="*50)
    
    # Prepare test data with covariates
    print("Preparing test data with covariates...")
    df_test_processed = test_split
    
    # Generate rolling sequences for test data
    '''X_test_covs = data_preparation.generate_rolling_sequences_covariates(
        df_test_processed, lookback, forecast, predictions_test
    )'''
    

    X_test_covs, _ = data_preparation.prepare_data(input_directory, code, lookback, forecast, relevant_feature_cols=diagnostic_covariates_list, train = False, univariate = False)
    
    print(f"Test covariates shape: {X_test_covs.shape}")
    
    # Load the trained residual model (in case it was saved and reloaded)
    if os.path.exists(residual_model_name):
        residual_model = load_trained_model(residual_model_name)
    
    # Predict residuals for the test set
    print("Predicting residuals for test set...")
    predicted_residuals = residual_model.predict(X_test_covs, verbose=1)
    predicted_residuals = np.squeeze(predicted_residuals, axis=-1) if predicted_residuals.shape[-1] == 1 else predicted_residuals
    
    # Correct the original forecast
    print("Computing corrected forecasts...")
    corrected_forecast = predictions_test + predicted_residuals
    
    print(f"Corrected forecast shape: {corrected_forecast.shape}")
    
    # PHASE 5: VISUALIZATION AND ANALYSIS
    print("\n" + "="*50)
    print("PHASE 5: VISUALIZATION AND ANALYSIS")
    print("="*50)
    
    # Plot stepwise errors comparison
    print("Plotting stepwise errors comparison...")
    plot_stepwise_errors_comparison(Y_test, predictions_test, corrected_forecast, "Residual Correction", model_name = residual_model_name)
    
    # Plot residuals analysis
    print("Plotting residuals analysis...")
    plot_residuals_analysis(predictions_test, corrected_forecast, Y_test, "Residual Correction", model_name = residual_model_name)
    
    # Create pandemic waves DataFrame
    df_waves = create_pandemic_waves_df()
    
    # Prepare data for plotting (average across forecast horizon if needed)
    predictions_to_plot = predictions_test.mean(axis=1) if len(predictions_test.shape) > 2 else predictions_test
    corrected_to_plot = corrected_forecast.mean(axis=1) if len(corrected_forecast.shape) > 2 else corrected_forecast
    Y_test_to_plot = Y_test.mean(axis=1) if len(Y_test.shape) > 2 else Y_test
    
    # Plot predictions with pandemic waves
    print("Plotting predictions with pandemic waves...")
    plot_predictions_with_pandemic_waves(
        Y_test_to_plot, predictions_to_plot, date_list_test, df_waves, model_name = residual_model_name
    )
    
    plot_predictions_with_pandemic_waves(
        Y_test_to_plot, corrected_to_plot, date_list_test, df_waves, model_name = residual_model_name
    )
    
    # Plot errors over time with waves
    print("Plotting errors over time with pandemic waves...")
    plot_errors_over_time_with_waves(
        Y_test_to_plot, predictions_to_plot, date_list_test, df_waves
    )
    
    plot_errors_over_time_with_waves(
        Y_test_to_plot, corrected_to_plot, date_list_test, df_waves
    )
    
    # Evaluate error significance during pandemic waves
    print("Evaluating error significance during pandemic waves...")
    print("Original predictions:")
    evaluate_error_significance_pandemic_waves(
        Y_test_to_plot, predictions_to_plot, date_list_test, df_waves
    )
    
    print("Corrected predictions:")
    evaluate_error_significance_pandemic_waves(
        Y_test_to_plot, corrected_to_plot, date_list_test, df_waves
    )
    
    # PHASE 6: PERFORMANCE SUMMARY
    print("\n" + "="*50)
    print("PHASE 6: PERFORMANCE SUMMARY")
    print("="*50)
    
    # Calculate metrics
    
    # Original model metrics
    original_mae = mean_absolute_error(Y_test_to_plot, predictions_to_plot)
    original_mse = mean_squared_error(Y_test_to_plot, predictions_to_plot)
    original_rmse = np.sqrt(original_mse)
    
    # Corrected model metrics
    corrected_mae = mean_absolute_error(Y_test_to_plot, corrected_to_plot)
    corrected_mse = mean_squared_error(Y_test_to_plot, corrected_to_plot)
    corrected_rmse = np.sqrt(corrected_mse)
    
    print("PERFORMANCE COMPARISON:")
    print("-" * 40)
    print(f"Original Model:")
    print(f"  MAE:  {original_mae:.6f}")
    print(f"  MSE:  {original_mse:.6f}")
    print(f"  RMSE: {original_rmse:.6f}")
    print()
    print(f"Residual Corrected Model:")
    print(f"  MAE:  {corrected_mae:.6f}")
    print(f"  MSE:  {corrected_mse:.6f}")
    print(f"  RMSE: {corrected_rmse:.6f}")
    print()
    print(f"IMPROVEMENT:")
    print(f"  MAE:  {((original_mae - corrected_mae) / original_mae * 100):+.2f}%")
    print(f"  MSE:  {((original_mse - corrected_mse) / original_mse * 100):+.2f}%")
    print(f"  RMSE: {((original_rmse - corrected_rmse) / original_rmse * 100):+.2f}%")
    
    # Save performance results to JSON
    save_performance_results(
        model_name=residual_model_name,
        original_mae=original_mae,
        original_mse=original_mse, 
        original_rmse=original_rmse,
        corrected_mae=corrected_mae,
        corrected_mse=corrected_mse,
        corrected_rmse=corrected_rmse,
        forecast=forecast,
        lookback=lookback,
        code=code
    )

    compare_model_performance()
    
    print("\n" + "="*50)
    print("RESIDUAL MULTIVARIATE TRANSFORMER PIPELINE COMPLETE")
    print("="*50)
    predictions_test_corrected = corrected_forecast 

    return predictions_train_corrected, predictions_test_corrected


if __name__ == "__main__":
    # Run the main training and evaluation pipeline
    predictions_train_corrected, predictions_test_corrected = main_train_diagnostic_residual_transformer(forecast=DEFAULT_FORECAST, lookback=DEFAULT_LOOKBACK, code="T14")
    
    # Optional: View all previous results (uncomment to use)
    # print("\n" + "="*60)
    # print("VIEWING ALL PERFORMANCE RESULTS")
    # print("="*60)
    # load_performance_results()
    # compare_model_performance(metric="MAE")