"""
Visualization Module for Residual Multivariate Transformers
===========================================================

This module contains visualization and plotting functions for residual multivariate
transformer models, including prediction plots, error analysis, and pandemic wave visualization.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import os

# Add the src directory to path for module imports
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(os.path.dirname(current_dir)) if 'residual_multivariate_transformers' in current_dir else os.path.dirname(current_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

try:
    import evaluation_plot_utils
except ImportError:
    print("Warning: evaluation_plot_utils module not found. Some plotting functions may not work.")
    evaluation_plot_utils = None

from .config_residual_transformer import PANDEMIC_WAVES


def plot_residuals_analysis(original_predictions, corrected_predictions, actual_values, title_prefix="", show_plt=False, model_name = "Model"):
    """
    Plot analysis comparing original predictions, corrected predictions, and actual values.
    
    Parameters:
    -----------
    original_predictions : np.ndarray
        Original model predictions
    corrected_predictions : np.ndarray
        Residual-corrected predictions
    actual_values : np.ndarray
        Actual target values
    title_prefix : str, optional
        Prefix for plot titles
    """
    # Ensure arrays are 1D for plotting
    if len(original_predictions.shape) > 1:
        original_predictions = original_predictions.mean(axis=1)
    if len(corrected_predictions.shape) > 1:
        corrected_predictions = corrected_predictions.mean(axis=1)
    if len(actual_values.shape) > 1:
        actual_values = actual_values.mean(axis=1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'{title_prefix} Residual Correction Analysis', fontsize=16)
    
    # Time series comparison
    axes[0, 0].plot(actual_values, label='Actual', alpha=0.8)
    axes[0, 0].plot(original_predictions, label='Original Predictions', alpha=0.8)
    axes[0, 0].plot(corrected_predictions, label='Corrected Predictions', alpha=0.8)
    axes[0, 0].set_title('Time Series Comparison')
    axes[0, 0].set_xlabel('Time Steps')
    axes[0, 0].set_ylabel('Values')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Residuals before and after correction
    original_residuals = actual_values - original_predictions
    corrected_residuals = actual_values - corrected_predictions
    
    axes[0, 1].plot(original_residuals, label='Original Residuals', alpha=0.7)
    axes[0, 1].plot(corrected_residuals, label='Corrected Residuals', alpha=0.7)
    axes[0, 1].set_title('Residuals Comparison')
    axes[0, 1].set_xlabel('Time Steps')
    axes[0, 1].set_ylabel('Residuals')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    
    # Scatter plot: Actual vs Predictions
    axes[1, 0].scatter(actual_values, original_predictions, alpha=0.6, label='Original')
    axes[1, 0].scatter(actual_values, corrected_predictions, alpha=0.6, label='Corrected')
    
    # Perfect prediction line
    min_val = min(actual_values.min(), original_predictions.min(), corrected_predictions.min())
    max_val = max(actual_values.max(), original_predictions.max(), corrected_predictions.max())
    axes[1, 0].plot([min_val, max_val], [min_val, max_val], 'r--', alpha=0.8, label='Perfect Prediction')
    
    axes[1, 0].set_title('Actual vs Predicted')
    axes[1, 0].set_xlabel('Actual Values')
    axes[1, 0].set_ylabel('Predicted Values')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Error distribution
    original_errors = np.abs(original_residuals)
    corrected_errors = np.abs(corrected_residuals)
    
    axes[1, 1].hist(original_errors, bins=30, alpha=0.7, label=f'Original (MAE: {np.mean(original_errors):.4f})')
    axes[1, 1].hist(corrected_errors, bins=30, alpha=0.7, label=f'Corrected (MAE: {np.mean(corrected_errors):.4f})')
    axes[1, 1].set_title('Error Distribution')
    axes[1, 1].set_xlabel('Absolute Error')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("plots_residual_transformers", exist_ok=True)
    plt.savefig(f"plots_residual_transformers/residuals_analysis_{model_name}.png")
    if show_plt:
        plt.show()


def plot_stepwise_errors_comparison(Y_test, original_predictions, corrected_predictions, title_prefix="", model_name = "Model"):
    """
    Plot stepwise errors for original and corrected predictions.
    
    Parameters:
    -----------
    Y_test : np.ndarray
        Test target values
    original_predictions : np.ndarray
        Original model predictions
    corrected_predictions : np.ndarray
        Residual-corrected predictions
    title_prefix : str, optional
        Prefix for plot titles
    """
    if evaluation_plot_utils is None:
        print("Warning: evaluation_plot_utils not available. Cannot plot stepwise errors.")
        return
    
    print(f"{title_prefix} - Original Predictions Stepwise Errors:")
    evaluation_plot_utils.plot_stepwise_errors(Y_test, original_predictions, model_name = model_name)
    
    print(f"{title_prefix} - Corrected Predictions Stepwise Errors:")
    evaluation_plot_utils.plot_stepwise_errors(Y_test, corrected_predictions, model_name = model_name)


def plot_predictions_with_pandemic_waves(Y_test, predictions, date_list, df_waves=None, title="Predictions with Pandemic Waves", model_name = "Model"):
    """
    Plot predictions overlaid with pandemic wave periods.
    
    Parameters:
    -----------
    Y_test : np.ndarray
        Test target values
    predictions : np.ndarray
        Model predictions
    date_list : list
        List of dates corresponding to predictions
    df_waves : pd.DataFrame, optional
        DataFrame with pandemic wave information
    title : str, optional
        Plot title
    """
    if evaluation_plot_utils is None:
        print("Warning: evaluation_plot_utils not available. Cannot plot with pandemic waves.")
        return
    
    # Create waves DataFrame if not provided
    if df_waves is None:
        df_waves = create_pandemic_waves_df()
    
    # Ensure predictions are 1D for plotting
    if len(predictions.shape) > 1:
        predictions_to_plot = predictions.mean(axis=1)
    else:
        predictions_to_plot = predictions
        
    if len(Y_test.shape) > 1:
        Y_test_to_plot = Y_test.mean(axis=1)
    else:
        Y_test_to_plot = Y_test
    
    print(f"Plotting {title}")
    evaluation_plot_utils.plot_predictions_with_waves(
        Y_test_to_plot, predictions_to_plot, date_list, df_waves, waves=False
    )


def plot_errors_over_time_with_waves(Y_test, predictions, date_list, df_waves=None):
    """
    Plot prediction errors over time with pandemic wave periods.
    
    Parameters:
    -----------
    Y_test : np.ndarray
        Test target values
    predictions : np.ndarray
        Model predictions
    date_list : list
        List of dates corresponding to predictions
    df_waves : pd.DataFrame, optional
        DataFrame with pandemic wave information
    """
    if evaluation_plot_utils is None:
        print("Warning: evaluation_plot_utils not available. Cannot plot errors over time.")
        return
    
    # Create waves DataFrame if not provided
    if df_waves is None:
        df_waves = create_pandemic_waves_df()
    
    # Ensure predictions are 1D for plotting
    if len(predictions.shape) > 1:
        predictions_to_plot = predictions.mean(axis=1)
    else:
        predictions_to_plot = predictions
        
    if len(Y_test.shape) > 1:
        Y_test_to_plot = Y_test.mean(axis=1)
    else:
        Y_test_to_plot = Y_test
    
    print("Plotting errors over time with pandemic waves")
    evaluation_plot_utils.plot_errors_over_time_with_waves(
        Y_test_to_plot, predictions_to_plot, date_list, df_waves, waves=False
    )


def evaluate_error_significance_pandemic_waves(Y_test, predictions, date_list, df_waves=None, model_name = "Model"):
    """
    Evaluate error significance during pandemic waves.
    
    Parameters:
    -----------
    Y_test : np.ndarray
        Test target values
    predictions : np.ndarray
        Model predictions
    date_list : list
        List of dates corresponding to predictions
    df_waves : pd.DataFrame, optional
        DataFrame with pandemic wave information
    """
    if evaluation_plot_utils is None:
        print("Warning: evaluation_plot_utils not available. Cannot evaluate error significance.")
        return
    
    # Create waves DataFrame if not provided
    if df_waves is None:
        df_waves = create_pandemic_waves_df()
    
    # Ensure predictions are 1D for analysis
    if len(predictions.shape) > 1:
        predictions_to_plot = predictions.mean(axis=1)
    else:
        predictions_to_plot = predictions
        
    if len(Y_test.shape) > 1:
        Y_test_to_plot = Y_test.mean(axis=1)
    else:
        Y_test_to_plot = Y_test
    
    print("Evaluating error significance during pandemic waves")
    evaluation_plot_utils.evaluate_error_significance_pandemic_waves(
        Y_test_to_plot, predictions_to_plot, date_list, df_waves
    )


def create_pandemic_waves_df():
    """
    Create a DataFrame with pandemic wave information from config.
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with pandemic wave periods
    """
    df_waves = pd.DataFrame(PANDEMIC_WAVES).T.reset_index()
    df_waves.columns = ["Onada", "Inici", "Final"]
    df_waves["Inici"] = pd.to_datetime(df_waves["Inici"])
    df_waves["Final"] = pd.to_datetime(df_waves["Final"])
    
    return df_waves


def plot_training_history(history, model_name="Model", show_plt=False):
    """
    Plot training history (loss and metrics over epochs).
    
    Parameters:
    -----------
    history : tf.keras.callbacks.History or dict
        Training history object or history dictionary
    model_name : str, optional
        Model name for plot title
    """
    if hasattr(history, 'history'):
        hist_dict = history.history
    else:
        hist_dict = history
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f'{model_name} Training History', fontsize=14)
    
    # Plot training loss
    if 'loss' in hist_dict:
        axes[0].plot(hist_dict['loss'], label='Training Loss')
    if 'val_loss' in hist_dict:
        axes[0].plot(hist_dict['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot metrics (MAE if available)
    if 'mae' in hist_dict:
        axes[1].plot(hist_dict['mae'], label='Training MAE')
    if 'val_mae' in hist_dict:
        axes[1].plot(hist_dict['val_mae'], label='Validation MAE')
    elif 'mse' in hist_dict:
        axes[1].plot(hist_dict['mse'], label='Training MSE')
        if 'val_mse' in hist_dict:
            axes[1].plot(hist_dict['val_mse'], label='Validation MSE')
    
    axes[1].set_title('Model Metrics')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Metric Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    os.makedirs("plots_residual_transformers", exist_ok=True)
    plt.savefig(f"plots_residual_transformers/training_history_{model_name}.png")
    if show_plt:
        plt.show()


def plot_model_comparison(models_results, metric='mae', title="Model Comparison", model_name = "Model", plt_show=False):
    """
    Plot comparison of multiple models' performance.
    
    Parameters:
    -----------
    models_results : dict
        Dictionary with model names as keys and results as values
    metric : str, optional
        Metric to compare ('mae', 'mse', 'loss')
    title : str, optional
        Plot title
    """
    model_names = list(models_results.keys())
    values = [models_results[name][metric] for name in model_names]
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(model_names, values, alpha=0.7)
    plt.title(f'{title} - {metric.upper()}')
    plt.xlabel('Model')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(values)*0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.grid(True, alpha=0.3)
    os.makedirs("plots_residual_transformers", exist_ok=True)
    plt.savefig(f"plots_residual_transformers/model_comparison_{metric}_{model_name}.png")
    if plt_show:
        plt.show()