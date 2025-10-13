"""
Visualization Functions for Univariate Transformer
=================================================
Contains plotting functions for model results, predictions, and data exploration.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import os


def plot_example(df, title):
    """
    Plot example of 10 diagnoses from raw data.
    
    Args:
        df: DataFrame with diagnostic data
        title: Title for the plot
    """
    # PLOT RAW DATA (example 10 diags)
    dff = df.copy()
    dff["date"] = dff.index
    dff = dff[['J00', 'COV-19', 'date']]
    sns.set_theme(rc={'figure.figsize': (20, 8)})
    sns.lineplot(data=dff.replace('nan', float('nan')).melt(id_vars=['date']), 
                 x='date', y='value', hue='variable').set(title=title)


def plt_model(y_test_inverse, yhat_inverse, model_name, col_idx=None, show_plt=False):
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
    show_plt : bool
        Whether to display the plot
        
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
        
        os.makedirs("plots", exist_ok=True)
        fig.savefig(f"plots/model_results_{model_name}.png")
        if show_plt:
            plt.show()

    except Exception as e:
        print(f"Error plotting model results: {str(e)}")


def plot_predictions_with_waves(Y_test, predictions, date_list, df_waves):
    """
    Plot actual vs. predicted values with dates as x-labels, showing only 15 evenly spaced date labels.
    Also highlights COVID-19 pandemic waves with a red background.

    Args:
        Y_test (array): Actual target values
        predictions (array): Predicted values from the model
        date_list (list): List of datetime values for the x-axis
        df_waves (DataFrame): Contains pandemic waves' start and end dates
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
    plt.legend()

    # Apply only 15 labels to the x-axis
    plt.xticks([date_list[i] for i in indices], [date_labels[i] for i in indices], rotation=45)

    plt.grid()
    plt.show()


def plot_training_history(history, model_name, save_plot=True):
    """
    Plot training history including loss and metrics.
    
    Args:
        history: Keras training history object
        model_name: Name of the model for plot title and saving
        save_plot: Whether to save the plot to file
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot training & validation loss
    axes[0, 0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0, 0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0, 0].set_title('Model Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    
    # Plot training & validation MAE
    if 'mae' in history.history:
        axes[0, 1].plot(history.history['mae'], label='Training MAE')
        if 'val_mae' in history.history:
            axes[0, 1].plot(history.history['val_mae'], label='Validation MAE')
        axes[0, 1].set_title('Model MAE')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].legend()
    
    # Plot training & validation MSE
    if 'mse' in history.history:
        axes[1, 0].plot(history.history['mse'], label='Training MSE')
        if 'val_mse' in history.history:
            axes[1, 0].plot(history.history['val_mse'], label='Validation MSE')
        axes[1, 0].set_title('Model MSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('MSE')
        axes[1, 0].legend()
    
    # Plot learning rate if available
    if 'lr' in history.history:
        axes[1, 1].plot(history.history['lr'], label='Learning Rate')
        axes[1, 1].set_title('Learning Rate Schedule')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Learning Rate')
        axes[1, 1].set_yscale('log')
        axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_plot:
        os.makedirs("plots", exist_ok=True)
        plt.savefig(f"plots/training_history_{model_name}.png")
    
    plt.show()


def plot_model_comparison(models_results, metric='mae'):
    """
    Compare multiple models' performance.
    
    Args:
        models_results: Dictionary with model names as keys and results as values
        metric: Metric to compare ('mae', 'mse', 'loss')
    """
    model_names = list(models_results.keys())
    values = [results[metric] for results in models_results.values()]
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(model_names, values)
    plt.title(f'Model Comparison - {metric.upper()}')
    plt.xlabel('Models')
    plt.ylabel(metric.upper())
    plt.xticks(rotation=45)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{value:.4f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()


def plot_residuals_analysis(y_true, y_pred, model_name):
    """
    Plot residuals analysis including residual distribution and Q-Q plot.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        model_name: Name of the model
    """
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Residuals vs Predicted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Predicted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Predicted')
    
    # Histogram of residuals
    axes[0, 1].hist(residuals, bins=30, alpha=0.7)
    axes[0, 1].set_xlabel('Residuals')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Histogram of Residuals')
    
    # Time series of residuals
    axes[1, 0].plot(residuals)
    axes[1, 0].axhline(y=0, color='r', linestyle='--')
    axes[1, 0].set_xlabel('Time')
    axes[1, 0].set_ylabel('Residuals')
    axes[1, 0].set_title('Residuals Over Time')
    
    # Q-Q plot (simplified)
    from scipy import stats
    stats.probplot(residuals.flatten(), dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Q-Q Plot')
    
    plt.suptitle(f'Residuals Analysis - {model_name}')
    plt.tight_layout()
    
    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/residuals_analysis_{model_name}.png")
    plt.show()