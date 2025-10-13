"""
Training and Evaluation Module for Residual Multivariate Transformers
=====================================================================

This module contains functions for training and evaluating residual multivariate
transformer models, including model training, GPU memory management, and callbacks.
"""

import os
import pickle
import tensorflow as tf
from .config_residual_transformer import (
    DEFAULT_TRAINING_PARAMS, 
    DEFAULT_SAVE_PARAMS, 
    MEMORY_LOG_FILE
)


def setup_gpu_memory():
    """
    Configure GPU memory growth to prevent TensorFlow from allocating all GPU memory.
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
                print(f"GPU memory growth enabled for {gpu}")
        except RuntimeError as e:
            print(f"Error setting GPU memory growth: {e}")
    else:
        print("No GPUs found. Running on CPU.")


def train_given_model_and_data(model, X, Y, 
                               batch_size=None,
                               model_name=None, 
                               epochs=None,
                               validation_split=None,
                               shuffle=None,
                               patience=None,
                               save_history=None, 
                               save_model=None, 
                               save_memory=None, 
                               callbacks=None):
    """
    Train a model with given data and parameters.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The model to train
    X : np.ndarray
        Training input data
    Y : np.ndarray
        Training target data
    batch_size : int, optional
        Batch size for training (default from config)
    model_name : str, optional
        Name for saving the model (default: "testing")
    epochs : int, optional
        Number of training epochs (default from config)
    validation_split : float, optional
        Fraction of data to use for validation (default from config)
    shuffle : bool, optional
        Whether to shuffle training data (default from config)
    patience : int, optional
        Early stopping patience (default from config)
    save_history : bool, optional
        Whether to save training history (default from config)
    save_model : bool, optional
        Whether to save the trained model (default from config)
    save_memory : bool, optional
        Whether to log GPU memory usage (default from config)
    callbacks : list, optional
        List of Keras callbacks to use during training
        
    Returns:
    --------
    tf.keras.callbacks.History or None
        Training history if model was trained, None if model already exists
    """
    # Use default parameters if not provided
    if batch_size is None:
        batch_size = DEFAULT_TRAINING_PARAMS['batch_size']
    if epochs is None:
        epochs = DEFAULT_TRAINING_PARAMS['epochs']
    if validation_split is None:
        validation_split = DEFAULT_TRAINING_PARAMS['validation_split']
    if shuffle is None:
        shuffle = DEFAULT_TRAINING_PARAMS['shuffle']
    if patience is None:
        patience = DEFAULT_TRAINING_PARAMS['patience']
    if save_history is None:
        save_history = DEFAULT_SAVE_PARAMS['save_history']
    if save_model is None:
        save_model = DEFAULT_SAVE_PARAMS['save_model']
    if save_memory is None:
        save_memory = DEFAULT_SAVE_PARAMS['save_memory']
    
    if model_name is None:
        model_name = "testing"

    # Configure GPU memory growth if save_memory is enabled
    if save_memory:
        setup_gpu_memory()

    # Define default callbacks if none provided
    if callbacks is None:
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            patience=patience, 
            restore_best_weights=True
        )
        callbacks = [early_stop]

    # Check if model already exists
    if os.path.exists(f'{model_name}'):
        print(f"Model {model_name} already exists")
        return None

    # Train the model
    print(f"Starting training for model: {model_name}")
    print(f"Training parameters:")
    print(f"  - Batch size: {batch_size}")
    print(f"  - Epochs: {epochs}")
    print(f"  - Validation split: {validation_split}")
    print(f"  - Shuffle: {shuffle}")
    
    history = model.fit(
        x=X, 
        y=Y, 
        batch_size=batch_size,
        epochs=epochs, 
        shuffle=shuffle,
        validation_split=validation_split,
        callbacks=callbacks,
        verbose=1
    )

    # Save training history
    if save_history:
        history_filename = f'{model_name}_history.pkl'
        with open(history_filename, 'wb') as file_pi:
            pickle.dump(history.history, file_pi)
        print(f"Training history saved to: {history_filename}")
    
    # Save model
    if save_model and epochs > 1:
        model.save(model_name)
        print(f"Model saved to: {model_name}")
        
    # Log memory usage if enabled
    if save_memory:
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            with open(MEMORY_LOG_FILE, 'a') as resultcsv:
                resultcsv.write(f"{model_name},{memory_info['peak']},train\n")
            print(f"Current memory usage: {memory_info['current'] / (batch_size**2)} MB")
            print(f"Peak memory usage: {memory_info['peak'] / (batch_size**2)} MB")
        except Exception as e:
            print(f"Could not log memory usage: {e}")

    return history


def evaluate_model(model, X_test, Y_test, verbose=1):
    """
    Evaluate a trained model on test data.
    
    Parameters:
    -----------
    model : tf.keras.Model
        The trained model to evaluate
    X_test : np.ndarray
        Test input data
    Y_test : np.ndarray
        Test target data
    verbose : int, optional
        Verbosity level for evaluation (default: 1)
        
    Returns:
    --------
    tuple
        (loss, mae, mse) evaluation metrics
    """
    print("Evaluating model on test data...")
    results = model.evaluate(X_test, Y_test, verbose=verbose)
    
    if len(results) == 3:
        loss, mae, mse = results
        print(f"Test Results - Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
        return loss, mae, mse
    else:
        print(f"Test Results - Loss: {results[0]:.4f}")
        return results


def load_trained_model(model_path):
    """
    Load a trained model from disk.
    
    Parameters:
    -----------
    model_path : str
        Path to the saved model
        
    Returns:
    --------
    tf.keras.Model
        Loaded model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at: {model_path}")
    
    print(f"Loading model from: {model_path}")
    model = tf.keras.models.load_model(model_path, compile=True)
    print("Model loaded successfully")
    return model


def create_model_directories():
    """
    Create necessary directories for model storage.
    """
    directories = ['models', 'plots', 'logs']
    
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            print(f"Created directory: {directory}")
        else:
            print(f"Directory already exists: {directory}")


def get_callbacks(patience=None, monitor='val_loss', mode='min', 
                  restore_best_weights=True, additional_callbacks=None):
    """
    Create a list of callbacks for model training.
    
    Parameters:
    -----------
    patience : int, optional
        Early stopping patience (default from config)
    monitor : str, optional
        Metric to monitor for early stopping (default: 'val_loss')
    mode : str, optional
        Mode for monitoring metric (default: 'min')
    restore_best_weights : bool, optional
        Whether to restore best weights (default: True)
    additional_callbacks : list, optional
        Additional callbacks to include
        
    Returns:
    --------
    list
        List of Keras callbacks
    """
    if patience is None:
        patience = DEFAULT_TRAINING_PARAMS['patience']
    
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor=monitor,
            mode=mode,
            patience=patience,
            restore_best_weights=restore_best_weights
        )
    ]
    
    if additional_callbacks:
        callbacks.extend(additional_callbacks)
    
    return callbacks