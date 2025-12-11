"""
Model Architecture Module for Residual Multivariate Transformers
================================================================

This module contains model architectures for residual multivariate transformer models,
including transformer encoders, hybrid LSTM-transformer models, and learning rate schedules.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import math

from config.base_transformer_config import BaseTransformerConfig

default_config = BaseTransformerConfig()

def transformer_encoder(inputs, head_size=None, num_heads=None, ff_dim=None, dropout=None, activation_function='tanh'):
    """
    Transformer Encoder Block for processing sequential data.
    
    Parameters:
    -----------
    inputs : tf.Tensor
        Input tensor with shape (batch_size, sequence_length, features)
    head_size : int, optional
        Dimensions of each attention head (default from config)
    num_heads : int, optional
        Number of attention heads (default from config)
    ff_dim : int, optional
        Dimensionality of feed-forward layer (default from config)
    dropout : float, optional
        Dropout rate (default from config)
        
    Returns:
    --------
    tf.Tensor
        Processed tensor with same shape as input
    """
    # Use default parameters if not provided
    if head_size is None:
        head_size = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS['head_size']
    if num_heads is None:
        num_heads = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS['num_heads']
    if ff_dim is None:
        ff_dim = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS['ff_dim']
    if dropout is None:
        dropout = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS['dropout']
    
    # Multi-Head Self-Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)                  
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)  
    x = layers.Dropout(dropout)(x) 
    res = x + inputs  

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)                       
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x) 
    x = layers.Dropout(dropout)(x)                                         
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)          
    return x + res    


def hybrid_lstm_transformer_model(input_shape, forecast, 
                                  lstm_params=None, 
                                  transformer_params=None,
                                  activation_function='tanh'):
    """
    Build a hybrid LSTM-Transformer model for residual learning.
    
    Parameters:
    -----------
    input_shape : tuple
        Shape of input data (sequence_length, features)
    forecast : int
        Number of future time steps to predict
    lstm_params : dict, optional
        LSTM layer parameters (default from config)
    transformer_params : dict, optional
        Transformer encoder parameters (default from config)
        
    Returns:
    --------
    tf.keras.Model
        Compiled hybrid LSTM-Transformer model
    """
    # Use default parameters if not provided
    if lstm_params is None:
        lstm_params = default_config.DEFAULT_RESIDUAL_LSTM_PARAMS.copy()
    if transformer_params is None:
        transformer_params = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS.copy()


    
    input_layer = keras.Input(shape=input_shape)

    # LSTM Block
    x = layers.LSTM(
        lstm_params['units_1'], 
        return_sequences=lstm_params['return_sequences']
    )(input_layer)
    x = layers.Dropout(lstm_params['dropout'])(x)  # Dropout to reduce overfitting
    
    x = layers.LSTM(
        lstm_params['units_2'], 
        return_sequences=lstm_params['return_sequences']
    )(x)
    x = layers.Dropout(lstm_params['dropout'])(x)

    # Transformer Block
    x = transformer_encoder(
        x, 
        head_size=transformer_params['head_size'],
        num_heads=transformer_params['num_heads'],
        ff_dim=transformer_params['ff_dim'],
        activation_function=activation_function,
        dropout=transformer_params['dropout']
    )

    # GlobalAveragePooling1D Layer
    x = layers.GlobalAveragePooling1D()(x) # May be changed to Flatten() if needed
    # Output Layer
    outputs = layers.Dense(forecast, activation=activation_function)(x)

    # Reshape Outputs
    outputs = layers.Reshape((forecast, 1))(outputs)


    # Build Model
    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile Model
    model.compile(optimizer='adam', loss='mse')

    return model


class CustomCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom Cosine Learning Rate Decay with Warmup.
    
    This learning rate schedule linearly increases the learning rate during
    a warmup phase, then applies cosine decay for the remaining training steps.
    """
    
    def __init__(self, initial_lr=1e-4, max_lr=1e-3, min_lr=1e-5, 
                 warmup_steps=20, total_steps=50):
        """
        Initialize the CustomCosineDecay learning rate schedule.
        
        Parameters:
        -----------
        initial_lr : float
            Starting learning rate during warmup
        max_lr : float
            Maximum learning rate reached after warmup
        min_lr : float
            Final minimum learning rate after decay
        warmup_steps : int
            Number of warmup steps to reach max_lr
        total_steps : int
            Total number of training steps
        """
        self.initial_lr = initial_lr  # Starting learning rate (0.01)
        self.max_lr = max_lr          # Maximum learning rate (0.1)
        self.min_lr = min_lr          # Final learning rate (0.0001)
        self.warmup_steps = warmup_steps  # Number of warmup steps to reach max_lr
        self.total_steps = total_steps    # Total number of steps (decay after warmup)

    def __call__(self, step):
        """
        Calculate learning rate for given step.
        
        Parameters:
        -----------
        step : int
            Current training step
            
        Returns:
        --------
        float
            Learning rate for the current step
        """
        # Warm-up phase: linearly increase to max_lr
        if step < self.warmup_steps:
            return self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.warmup_steps)

        # Cosine decay phase after warmup
        decay_steps = self.total_steps - self.warmup_steps
        step_after_warmup = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_after_warmup / decay_steps))
        decayed = (self.max_lr - self.min_lr) * cosine_decay + self.min_lr
        return decayed

    def get_config(self):
        """Return the configuration of the learning rate schedule."""
        return {
            'initial_lr': self.initial_lr,
            'max_lr': self.max_lr,
            'min_lr': self.min_lr,
            'warmup_steps': self.warmup_steps,
            'total_steps': self.total_steps
        }