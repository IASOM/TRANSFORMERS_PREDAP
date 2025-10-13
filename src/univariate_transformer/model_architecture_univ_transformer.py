"""
Model Architecture for Univariate Transformer
==============================================
Contains transformer encoder, model building functions, and learning rate schedules.
"""

import tensorflow as tf
import math
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam


def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    """
    Transformer encoder block with multi-head attention and feed-forward layers.
    
    Args:
        inputs: Input tensor (batch_size, sequence_length, features)
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        dropout: Dropout rate
        
    Returns:
        Encoded tensor with residual connections
    """
    d_model = max(head_size * num_heads, 8)
    x = layers.Dense(d_model, activation="relu")(inputs)
    
    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)  # to inputs to stabilize training
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x)  # self attention to normalized input 
    x = layers.Dropout(dropout)(x)  # (dropout to reduce overfitting)
    res = x + inputs  # attention output added to original inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)  # again after resid connection
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="tanh")(x)  # point-wise convol.: Expands feature dim to ff_dim using tanh activ
    x = layers.Dropout(dropout)(x)  # dropout again
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)  # reduces feature dim back to match input size
    x = x + res
    
    return x


def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, dropout=0, mlp_dropout=0, n_pred=1):
    """
    Build complete transformer model for univariate time series forecasting.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        num_transformer_blocks: Number of transformer encoder blocks
        mlp_units: List of MLP layer dimensions
        dropout: Dropout rate for transformer layers
        mlp_dropout: Dropout rate for MLP layers
        n_pred: Number of prediction steps
        
    Returns:
        Compiled Keras model
    """
    inputs = keras.Input(shape=input_shape)  # defines input tensor
    
    x = inputs  # initial input
    
    for _ in range(num_transformer_blocks):  # apply num_transformer_blocks transformer encoder layers seq.
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)  # uses previous defined trans_encoder layer

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)  # reduces seq dimension (timesteps) averaging for each feature channel
    for dim in mlp_units:  # multi layer perceptron (dropout to avoid overfitting)
        x = layers.Dense(dim, activation="tanh")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_pred)(x)
    return keras.Model(inputs, outputs)


class CustomCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom cosine decay learning rate schedule with warmup phase.
    Defaults to 50 epochs total and 20 warmup steps.
    """
    
    def __init__(self, initial_lr=1e-4, max_lr=1e-3, min_lr=1e-5, warmup_steps=20, total_steps=50):
        """
        Initialize the learning rate schedule.
        
        Args:
            initial_lr: Starting learning rate (0.01)
            max_lr: Maximum learning rate (0.1)
            min_lr: Final learning rate (0.0001)
            warmup_steps: Number of warmup steps to reach max_lr
            total_steps: Total number of steps (decay after warmup)
        """
        self.initial_lr = initial_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

    def __call__(self, step):
        """Apply the learning rate schedule."""
        # Warm-up phase: linearly increase to max_lr
        if step < self.warmup_steps:
            return self.initial_lr + (self.max_lr - self.initial_lr) * (step / self.warmup_steps)

        # Cosine decay phase after warmup
        decay_steps = self.total_steps - self.warmup_steps
        step_after_warmup = step - self.warmup_steps
        cosine_decay = 0.5 * (1 + math.cos(math.pi * step_after_warmup / decay_steps))
        decayed = (self.max_lr - self.min_lr) * cosine_decay + self.min_lr
        return decayed