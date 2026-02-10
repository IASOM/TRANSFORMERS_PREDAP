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
from tensorflow.keras.losses import Huber
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.constraints import MaxNorm

from config.base_transformer_config import BaseTransformerConfig
from univariate_transformer.model_architechture_informer import build_informer_model
from univariate_transformer.model_architechture_log_transformer import build_log_transformer_model
from univariate_transformer.model_architechture_LSTNet import build_lstnet_model

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
    revin_layer = RevIN()
    x = revin_layer(input_layer, mode='norm')
    
    d_model = max(transformer_params['head_size'] * transformer_params['num_heads'], 32)
    x = layers.Dense(d_model)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    # Transformer Block
    x = PositionalEncoding(input_shape[0], d_model)(x)
    
    
    '''
    # LSTM Block
    x = layers.LSTM(
        lstm_params['units_1'], 
        return_sequences=lstm_params['return_sequences'],
        recurrent_activation='tanh',
        #recurrent_dropout = 0.2,
        #recurrent_constraint=MaxNorm(1.0),
    )(input_layer)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(lstm_params['dropout'])(x)  # Dropout to reduce overfitting
    
    x = layers.LSTM(
        lstm_params['units_2'], 
        return_sequences=lstm_params['return_sequences'],
        recurrent_activation='tanh',
        #recurrent_dropout = 0.2,
        #recurrent_constraint=MaxNorm(1.0),
     )(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    x = layers.Dropout(lstm_params['dropout'])(x)
    '''

    # Transformer Encoder Block

    '''d_model = max(transformer_params['head_size'] * transformer_params['num_heads'], 32)
    x = layers.Dense(d_model)(input_layer)'''


    #x = PositionalEncoding(x.shape[1], x.shape[2])(x)
    for _ in range(transformer_params['num_transformer_blocks']):
        x = transformer_encoder(
            x, 
            head_size=transformer_params['head_size'],
            num_heads=transformer_params['num_heads'],
            ff_dim=transformer_params['ff_dim'],
            activation_function=activation_function,
            dropout=transformer_params['dropout']
        )

    # GlobalAveragePooling1D Layer
    #x = layers.GlobalAveragePooling1D(data_format="channels_last")(x) # May be changed to Flatten() if needed 
    if input_shape[0] >= 60:  # Only apply pooling if sequence length is sufficient
        x = layers.AveragePooling1D(14, data_format="channels_first")(x)
    x = layers.Flatten()(x)
    
    for dim in transformer_params['mlp_units']:
        x = layers.Dense(dim, activation=activation_function)(x)
        x = layers.Dropout(transformer_params['dropout'])(x)
    # Output Layer
    outputs = layers.Dense(forecast, activation='linear')(x)

    # Reshape Outputs
    outputs = layers.Reshape((forecast, 1))(outputs)
    outputs = revin_layer(outputs, mode='denorm')
    outputs = layers.Reshape((forecast,))(outputs)


    # Build Model
    model = keras.Model(inputs=input_layer, outputs=outputs)

    # Compile Model
    #model.compile(optimizer='adam', loss='mse')

    return model

    

'''def hybrid_lstm_transformer_model(input_shape, forecast, 
                                  lstm_params=None, 
                                  transformer_params=None,
                                  activation_function='tanh'):
    if lstm_params is None:
        lstm_params = default_config.DEFAULT_RESIDUAL_LSTM_PARAMS.copy()
    if transformer_params is None:
        transformer_params = default_config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS.copy()

    #return build_informer_model(input_shape, transformer_params['head_size'], transformer_params['num_heads'], transformer_params['ff_dim'], transformer_params['num_transformer_blocks'], transformer_params['mlp_units'], activation_function, transformer_params['dropout'], transformer_params['dropout'], forecast, pos_encoding=True)
    #return build_log_transformer_model(input_shape, transformer_params['head_size'], transformer_params['num_heads'], transformer_params['ff_dim'], transformer_params['num_transformer_blocks'], transformer_params['mlp_units'], activation_function, transformer_params['dropout'], transformer_params['dropout'], forecast, pos_encoding=True)
    return build_lstnet_model(input_shape, n_filters=transformer_params['head_size']*transformer_params['num_heads'], kernel_size=6, rnn_units=transformer_params['ff_dim'], skip_units=transformer_params['ff_dim']//2, skip=7, n_pred=forecast, dropout=transformer_params['dropout'])
'''

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
    
@keras.saving.register_keras_serializable(package="predap")
class PositionalEncoding(layers.Layer):
    def __init__(self, sequence_length, d_model, **kwargs):
        super().__init__(**kwargs)
        self.sequence_length = sequence_length
        self.d_model = d_model

        # Create the positional encoding matrix once
        pos = tf.range(start=0, limit=sequence_length, delta=1, dtype=tf.float32)[:, tf.newaxis]  # (seq_len, 1)
        i = tf.range(start=0, limit=d_model, delta=1, dtype=tf.float32)[tf.newaxis, :]             # (1, d_model)
        # compute the angle rates
        angle_rates = 1 / (10000 ** ( (2 * (i//2)) / tf.cast(d_model, tf.float32) ))                  # (1, d_model)
        angle_rads = pos * angle_rates                                                              # (seq_len, d_model)

        # apply sin to even indices in the array; cos to odd indices
        sines = tf.sin(angle_rads[:, 0::2])
        coses = tf.cos(angle_rads[:, 1::2])
        # now interleave sines & coses into one matrix
        pos_encoding = tf.concat([sines, coses], axis=-1)                                           # (seq_len, d_model)
        pos_encoding = pos_encoding[tf.newaxis, ...]                                                # (1, seq_len, d_model)
        self.pos_encoding = tf.cast(pos_encoding, dtype=tf.float32)

    def call(self, x):
        # x shape: (batch_size, seq_len, d_model)
        seq_len = tf.shape(x)[1]
        return x + self.pos_encoding[:, :seq_len, :]
    
@keras.saving.register_keras_serializable(package="predap")
class RevIN(layers.Layer):
    def __init__(self, eps=1e-5, detach_grad=False, **kwargs):
        super(RevIN, self).__init__(**kwargs)
        self.eps = eps
        self.detach_grad = detach_grad
        # These will store the mean and stdev of the current batch/instance
        self.mean = None
        self.stdev = None

    def call(self, x, mode='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = (x - self.mean) / self.stdev
            return x
        elif mode == 'denorm':
            # Use the statistics stored during the last 'norm' call
            dims = x.shape[-1]
            x = x * self.stdev[:, :, :dims] + self.mean[:, :, :dims]
            return x

    def _get_statistics(self, x):
        # Calculate mean and stdev across the time dimension (axis 1)
        # Assuming shape: (batch, time_steps, features)
        self.mean = tf.reduce_mean(x, axis=1, keepdims=True)
        self.stdev = tf.math.reduce_std(x, axis=1, keepdims=True) + self.eps
        
        if self.detach_grad:
            self.mean = tf.stop_gradient(self.mean)
            self.stdev = tf.stop_gradient(self.stdev)