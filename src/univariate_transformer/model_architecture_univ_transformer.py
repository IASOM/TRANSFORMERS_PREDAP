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
from .transformer_architechtures.model_architechture_informer import build_informer_model
from .transformer_architechtures.model_architechture_log_transformer import build_log_transformer_model
from .transformer_architechtures.model_architechture_LSTNet import build_lstnet_model
from .transformer_architechtures.model_architechture_base_tranformer import build_base_model 




def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function = "tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding = False):
    #return build_informer_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function, dropout, mlp_dropout, n_pred, pos_encoding)
    #return build_lstnet_model(input_shape, n_filters=head_size*num_heads, kernel_size=6, rnn_units=ff_dim, skip_units=ff_dim//2, skip=7, n_pred=n_pred, dropout=dropout)
    return build_base_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function, dropout, mlp_dropout, n_pred, pos_encoding)



class CustomCosineDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """
    Custom cosine decay learning rate schedule with warmup phase.
    Defaults to 50 epochs total and 20 warmup steps.
    """
    
    def __init__(self, initial_lr=1e-5, max_lr=5e-5, min_lr=1e-6, warmup_steps=20, total_steps=50):
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

