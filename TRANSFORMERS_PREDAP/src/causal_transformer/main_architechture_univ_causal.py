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


def transformer_decoder(decoder_inputs, encoder_outputs, head_size, num_heads, ff_dim, activation_function='tanh', dropout=0):
    """
    Transformer decoder block with masked self-attention and cross-attention.
    
    Args:
        decoder_inputs: Decoder input tensor (batch_size, target_sequence_length, features)
        encoder_outputs: Encoder output tensor (batch_size, source_sequence_length, features)
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        activation_function: Activation function for feed-forward layers
        dropout: Dropout rate
        
    Returns:
        Decoded tensor with cross-attention to encoder outputs
    """
    
    # 1. Masked Self-Attention (with causal masking)
    x = layers.LayerNormalization(epsilon=1e-6)(decoder_inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout
    )(x, x, use_causal_mask=True)  # Causal masking for autoregressive generation
    x = layers.Dropout(dropout)(x)
    res1 = x + decoder_inputs  # First residual connection
    
    # 2. Cross-Attention (decoder attends to encoder outputs)
    x = layers.LayerNormalization(epsilon=1e-6)(res1)
    x = layers.MultiHeadAttention(
        key_dim=head_size, 
        num_heads=num_heads, 
        dropout=dropout
    )(x, encoder_outputs)  # Query from decoder, Key/Value from encoder
    x = layers.Dropout(dropout)(x)
    res2 = x + res1  # Second residual connection
    
    # 3. Feed-Forward Network
    x = layers.LayerNormalization(epsilon=1e-6)(res2)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=decoder_inputs.shape[-1], kernel_size=1)(x)
    x = x + res2  # Third residual connection
    
    return x


def transformer_encoder(inputs, head_size, num_heads, ff_dim, activation_function='tanh', dropout=0, causal_masking=False):
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

    # Normalization and Attention
    x = layers.LayerNormalization(epsilon=1e-6)(inputs)  # to inputs to stabilize training
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout)(x, x, use_causal_mask=causal_masking)  # self attention to normalized input 
    x = layers.Dropout(dropout)(x)  # (dropout to reduce overfitting)
    res = x + inputs  # attention output added to original inputs

    # Feed Forward Part
    x = layers.LayerNormalization(epsilon=1e-6)(res)  # again after resid connection
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation=activation_function)(x)  # point-wise convol.: Expands feature dim to ff_dim 
    x = layers.Dropout(dropout)(x)  # dropout again
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)  # reduces feature dim back to match input size
    x = x + res
    
    return x


def build_model(input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function = "tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding = False, causal_masking=False):
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
    d_model = max(head_size * num_heads, 32)
    x = layers.Dense(d_model, activation=activation_function)(inputs)
    '''if pos_encoding == True:
        x = PositionalEncoding(input_shape[0], d_model)(x)  # add positional encoding if enabled
    '''
    for _ in range(num_transformer_blocks):  # apply num_transformer_blocks transformer encoder layers seq.
        x = transformer_encoder(x, head_size, num_heads, ff_dim, activation_function, dropout, causal_masking)  # uses previous defined trans_encoder layer

    x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)  # reduces seq dimension (timesteps) averaging for each feature channel
    #x = layers.GlobalAveragePooling1D(data_format="channels_last")(x)
    
    
    for dim in mlp_units:  # multi layer perceptron (dropout to avoid overfitting)
        x = layers.Dense(dim, activation=activation_function)(x)
        x = layers.Dropout(mlp_dropout)(x)
   
    outputs = layers.Dense(n_pred)(x)
    return keras.Model(inputs, outputs)


def build_encoder_decoder_model(input_shape, target_length, head_size, num_heads, ff_dim, 
                               num_encoder_blocks, num_decoder_blocks, mlp_units, 
                               activation_function='tanh', dropout=0, mlp_dropout=0, n_pred=1, 
                               pos_encoding=False):
    """
    Build complete encoder-decoder transformer model for time series forecasting.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        target_length: Length of target sequence for decoder
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        num_encoder_blocks: Number of transformer encoder blocks
        num_decoder_blocks: Number of transformer decoder blocks
        mlp_units: List of MLP layer dimensions
        activation_function: Activation function
        dropout: Dropout rate for transformer layers
        mlp_dropout: Dropout rate for MLP layers
        n_pred: Number of prediction steps
        pos_encoding: Whether to use positional encoding
        
    Returns:
        Compiled Keras model with encoder-decoder architecture
    """
    
    # ==================== ENCODER ====================
    encoder_inputs = keras.Input(shape=input_shape, name='encoder_inputs')
    
    # Project to model dimension
    d_model = max(head_size * num_heads, 8)
    encoder_x = layers.Dense(d_model, activation=activation_function, name='encoder_projection')(encoder_inputs)
    
    # Add positional encoding if enabled
    if pos_encoding:
        encoder_x = PositionalEncoding(input_shape[0], d_model, name='encoder_pos_encoding')(encoder_x)
    
    # Apply encoder blocks
    for i in range(num_encoder_blocks):
        encoder_x = transformer_encoder(
            encoder_x, head_size, num_heads, ff_dim, activation_function, dropout, causal_masking=False
        )
    
    # ==================== DECODER ====================
    # Create decoder inputs (can be learnable embeddings or zeros for forecasting)
    decoder_inputs = layers.Input(shape=(target_length, d_model), name='decoder_inputs')
    
    # Apply decoder blocks with cross-attention to encoder outputs
    decoder_x = decoder_inputs
    for i in range(num_decoder_blocks):
        decoder_x = transformer_decoder(
            decoder_x, encoder_x, head_size, num_heads, ff_dim, activation_function, dropout
        )
    
    # ==================== OUTPUT HEAD ====================
    # Global pooling or use last time step
    if target_length == 1:
        # For single-step prediction, use the single decoder output
        x = layers.Reshape((d_model,))(decoder_x)
    else:
        # For multi-step prediction, use global average pooling
        x = layers.GlobalAveragePooling1D(name='global_pooling')(decoder_x)
    
    # MLP head for final predictions
    for i, dim in enumerate(mlp_units):
        x = layers.Dense(dim, activation=activation_function, name=f'mlp_dense_{i}')(x)
        x = layers.Dropout(mlp_dropout, name=f'mlp_dropout_{i}')(x)
    
    # Final output layer
    outputs = layers.Dense(n_pred, name='output_layer')(x)
    
    # Create the model with both encoder and decoder inputs
    model = keras.Model([encoder_inputs, decoder_inputs], outputs, name='encoder_decoder_transformer')
    
    return model


def build_teacher_forcing_model(input_shape, target_length, head_size, num_heads, ff_dim, 
                               num_encoder_blocks, num_decoder_blocks, mlp_units, 
                               activation_function='tanh', dropout=0, mlp_dropout=0, 
                               n_pred=1, pos_encoding=False):
    """
    Build encoder-decoder model that accepts ground truth targets for teacher forcing.
    
    This model has two modes:
    - Training: Uses shifted ground truth as decoder inputs (teacher forcing)
    - Inference: Uses learned embeddings or zeros as decoder inputs
    
    Args:
        input_shape: Shape of input data (sequence_length, features) 
        target_length: Length of target sequence for decoder
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        num_encoder_blocks: Number of transformer encoder blocks
        num_decoder_blocks: Number of transformer decoder blocks
        mlp_units: List of MLP layer dimensions
        activation_function: Activation function
        dropout: Dropout rate for transformer layers
        mlp_dropout: Dropout rate for MLP layers
        n_pred: Number of prediction steps
        pos_encoding: Whether to use positional encoding
        
    Returns:
        Keras model that accepts [encoder_inputs, decoder_inputs] during training
        and only encoder_inputs during inference
    """
    
    # ==================== ENCODER ====================
    encoder_inputs = keras.Input(shape=input_shape, name='encoder_inputs')
    
    # Project to model dimension
    d_model = max(head_size * num_heads, 8)
    encoder_x = layers.Dense(d_model, activation=activation_function, name='encoder_projection')(encoder_inputs)
    
    # Add positional encoding if enabled
    if pos_encoding:
        encoder_x = PositionalEncoding(input_shape[0], d_model, name='encoder_pos_encoding')(encoder_x)
    
    # Apply encoder blocks
    for i in range(num_encoder_blocks):
        encoder_x = transformer_encoder(
            encoder_x, head_size, num_heads, ff_dim, activation_function, dropout, causal_masking=False
        )
    
    # ==================== DECODER ====================
    # Decoder inputs for teacher forcing (ground truth targets shifted right)
    decoder_inputs = layers.Input(shape=(target_length, d_model), name='decoder_inputs')
    
    # Add positional encoding to decoder if enabled
    decoder_x = decoder_inputs
    if pos_encoding:
        decoder_x = PositionalEncoding(target_length, d_model, name='decoder_pos_encoding')(decoder_x)
    
    # Apply decoder blocks with cross-attention to encoder outputs
    for i in range(num_decoder_blocks):
        decoder_x = transformer_decoder(
            decoder_x, encoder_x, head_size, num_heads, ff_dim, activation_function, dropout
        )
    
    # ==================== OUTPUT HEAD ====================
    # Process decoder outputs to predictions
    if target_length == 1:
        # For single-step prediction, use the single decoder output
        x = layers.Reshape((d_model,))(decoder_x)
    else:
        # For multi-step prediction, use last time step or global pooling
        x = decoder_x[:, -1, :]  # Use last decoder output
        # Alternative: x = layers.GlobalAveragePooling1D()(decoder_x)
    
    # MLP head for final predictions
    for i, dim in enumerate(mlp_units):
        x = layers.Dense(dim, activation=activation_function, name=f'mlp_dense_{i}')(x)
        x = layers.Dropout(mlp_dropout, name=f'mlp_dropout_{i}')(x)
    
    # Final output layer
    outputs = layers.Dense(n_pred, name='output_layer')(x)
    
    # Create the model with both encoder and decoder inputs
    model = keras.Model([encoder_inputs, decoder_inputs], outputs, name='teacher_forcing_transformer')
    
    return model


def create_decoder_inputs(batch_size, target_length, d_model, strategy='learned'):
    """
    Create decoder inputs for the encoder-decoder model.
    
    Args:
        batch_size: Batch size
        target_length: Length of target sequence
        d_model: Model dimension
        strategy: Strategy for creating inputs ('zeros', 'learned', 'random')
        
    Returns:
        Decoder inputs tensor
    """
    if strategy == 'zeros':
        return tf.zeros((batch_size, target_length, d_model))
    elif strategy == 'learned':
        # Create learnable embeddings (would need to be part of model)
        return tf.random.normal((batch_size, target_length, d_model), stddev=0.1)
    elif strategy == 'random':
        return tf.random.normal((batch_size, target_length, d_model), stddev=0.1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def create_teacher_forcing_inputs(targets, start_token_value=0.0):
    """
    Create decoder inputs for teacher forcing using shifted ground truth.
    
    Args:
        targets: Ground truth target sequence (batch_size, target_length, features)
        start_token_value: Value to use as start token (prepended to sequence)
        
    Returns:
        Shifted decoder inputs for teacher forcing
    """
    batch_size = tf.shape(targets)[0]
    target_length = tf.shape(targets)[1]
    features = tf.shape(targets)[2]
    
    # Create start tokens (BOS - Beginning of Sequence)
    start_tokens = tf.fill((batch_size, 1, features), start_token_value)
    
    # Shift targets right by prepending start token and removing last element
    # This creates the "teacher forcing" input: [BOS, y0, y1, ..., y_{n-2}]
    # while targets are: [y0, y1, y2, ..., y_{n-1}]
    decoder_inputs = tf.concat([start_tokens, targets[:, :-1, :]], axis=1)
    
    return decoder_inputs


class TeacherForcingWrapper(keras.Model):
    """
    Wrapper class for encoder-decoder model with teacher forcing support.
    
    During training: Uses ground truth targets (shifted) as decoder inputs
    During inference: Uses learned embeddings or zeros as decoder inputs
    """
    
    def __init__(self, encoder_decoder_model, target_length, d_model, input_features=1, 
                 decoder_strategy='learned', start_token_value=0.0, **kwargs):
        super().__init__(**kwargs)
        self.encoder_decoder_model = encoder_decoder_model
        self.target_length = target_length
        self.d_model = d_model
        self.input_features = input_features
        self.decoder_strategy = decoder_strategy
        self.start_token_value = start_token_value
        
        # Projection layer to convert target features to d_model dimension
        self.target_projection = layers.Dense(d_model, name='target_projection')
        
        # Create learnable decoder inputs if using learned strategy for inference
        if decoder_strategy == 'learned':
            self.decoder_embeddings = self.add_weight(
                shape=(1, target_length, d_model),
                initializer='random_normal',
                trainable=True,
                name='decoder_embeddings'
            )
    
    def call(self, inputs, training=None):
        if isinstance(inputs, list) and len(inputs) == 2:
            # Training mode: [encoder_inputs, target_sequence]
            encoder_inputs, targets = inputs
            batch_size = tf.shape(encoder_inputs)[0]
            
            if training:
                # Teacher forcing: use shifted ground truth as decoder inputs
                decoder_inputs = create_teacher_forcing_inputs(targets, self.start_token_value)
                # Project target features to model dimension
                decoder_inputs = self.target_projection(decoder_inputs)
            else:
                # Inference mode: use learned embeddings or zeros
                if self.decoder_strategy == 'learned':
                    decoder_inputs = tf.tile(self.decoder_embeddings, [batch_size, 1, 1])
                else:
                    decoder_inputs = create_decoder_inputs(
                        batch_size, self.target_length, self.d_model, self.decoder_strategy
                    )
        else:
            # Inference mode: only encoder_inputs provided
            encoder_inputs = inputs
            batch_size = tf.shape(encoder_inputs)[0]
            
            if self.decoder_strategy == 'learned':
                decoder_inputs = tf.tile(self.decoder_embeddings, [batch_size, 1, 1])
            else:
                decoder_inputs = create_decoder_inputs(
                    batch_size, self.target_length, self.d_model, self.decoder_strategy
                )
        
        return self.encoder_decoder_model([encoder_inputs, decoder_inputs], training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'target_length': self.target_length,
            'd_model': self.d_model,
            'input_features': self.input_features,
            'decoder_strategy': self.decoder_strategy,
            'start_token_value': self.start_token_value
        })
        return config


class EncoderDecoderWrapper(keras.Model):
    """
    Original wrapper class for encoder-decoder model that handles decoder input creation.
    """
    
    def __init__(self, encoder_decoder_model, target_length, d_model, decoder_strategy='zeros', **kwargs):
        super().__init__(**kwargs)
        self.encoder_decoder_model = encoder_decoder_model
        self.target_length = target_length
        self.d_model = d_model
        self.decoder_strategy = decoder_strategy
        
        # Create learnable decoder inputs if using learned strategy
        if decoder_strategy == 'learned':
            self.decoder_embeddings = self.add_weight(
                shape=(1, target_length, d_model),
                initializer='random_normal',
                trainable=True,
                name='decoder_embeddings'
            )
    
    def call(self, encoder_inputs, training=None):
        batch_size = tf.shape(encoder_inputs)[0]
        
        if self.decoder_strategy == 'learned':
            # Repeat learned embeddings for the batch
            decoder_inputs = tf.tile(self.decoder_embeddings, [batch_size, 1, 1])
        else:
            # Create decoder inputs based on strategy
            decoder_inputs = create_decoder_inputs(
                batch_size, self.target_length, self.d_model, self.decoder_strategy
            )
        
        return self.encoder_decoder_model([encoder_inputs, decoder_inputs], training=training)
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'target_length': self.target_length,
            'd_model': self.d_model,
            'decoder_strategy': self.decoder_strategy
        })
        return config


def build_encoder_decoder_wrapper(input_shape, target_length, head_size, num_heads, ff_dim, 
                                 num_encoder_blocks, num_decoder_blocks, mlp_units, 
                                 activation_function='tanh', dropout=0, mlp_dropout=0, 
                                 n_pred=1, pos_encoding=False, decoder_strategy='zeros'):
    """
    Build encoder-decoder model with automatic decoder input handling.
    
    This is the recommended function to use for creating encoder-decoder models.
    """
    d_model = max(head_size * num_heads, 8)
    
    # Build the core encoder-decoder model
    core_model = build_encoder_decoder_model(
        input_shape=input_shape,
        target_length=target_length,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=num_decoder_blocks,
        mlp_units=mlp_units,
        activation_function=activation_function,
        dropout=dropout,
        mlp_dropout=mlp_dropout,
        n_pred=n_pred,
        pos_encoding=pos_encoding
    )
    
    # Wrap it to handle decoder inputs automatically
    wrapped_model = EncoderDecoderWrapper(
        encoder_decoder_model=core_model,
        target_length=target_length,
        d_model=d_model,
        decoder_strategy=decoder_strategy
    )
    
    return wrapped_model


def build_teacher_forcing_wrapper(input_shape, target_length, head_size, num_heads, ff_dim, 
                                 num_encoder_blocks, num_decoder_blocks, mlp_units, 
                                 activation_function='tanh', dropout=0, mlp_dropout=0, 
                                 n_pred=1, pos_encoding=False, decoder_strategy='learned',
                                 input_features=1, start_token_value=0.0):
    """
    Build encoder-decoder model with teacher forcing support.
    
    This model uses ground truth targets (shifted right) as decoder inputs during training,
    and learned embeddings or zeros during inference.
    
    Args:
        input_shape: Shape of input data (sequence_length, features)
        target_length: Length of target sequence for decoder
        head_size: Dimension of each attention head
        num_heads: Number of attention heads
        ff_dim: Dimension of feed-forward layer
        num_encoder_blocks: Number of transformer encoder blocks
        num_decoder_blocks: Number of transformer decoder blocks
        mlp_units: List of MLP layer dimensions
        activation_function: Activation function
        dropout: Dropout rate for transformer layers
        mlp_dropout: Dropout rate for MLP layers
        n_pred: Number of prediction steps
        pos_encoding: Whether to use positional encoding
        decoder_strategy: Strategy for inference ('learned', 'zeros', 'random')
        input_features: Number of input features (for target projection)
        start_token_value: Value for start-of-sequence token
        
    Returns:
        TeacherForcingWrapper model that handles teacher forcing automatically
    """
    d_model = max(head_size * num_heads, 8)
    
    # Build the core teacher forcing model
    core_model = build_teacher_forcing_model(
        input_shape=input_shape,
        target_length=target_length,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_blocks=num_encoder_blocks,
        num_decoder_blocks=num_decoder_blocks,
        mlp_units=mlp_units,
        activation_function=activation_function,
        dropout=dropout,
        mlp_dropout=mlp_dropout,
        n_pred=n_pred,
        pos_encoding=pos_encoding
    )
    
    # Wrap it to handle teacher forcing automatically
    wrapped_model = TeacherForcingWrapper(
        encoder_decoder_model=core_model,
        target_length=target_length,
        d_model=d_model,
        input_features=input_features,
        decoder_strategy=decoder_strategy,
        start_token_value=start_token_value
    )
    
    return wrapped_model


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