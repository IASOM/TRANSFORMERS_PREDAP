#!/usr/bin/env python3
"""
Encoder-Decoder Transformer Architecture Demonstration

This script demonstrates the new encoder-decoder transformer implementation
within the causal transformer framework. It showcases the key differences
between standard transformer (encoder-only) and encoder-decoder architectures.

Usage:
    python encoder_decoder_demo.py
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

from main_architechture_univ_causal import (
    build_model, 
    build_encoder_decoder_wrapper,
    transformer_encoder,
    transformer_decoder,
    create_decoder_inputs
)

def create_synthetic_data(batch_size: int = 32, seq_length: int = 60, features: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic time series data for demonstration.
    
    Args:
        batch_size: Number of samples
        seq_length: Sequence length
        features: Number of features
        
    Returns:
        Tuple of (X, y) where X is input sequences and y is target values
    """
    # Create synthetic sine wave with noise
    t = np.linspace(0, 4*np.pi, seq_length)
    
    X = []
    y = []
    
    for i in range(batch_size):
        # Random frequency and phase
        freq = np.random.uniform(0.5, 2.0)
        phase = np.random.uniform(0, 2*np.pi)
        noise_level = np.random.uniform(0.05, 0.15)
        
        # Generate sine wave with noise
        signal = np.sin(freq * t + phase) + np.random.normal(0, noise_level, seq_length)
        
        # Input is the sequence, target is the next value
        X.append(signal[:-1].reshape(-1, features))
        y.append(signal[-1])
    
    return np.array(X), np.array(y).reshape(-1, 1)


def demonstrate_architectures():
    """
    Demonstrate the difference between standard and encoder-decoder architectures.
    """
    print("="*80)
    print("ENCODER-DECODER TRANSFORMER ARCHITECTURE DEMONSTRATION")
    print("="*80)
    
    # Configuration parameters
    batch_size = 8
    seq_length = 60
    features = 1
    target_length = 1
    
    # Model parameters
    head_size = 64
    num_heads = 4
    ff_dim = 128
    dropout = 0.1
    
    # Create synthetic data
    X, y = create_synthetic_data(batch_size, seq_length, features)
    print(f"Data shapes: X={X.shape}, y={y.shape}")
    
    print("\n" + "="*60)
    print("1. STANDARD TRANSFORMER (ENCODER-ONLY)")
    print("="*60)
    
    # Build standard transformer
    standard_model = build_model(
        input_shape=(seq_length-1, features),
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=2,
        mlp_units=[64, 32],
        dropout=dropout,
        n_pred=1,
        pos_encoding=True
    )
    
    print("Standard Transformer Architecture:")
    standard_model.summary()
    
    # Test forward pass
    standard_pred = standard_model(X)
    print(f"Standard model prediction shape: {standard_pred.shape}")
    
    print("\n" + "="*60)
    print("2. ENCODER-DECODER TRANSFORMER")
    print("="*60)
    
    # Build encoder-decoder transformer
    encoder_decoder_model = build_encoder_decoder_wrapper(
        input_shape=(seq_length-1, features),
        target_length=target_length,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_blocks=2,
        num_decoder_blocks=1,
        mlp_units=[64, 32],
        dropout=dropout,
        n_pred=1,
        pos_encoding=True,
        decoder_strategy='learned'
    )
    
    print("Encoder-Decoder Transformer Architecture:")
    # Note: Since this is a custom wrapper, we'll show the core model structure
    print(f"Input shape: {(seq_length-1, features)}")
    print(f"Target length: {target_length}")
    print(f"Encoder blocks: 2")
    print(f"Decoder blocks: 1")
    print(f"Head size: {head_size}")
    print(f"Number of heads: {num_heads}")
    print(f"Feed-forward dimension: {ff_dim}")
    
    # Test forward pass
    encoder_decoder_pred = encoder_decoder_model(X)
    print(f"Encoder-decoder model prediction shape: {encoder_decoder_pred.shape}")
    
    print("\n" + "="*60)
    print("3. ARCHITECTURE COMPARISON")
    print("="*60)
    
    print("Standard Transformer (Encoder-Only):")
    print("  ✓ Uses only self-attention within the input sequence")
    print("  ✓ Processes entire input sequence simultaneously")
    print("  ✓ Global pooling to create fixed-size representation")
    print("  ✓ Simple and efficient for many tasks")
    
    print("\nEncoder-Decoder Transformer:")
    print("  ✓ Encoder processes input with bidirectional attention")
    print("  ✓ Decoder uses causal masking for autoregressive generation")
    print("  ✓ Cross-attention allows decoder to attend to encoder outputs")
    print("  ✓ More flexible for sequence-to-sequence tasks")
    print("  ✓ Learned decoder inputs for better target representation")
    
    print("\n" + "="*60)
    print("4. MODEL PARAMETER COMPARISON")
    print("="*60)
    
    # Count parameters
    standard_params = standard_model.count_params()
    encoder_decoder_params = encoder_decoder_model.count_params()
    
    print(f"Standard Transformer Parameters: {standard_params:,}")
    print(f"Encoder-Decoder Parameters: {encoder_decoder_params:,}")
    print(f"Parameter Difference: {encoder_decoder_params - standard_params:,}")
    print(f"Parameter Ratio: {encoder_decoder_params/standard_params:.2f}x")
    
    return {
        'standard_model': standard_model,
        'encoder_decoder_model': encoder_decoder_model,
        'data': (X, y),
        'predictions': {
            'standard': standard_pred,
            'encoder_decoder': encoder_decoder_pred
        },
        'parameters': {
            'standard': standard_params,
            'encoder_decoder': encoder_decoder_params
        }
    }


def demonstrate_attention_mechanisms():
    """
    Demonstrate the attention mechanisms in encoder-decoder architecture.
    """
    print("\n" + "="*60)
    print("5. ATTENTION MECHANISM DEMONSTRATION")
    print("="*60)
    
    # Create sample inputs
    batch_size = 2
    encoder_seq_len = 10
    decoder_seq_len = 3
    d_model = 64
    
    # Simulated encoder outputs and decoder inputs
    encoder_outputs = tf.random.normal((batch_size, encoder_seq_len, d_model))
    decoder_inputs = tf.random.normal((batch_size, decoder_seq_len, d_model))
    
    print(f"Encoder outputs shape: {encoder_outputs.shape}")
    print(f"Decoder inputs shape: {decoder_inputs.shape}")
    
    # Demonstrate decoder with cross-attention
    head_size = 32
    num_heads = 2
    ff_dim = 128
    
    decoder_output = transformer_decoder(
        decoder_inputs=decoder_inputs,
        encoder_outputs=encoder_outputs,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        dropout=0.1
    )
    
    print(f"Decoder output shape: {decoder_output.shape}")
    
    print("\nDecoder Attention Flow:")
    print("  1. Masked Self-Attention: Decoder attends to previous positions")
    print("  2. Cross-Attention: Decoder attends to encoder outputs")
    print("  3. Feed-Forward: Position-wise processing")
    print("  4. Residual Connections: Skip connections at each step")
    
    return decoder_output


def run_training_comparison():
    """
    Run a quick training comparison between architectures.
    """
    print("\n" + "="*60)
    print("6. QUICK TRAINING COMPARISON")
    print("="*60)
    
    # Create more data for training
    X_train, y_train = create_synthetic_data(batch_size=128, seq_length=31)
    X_val, y_val = create_synthetic_data(batch_size=32, seq_length=31)
    
    # Build models
    standard_model = build_model(
        input_shape=(30, 1),
        head_size=32,
        num_heads=2,
        ff_dim=64,
        num_transformer_blocks=1,
        mlp_units=[32],
        dropout=0.1,
        n_pred=1
    )
    
    encoder_decoder_model = build_encoder_decoder_wrapper(
        input_shape=(30, 1),
        target_length=1,
        head_size=32,
        num_heads=2,
        ff_dim=64,
        num_encoder_blocks=1,
        num_decoder_blocks=1,
        mlp_units=[32],
        dropout=0.1,
        n_pred=1,
        decoder_strategy='learned'
    )
    
    # Compile models
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    standard_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    encoder_decoder_model.compile(
        optimizer=optimizer,
        loss='mse',
        metrics=['mae']
    )
    
    # Quick training (few epochs for demo)
    epochs = 5
    
    print("Training Standard Transformer...")
    standard_history = standard_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        verbose=0
    )
    
    print("Training Encoder-Decoder Transformer...")
    encoder_decoder_history = encoder_decoder_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=16,
        verbose=0
    )
    
    # Compare results
    standard_loss = standard_history.history['val_loss'][-1]
    encoder_decoder_loss = encoder_decoder_history.history['val_loss'][-1]
    
    standard_mae = standard_history.history['val_mae'][-1]
    encoder_decoder_mae = encoder_decoder_history.history['val_mae'][-1]
    
    print(f"\nTraining Results ({epochs} epochs):")
    print(f"Standard Transformer    - Val Loss: {standard_loss:.4f}, Val MAE: {standard_mae:.4f}")
    print(f"Encoder-Decoder         - Val Loss: {encoder_decoder_loss:.4f}, Val MAE: {encoder_decoder_mae:.4f}")
    
    # Performance difference
    loss_improvement = ((standard_loss - encoder_decoder_loss) / standard_loss) * 100
    mae_improvement = ((standard_mae - encoder_decoder_mae) / standard_mae) * 100
    
    print(f"\nRelative Performance:")
    print(f"Loss improvement: {loss_improvement:.2f}%")
    print(f"MAE improvement: {mae_improvement:.2f}%")
    
    return {
        'standard_history': standard_history,
        'encoder_decoder_history': encoder_decoder_history,
        'results': {
            'standard': {'loss': standard_loss, 'mae': standard_mae},
            'encoder_decoder': {'loss': encoder_decoder_loss, 'mae': encoder_decoder_mae}
        }
    }


def main():
    """Main demonstration function."""
    print("Starting Encoder-Decoder Transformer Demonstration...")
    
    try:
        # 1. Architecture demonstration
        arch_results = demonstrate_architectures()
        
        # 2. Attention mechanism demonstration
        attention_results = demonstrate_attention_mechanisms()
        
        # 3. Training comparison
        training_results = run_training_comparison()
        
        print("\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        
        print("✅ Successfully demonstrated encoder-decoder architecture")
        print("✅ Showed attention mechanism flow")
        print("✅ Compared training performance")
        print("✅ Architecture is ready for integration into main pipeline")
        
        print("\nKey Features Implemented:")
        print("  • Proper cross-attention in decoder")
        print("  • Causal masking for autoregressive generation")
        print("  • Learnable decoder input embeddings")
        print("  • Configurable encoder/decoder blocks")
        print("  • Seamless integration with existing pipeline")
        
        print("\nNext Steps:")
        print("  1. Use TransformerUnivConfig.create_encoder_decoder_config()")
        print("  2. Set use_encoder_decoder=True in configuration")
        print("  3. Adjust num_encoder_blocks and num_decoder_blocks as needed")
        print("  4. Choose decoder_strategy: 'learned', 'zeros', or 'random'")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()