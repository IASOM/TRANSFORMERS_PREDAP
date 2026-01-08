#!/usr/bin/env python3
"""
Teacher Forcing Encoder-Decoder Transformer Demonstration

This script demonstrates the teacher forcing implementation for encoder-decoder transformers.
Teacher forcing uses shifted ground truth as decoder inputs during training, which can lead to
faster convergence and better performance.

Usage:
    python teacher_forcing_demo.py
"""

import numpy as np
import tensorflow as tf
from typing import Tuple, Dict, Any
import matplotlib.pyplot as plt

from main_architechture_univ_causal import (
    build_teacher_forcing_wrapper,
    build_encoder_decoder_wrapper,
    create_teacher_forcing_inputs
)

def create_sample_data(n_samples: int = 100, seq_length: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sample time series data for demonstration.
    
    Args:
        n_samples: Number of samples
        seq_length: Length of each sequence
        
    Returns:
        Tuple of (X, y) where X is input and y is target
    """
    # Create synthetic time series with trend and seasonality
    t = np.linspace(0, 4*np.pi, seq_length + 1)
    
    X = []
    y = []
    
    for i in range(n_samples):
        # Random parameters for variation
        trend = np.random.uniform(-0.1, 0.1)
        amplitude = np.random.uniform(0.5, 1.5)
        frequency = np.random.uniform(0.5, 2.0)
        noise = np.random.normal(0, 0.1, seq_length + 1)
        
        # Generate series: trend + sine wave + noise
        series = trend * t + amplitude * np.sin(frequency * t) + noise
        
        # Input is first seq_length points, target is the next point
        X.append(series[:-1].reshape(-1, 1))
        y.append(series[-1])
    
    return np.array(X), np.array(y).reshape(-1, 1)


def demonstrate_teacher_forcing_inputs():
    """
    Demonstrate how teacher forcing inputs are created from ground truth.
    """
    print("="*80)
    print("TEACHER FORCING INPUT CREATION DEMONSTRATION")
    print("="*80)
    
    # Create sample targets
    batch_size = 3
    target_length = 5
    features = 1
    
    # Sample target sequences (what we want to predict)
    targets = np.random.randn(batch_size, target_length, features)
    targets = tf.constant(targets, dtype=tf.float32)
    
    print(f"Original targets shape: {targets.shape}")
    print("Original targets:")
    for i in range(batch_size):
        print(f"  Sample {i}: {targets[i, :, 0].numpy()}")
    
    # Create teacher forcing inputs (shifted right with start token)
    teacher_inputs = create_teacher_forcing_inputs(targets, start_token_value=0.0)
    
    print(f"\nTeacher forcing inputs shape: {teacher_inputs.shape}")
    print("Teacher forcing inputs (shifted with start token):")
    for i in range(batch_size):
        print(f"  Sample {i}: {teacher_inputs[i, :, 0].numpy()}")
    
    print("\nExplanation:")
    print("- Teacher forcing inputs = [START_TOKEN, target[0], target[1], ..., target[n-2]]")
    print("- Targets remain         = [target[0], target[1], target[2], ..., target[n-1]]")
    print("- This allows the decoder to use ground truth context during training")
    
    return teacher_inputs, targets


def compare_architectures():
    """
    Compare standard encoder-decoder vs teacher forcing encoder-decoder.
    """
    print("\n" + "="*80)
    print("ARCHITECTURE COMPARISON: STANDARD vs TEACHER FORCING")
    print("="*80)
    
    # Model parameters
    input_shape = (10, 1)
    target_length = 1
    head_size = 32
    num_heads = 2
    ff_dim = 64
    
    print("Building Standard Encoder-Decoder...")
    standard_model = build_encoder_decoder_wrapper(
        input_shape=input_shape,
        target_length=target_length,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_blocks=2,
        num_decoder_blocks=1,
        mlp_units=[32],
        dropout=0.1,
        decoder_strategy='learned'
    )
    
    print("Building Teacher Forcing Encoder-Decoder...")
    teacher_forcing_model = build_teacher_forcing_wrapper(
        input_shape=input_shape,
        target_length=target_length,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_encoder_blocks=2,
        num_decoder_blocks=1,
        mlp_units=[32],
        dropout=0.1,
        decoder_strategy='learned'
    )
    
    # Create sample data
    X_sample, y_sample = create_sample_data(n_samples=16, seq_length=10)
    
    print(f"\nSample data shapes: X={X_sample.shape}, y={y_sample.shape}")
    
    # Test forward passes
    print("\nTesting forward passes...")
    
    # Standard model: only needs encoder inputs
    standard_pred = standard_model(X_sample)
    print(f"Standard model output shape: {standard_pred.shape}")
    
    # Teacher forcing model: needs both encoder inputs and targets during training
    # For inference, it only needs encoder inputs
    teacher_pred_inference = teacher_forcing_model(X_sample)
    print(f"Teacher forcing model (inference) output shape: {teacher_pred_inference.shape}")
    
    # For training, it would receive [X_sample, y_sample_reshaped]
    y_reshaped = y_sample.reshape(-1, 1, 1)  # (batch, target_length, features)
    teacher_pred_training = teacher_forcing_model([X_sample, y_reshaped], training=True)
    print(f"Teacher forcing model (training) output shape: {teacher_pred_training.shape}")
    
    # Parameter comparison
    standard_params = standard_model.count_params()
    teacher_params = teacher_forcing_model.count_params()
    
    print(f"\nParameter Comparison:")
    print(f"Standard Encoder-Decoder:    {standard_params:,} parameters")
    print(f"Teacher Forcing Model:       {teacher_params:,} parameters")
    print(f"Difference:                  {teacher_params - standard_params:,} parameters")
    
    return {
        'standard_model': standard_model,
        'teacher_forcing_model': teacher_forcing_model,
        'sample_data': (X_sample, y_sample),
        'predictions': {
            'standard': standard_pred,
            'teacher_inference': teacher_pred_inference,
            'teacher_training': teacher_pred_training
        }
    }


def run_training_comparison():
    """
    Compare training performance between standard and teacher forcing models.
    """
    print("\n" + "="*80)
    print("TRAINING PERFORMANCE COMPARISON")
    print("="*80)
    
    # Generate larger dataset for training
    X_train, y_train = create_sample_data(n_samples=500, seq_length=15)
    X_val, y_val = create_sample_data(n_samples=100, seq_length=15)
    
    print(f"Training data: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation data: X={X_val.shape}, y={y_val.shape}")
    
    # Model configuration
    config = {
        'input_shape': (15, 1),
        'target_length': 1,
        'head_size': 64,
        'num_heads': 4,
        'ff_dim': 128,
        'num_encoder_blocks': 2,
        'num_decoder_blocks': 1,
        'mlp_units': [32],
        'dropout': 0.1
    }
    
    # Build models
    print("Building models...")
    standard_model = build_encoder_decoder_wrapper(**config, decoder_strategy='learned')
    teacher_model = build_teacher_forcing_wrapper(**config, decoder_strategy='learned')
    
    # Compile models
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)
    
    standard_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    teacher_model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
    
    # Training parameters
    epochs = 10
    batch_size = 32
    
    print(f"\nTraining for {epochs} epochs with batch size {batch_size}...")
    
    # Train standard model
    print("Training Standard Encoder-Decoder...")
    standard_history = standard_model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Train teacher forcing model
    print("Training Teacher Forcing Model...")
    # Prepare teacher forcing inputs
    y_train_reshaped = y_train.reshape(-1, 1, 1)
    y_val_reshaped = y_val.reshape(-1, 1, 1)
    
    teacher_history = teacher_model.fit(
        [X_train, y_train_reshaped], y_train_reshaped,
        validation_data=([X_val, y_val_reshaped], y_val_reshaped),
        epochs=epochs,
        batch_size=batch_size,
        verbose=0
    )
    
    # Compare results
    print("\n" + "="*50)
    print("TRAINING RESULTS")
    print("="*50)
    
    # Final metrics
    standard_val_loss = standard_history.history['val_loss'][-1]
    standard_val_mae = standard_history.history['val_mae'][-1]
    
    teacher_val_loss = teacher_history.history['val_loss'][-1]
    teacher_val_mae = teacher_history.history['val_mae'][-1]
    
    print(f"Standard Model - Final Val Loss: {standard_val_loss:.6f}, Val MAE: {standard_val_mae:.6f}")
    print(f"Teacher Model  - Final Val Loss: {teacher_val_loss:.6f}, Val MAE: {teacher_val_mae:.6f}")
    
    # Improvement metrics
    loss_improvement = ((standard_val_loss - teacher_val_loss) / standard_val_loss) * 100
    mae_improvement = ((standard_val_mae - teacher_val_mae) / standard_val_mae) * 100
    
    print(f"\nImprovement with Teacher Forcing:")
    print(f"Loss improvement: {loss_improvement:.2f}%")
    print(f"MAE improvement: {mae_improvement:.2f}%")
    
    # Convergence analysis
    standard_losses = standard_history.history['val_loss']
    teacher_losses = teacher_history.history['val_loss']
    
    print(f"\nConvergence Analysis:")
    print(f"Standard model best val loss: {min(standard_losses):.6f} at epoch {np.argmin(standard_losses) + 1}")
    print(f"Teacher model best val loss:  {min(teacher_losses):.6f} at epoch {np.argmin(teacher_losses) + 1}")
    
    return {
        'standard_history': standard_history,
        'teacher_history': teacher_history,
        'results': {
            'standard': {'loss': standard_val_loss, 'mae': standard_val_mae},
            'teacher': {'loss': teacher_val_loss, 'mae': teacher_val_mae}
        },
        'improvements': {
            'loss': loss_improvement,
            'mae': mae_improvement
        }
    }


def demonstrate_inference_modes():
    """
    Demonstrate how teacher forcing model works in training vs inference modes.
    """
    print("\n" + "="*80)
    print("TRAINING vs INFERENCE MODE DEMONSTRATION")
    print("="*80)
    
    # Build a small teacher forcing model
    model = build_teacher_forcing_wrapper(
        input_shape=(5, 1),
        target_length=1,
        head_size=32,
        num_heads=2,
        ff_dim=64,
        num_encoder_blocks=1,
        num_decoder_blocks=1,
        mlp_units=[16],
        dropout=0.0,  # No dropout for consistent results
        decoder_strategy='zeros'  # Use zeros for deterministic behavior
    )
    
    # Sample data
    X_sample = np.random.randn(2, 5, 1).astype(np.float32)
    y_sample = np.random.randn(2, 1).astype(np.float32)
    y_reshaped = y_sample.reshape(-1, 1, 1)
    
    print("Sample encoder inputs:")
    print(X_sample.squeeze())
    print("Sample targets:")
    print(y_sample.squeeze())
    
    # Inference mode (only encoder inputs)
    print("\nInference Mode (only encoder inputs):")
    pred_inference = model(X_sample, training=False)
    print(f"Predictions: {pred_inference.numpy().squeeze()}")
    
    # Training mode (encoder inputs + targets)
    print("\nTraining Mode (encoder inputs + targets):")
    pred_training = model([X_sample, y_reshaped], training=True)
    print(f"Predictions: {pred_training.numpy().squeeze()}")
    
    print("\nExplanation:")
    print("- In inference mode: Model uses learned/zero decoder inputs")
    print("- In training mode: Model uses shifted ground truth as decoder inputs")
    print("- This allows the model to learn better representations during training")
    
    return model, pred_inference, pred_training


def main():
    """Main demonstration function."""
    print("Starting Teacher Forcing Encoder-Decoder Demonstration...")
    
    try:
        # 1. Demonstrate teacher forcing input creation
        teacher_inputs, targets = demonstrate_teacher_forcing_inputs()
        
        # 2. Compare architectures
        arch_results = compare_architectures()
        
        # 3. Training performance comparison
        training_results = run_training_comparison()
        
        # 4. Demonstrate inference modes
        inference_results = demonstrate_inference_modes()
        
        print("\n" + "="*80)
        print("DEMONSTRATION SUMMARY")
        print("="*80)
        
        print("✅ Teacher forcing input creation demonstrated")
        print("✅ Architecture comparison completed")
        print("✅ Training performance comparison completed")
        print("✅ Inference modes demonstrated")
        
        print("\nKey Benefits of Teacher Forcing:")
        print("  • Faster convergence during training")
        print("  • Better gradient flow through decoder")
        print("  • More stable training process")
        print("  • Improved final performance")
        
        print("\nUsage Guidelines:")
        print("  • Use teacher forcing for sequence-to-sequence tasks")
        print("  • Particularly effective for time series forecasting")
        print("  • Start with simple configurations and scale up")
        print("  • Monitor for overfitting with teacher forcing")
        
        print("\nImplementation in Pipeline:")
        print("  1. Set use_encoder_decoder=True")
        print("  2. Set use_teacher_forcing=True")
        print("  3. Configure start_token_value (usually 0.0)")
        print("  4. Use UnivariateTransformerPipeline.create_teacher_forcing_config()")
        
        # Show improvement summary
        if 'improvements' in training_results:
            improvements = training_results['improvements']
            print(f"\nPerformance Improvement:")
            print(f"  Loss: {improvements['loss']:+.2f}%")
            print(f"  MAE:  {improvements['mae']:+.2f}%")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()