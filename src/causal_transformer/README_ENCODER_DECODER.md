# Encoder-Decoder Transformer Architecture with Teacher Forcing

This document describes the newly implemented encoder-decoder transformer architecture within the causal transformer framework, including support for teacher forcing training.

## Overview

The encoder-decoder architecture provides a more sophisticated approach to sequence modeling by separating the encoding and decoding phases. This implementation includes proper cross-attention mechanisms and **teacher forcing** support, where shifted ground truth targets are used as decoder inputs during training for improved performance and faster convergence.

## Architecture Components

### 1. Transformer Encoder
- **Self-attention**: Bidirectional attention over input sequences
- **Position encoding**: Optional positional information
- **Multiple layers**: Configurable number of encoder blocks
- **Feed-forward networks**: Position-wise transformations

### 2. Transformer Decoder  
- **Masked self-attention**: Causal attention for autoregressive generation
- **Cross-attention**: Attends to encoder outputs (key innovation)
- **Position encoding**: Applied to decoder inputs
- **Multiple layers**: Configurable number of decoder blocks

### 3. Teacher Forcing Support
- **Training mode**: Uses shifted ground truth as decoder inputs
- **Inference mode**: Uses learned embeddings or zeros
- **Automatic handling**: Model wrapper manages input switching
- **Better convergence**: Faster training and improved performance

### 4. Key Improvements
- **Cross-attention mechanism**: Decoder can access encoder representations
- **Teacher forcing**: Uses ground truth during training for better gradients
- **Learnable decoder inputs**: Better initialization than zeros
- **Flexible configuration**: Separate control of encoder/decoder complexity
- **Causal masking**: Proper autoregressive behavior

## Usage

### Basic Configuration (Standard Encoder-Decoder)

```python
from main_train_univ_causal import UnivariateTransformerPipeline

# Create encoder-decoder configuration
config = UnivariateTransformerPipeline.create_encoder_decoder_config(
    lookback=60,
    forecast=1,
    code="T14",
    num_encoder_blocks=4,      # Number of encoder layers
    num_decoder_blocks=2,      # Number of decoder layers
    decoder_target_length=1,   # Decoder sequence length
    decoder_strategy='learned', # 'learned', 'zeros', or 'random'
    head_size=256,
    num_heads=8,
    ff_dim=512,
    dropout=0.1
)

# Run training
pipeline = UnivariateTransformerPipeline(config)
model, model_name, loss, mae, mse = pipeline.run_complete_pipeline()
```

### Teacher Forcing Configuration (Recommended)

```python
# Create teacher forcing configuration
config = UnivariateTransformerPipeline.create_teacher_forcing_config(
    lookback=60,
    forecast=1,
    code="T14",
    num_encoder_blocks=4,
    num_decoder_blocks=2,
    decoder_target_length=1,
    start_token_value=0.0,      # Start-of-sequence token value
    decoder_strategy='learned',
    head_size=256,
    num_heads=8,
    ff_dim=512,
    dropout=0.1
)

# Run training with teacher forcing
pipeline = UnivariateTransformerPipeline(config)
model, model_name, loss, mae, mse = pipeline.run_complete_pipeline()
```

### Manual Configuration

```python
from main_train_univ_causal import TransformerUnivConfig

# Standard encoder-decoder
config = TransformerUnivConfig(
    # Standard parameters
    lookback=60,
    forecast=1,
    code="T14",
    
    # Enable encoder-decoder
    use_encoder_decoder=True,
    
    # Architecture parameters
    num_encoder_blocks=4,
    num_decoder_blocks=2,
    decoder_target_length=1,
    decoder_strategy='learned',
    
    # Attention parameters
    head_size=256,
    num_heads=8,
    ff_dim=512,
    dropout=0.1
)

# Teacher forcing encoder-decoder
config = TransformerUnivConfig(
    # Standard parameters
    lookback=60,
    forecast=1,
    code="T14",
    
    # Enable encoder-decoder with teacher forcing
    use_encoder_decoder=True,
    use_teacher_forcing=True,
    
    # Architecture parameters
    num_encoder_blocks=4,
    num_decoder_blocks=2,
    decoder_target_length=1,
    decoder_strategy='learned',
    start_token_value=0.0,
    
    # Attention parameters
    head_size=256,
    num_heads=8,
    ff_dim=512,
    dropout=0.1
)
```

## Configuration Parameters

### Encoder-Decoder Specific

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `use_encoder_decoder` | bool | False | Enable encoder-decoder architecture |
| `num_encoder_blocks` | int | 4 | Number of encoder transformer blocks |
| `num_decoder_blocks` | int | 2 | Number of decoder transformer blocks |
| `decoder_target_length` | int | 1 | Length of decoder sequence |
| `decoder_strategy` | str | 'learned' | Decoder input strategy |
| `use_teacher_forcing` | bool | False | Enable teacher forcing during training |
| `start_token_value` | float | 0.0 | Value for start-of-sequence token |

### Decoder Input Strategies

1. **'learned'**: Learnable parameter embeddings (recommended)
   - Trainable parameters that adapt during training
   - Better performance for most tasks

2. **'zeros'**: Zero-initialized inputs
   - Simple baseline approach
   - Faster initialization

3. **'random'**: Random noise inputs
   - Adds stochasticity to decoder
   - Can help with regularization

## Teacher Forcing Explained

Teacher forcing is a training technique where the model uses the actual ground truth from prior time steps as input, rather than its own predictions. This leads to faster convergence and better performance.

### How Teacher Forcing Works

**Without Teacher Forcing (Standard Decoder):**
```
Training: Decoder inputs = [learned_embeddings] → Predictions
Inference: Decoder inputs = [learned_embeddings] → Predictions
```

**With Teacher Forcing:**
```
Training:  Decoder inputs = [START, y₀, y₁, ..., y_{n-2}] → Predictions = [y₁, y₂, ..., y_{n-1}]
Inference: Decoder inputs = [learned_embeddings] → Predictions
```

### Teacher Forcing Process

1. **Input Preparation**: Ground truth targets are shifted right with a start token
   - Original targets: `[y₁, y₂, y₃, y₄]`
   - Decoder inputs: `[START, y₁, y₂, y₃]`
   - Target outputs: `[y₁, y₂, y₃, y₄]`

2. **Training Phase**: Model learns to predict next value given previous ground truth
3. **Inference Phase**: Model uses learned embeddings instead of ground truth

### Benefits of Teacher Forcing

- ✅ **Faster Convergence**: Better gradient flow during training
- ✅ **Stable Training**: Reduces exposure bias during early training
- ✅ **Better Performance**: Often achieves lower final loss
- ✅ **Robust Learning**: Model learns proper sequence dependencies

## Architecture Comparison

### Standard Transformer (Encoder-Only)
```
Input → Encoder Blocks → Global Pool → MLP → Output
```

### Encoder-Decoder Transformer
```
Input → Encoder Blocks → Cross-Attention
                              ↓
Decoder Inputs → Decoder Blocks → MLP → Output
```

### Benefits of Encoder-Decoder

1. **Separation of concerns**: Encoding and decoding are separate processes
2. **Cross-attention**: Decoder can selectively attend to input features
3. **Autoregressive capability**: Natural support for sequence generation
4. **Flexible target lengths**: Can handle variable output sequences
5. **Better representation**: Dedicated decoder processing

## Implementation Details

### Cross-Attention Mechanism

The decoder implements proper cross-attention where:
- **Query (Q)**: Comes from decoder representations
- **Key (K) and Value (V)**: Come from encoder outputs
- **No causal masking**: Decoder can attend to all encoder positions

```python
def transformer_decoder(decoder_inputs, encoder_outputs, ...):
    # 1. Masked self-attention (causal)
    x = MultiHeadAttention(use_causal_mask=True)(decoder_inputs, decoder_inputs)
    
    # 2. Cross-attention (no masking)
    x = MultiHeadAttention()(x, encoder_outputs)  # Q from decoder, K,V from encoder
    
    # 3. Feed-forward network
    x = FeedForwardNetwork(x)
```

### Model Building Process

1. **Encoder**: Processes input sequence with bidirectional attention
2. **Decoder initialization**: Creates decoder inputs based on strategy
3. **Cross-attention layers**: Connect encoder outputs to decoder
4. **Output projection**: Maps decoder outputs to predictions

## Performance Considerations

### Computational Complexity
- **Encoder**: O(n²) for self-attention over input sequence
- **Decoder**: O(m²) for self-attention + O(n×m) for cross-attention
- **Total**: Higher than encoder-only but more expressive

### Memory Usage
- **Additional parameters**: Decoder blocks and learnable embeddings
- **Attention matrices**: Cross-attention requires encoder-decoder alignment
- **Typical increase**: 20-50% more parameters than encoder-only

### Training Speed
- **Slightly slower**: Due to additional decoder computations
- **Better convergence**: Often requires fewer epochs
- **Overall efficiency**: Comparable or better for complex tasks

## Examples and Demonstrations

### Quick Test
```python
# Run the encoder-decoder demonstration
python encoder_decoder_demo.py

# Run the teacher forcing demonstration
python teacher_forcing_demo.py
```

### Training Examples
```python
# Standard transformer
python main_train_univ_causal.py  # Choose option 1

# Encoder-decoder transformer  
python main_train_univ_causal.py  # Choose option 2

# Teacher forcing encoder-decoder transformer (recommended)
python main_train_univ_causal.py  # Choose option 3
```

### Custom Configuration
```python
from main_train_univ_causal import UnivariateTransformerPipeline

# Heavy encoder, light decoder
config = UnivariateTransformerPipeline.create_encoder_decoder_config(
    num_encoder_blocks=6,
    num_decoder_blocks=1,
    head_size=512,
    ff_dim=1024
)

# Balanced configuration
config = UnivariateTransformerPipeline.create_encoder_decoder_config(
    num_encoder_blocks=4,
    num_decoder_blocks=3,
    decoder_strategy='learned'
)

# Light encoder, heavy decoder
config = UnivariateTransformerPipeline.create_encoder_decoder_config(
    num_encoder_blocks=2,
    num_decoder_blocks=4,
    decoder_target_length=3  # Multi-step output
)
```

## Integration with Existing Code

The encoder-decoder architecture is fully integrated with the existing causal transformer pipeline:

- ✅ **Backward compatible**: Existing code continues to work
- ✅ **Same configuration system**: Uses TransformerUnivConfig
- ✅ **Same training pipeline**: UnivariateTransformerPipeline handles both
- ✅ **Same evaluation**: Existing metrics and evaluation code work
- ✅ **Same data preparation**: No changes needed to data processing

## Files Modified

### Core Architecture
- `main_architechture_univ_causal.py`: Added encoder-decoder functions
  - `transformer_decoder()`: New decoder with cross-attention
  - `build_encoder_decoder_model()`: Core model builder
  - `build_encoder_decoder_wrapper()`: Convenience wrapper
  - `EncoderDecoderWrapper`: Auto-handles decoder inputs

### Training Pipeline  
- `main_train_univ_causal.py`: Enhanced configuration and training
  - Added encoder-decoder parameters to `TransformerUnivConfig`
  - Updated `build_model()` to support both architectures
  - Added `create_encoder_decoder_config()` convenience method
  - Enhanced examples and demonstrations

### Demonstration
- `encoder_decoder_demo.py`: Comprehensive demonstration script
  - Architecture comparison
  - Attention mechanism visualization
  - Training performance comparison
  - Usage examples

## Troubleshooting

### Common Issues

1. **Import errors**: Ensure you're importing from the causal_transformer module
2. **Shape mismatches**: Check decoder_target_length matches your use case
3. **Memory issues**: Reduce num_heads or head_size for large sequences
4. **Slow training**: Try fewer decoder_blocks first, then optimize

### Configuration Tips

1. **Start simple**: Use 2-4 encoder blocks, 1-2 decoder blocks
2. **Decoder strategy**: 'learned' usually performs best
3. **Target length**: Use 1 for forecasting, higher for sequence generation
4. **Attention heads**: 4-8 heads work well for most problems

### Performance Optimization

1. **Encoder-heavy**: More encoder blocks for better input representation
2. **Decoder-heavy**: More decoder blocks for complex output patterns
3. **Balanced**: Equal encoder/decoder blocks for general use
4. **Memory constrained**: Reduce head_size and ff_dim proportionally

## Future Enhancements

Potential improvements to consider:

1. **Multi-head cross-attention variants**
2. **Relative position encoding**
3. **Sparse attention patterns**
4. **Dynamic decoder length**
5. **Beam search for generation**
6. **Attention visualization tools**

## References

- Attention Is All You Need (Vaswani et al., 2017)
- The Transformer architecture
- Cross-attention mechanisms in sequence-to-sequence models