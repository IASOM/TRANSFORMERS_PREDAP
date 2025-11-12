"""
Configuration and Parameters for Univariate Transformer
======================================================
Centralized configuration file for model parameters and settings.
"""


import keras


class TransformerConfig:
    """Configuration class for transformer model parameters."""
    
    def __init__(self):
        # Model Architecture Parameters
        self.HEAD_SIZE = 2              # dimensions of each attention head
        self.NUM_HEADS = 2              # number of attention heads  
        self.NUM_TRANSFORMER_BLOCKS = 2 # number of transformer layers
        self.FF_DIM = 8                 # dimensionality of feed-forward layer
        self.ACTIVATION = 'tanh'        # activation function

        
        # Time Series Parameters
        self.FORECAST = 7      # number of future time steps to predict (prediction horizon)
        self.LOOKBACK = 7      # number of past time steps as input (input sequence length)
        
        # Multi-Layer Perceptron (MLP) Parameters
        self.MLP_UNITS = 32     # number of neurons in fully connected layers
        self.MLP_DROPOUT = 0.5 # dropout rate for MLP layers
        
        # Regularization Parameters
        self.DROPOUT = 0.2      # dropout rate for transformer layers
        
        # Training Parameters
        self.LEARNING_RATE = 5e-5     # initial learning rate
        self.EPOCHS = 100             # maximum number of training epochs
        self.BATCH_SIZE = 16           # batch size for training
        self.EARLY_STOP_PATIENCE = 20  # early stopping patience
        self.VALIDATION_SPLIT = 0.3    # validation data split ratio
        
        # Learning Rate Schedule Parameters
        self.LR_WARMUP_RATIO = 0.2     # warmup steps as ratio of total epochs
        self.LR_MAX_MULTIPLIER = 100   # max LR = initial_lr * this value
        self.LR_MIN_MULTIPLIER = 10    # min LR = initial_lr * this value
        
        # Data Parameters
        self.DATA_PATH = "src/data/longitudinalitat_DIAGNOSTICS_GROUPED_timestamp.csv"
        
        self.TARGET_CODE = "T14"       # diagnostic code to predict
        self.CODES_LIST = ["J00","T14","M54"]
        self.DATE_CUTOFF = '2008-01-01'  # filter data from this date
        self.TRAIN_SPLIT = 0.8  
               # train/test split ratio
        
        # Directory Parameters
        self.MODEL_DIR = "transformer_outputs/models_covid_token"
        self.PLOTS_DIR = "transformer_outputs/plots"
        self.LOGS_DIR = "logs"
        
        # Evaluation Parameters
        self.SLIDING_WINDOW = 10       # window size for sliding evaluation
        self.SHUFFLE_DATA = False      # whether to shuffle training data
        self.COVID_TOKEN = True      # whether to include COVID-19 token feature
        self.SAVE_TRAIN_HISTORY = True        # whether to save training history

        # Hyperparameter Search Lists
        self.LOOKBACK_LIST = [7, 14, 30, 60, 182, 365]
        self.FORECAST_LIST = [7, 14, 30, 60, 182, 365]
        self.HEAD_SIZE_LIST = [2, 4, 8]
        self.NUM_HEADS_LIST = [2, 4, 8]
        self.ACTIVATIONS_LIST = [keras.activations.gelu, keras.activations.tanh, keras.activations.relu]
        self.COVID_TOKEN_LIST = [False, True]
        self.HEAD_SIZE_LIST = [2, 8, 16, 32]
        self.NUM_HEADS_LIST = [2, 4, 8]
        self.FF_DIM_LIST = [8, 16, 32, 64]
        self.MLP_UNITS_LIST = [16, 32, 64, 128]

        
        
    def get_model_name(self, code=None, forecast=None, ff_dim=None, lookback=None, lr=None):
        """Generate standardized model name."""
        code = code or self.TARGET_CODE
        forecast = forecast or self.FORECAST
        ff_dim = ff_dim or self.FF_DIM
        lookback = lookback or self.LOOKBACK
        lr = lr or self.LEARNING_RATE
        
        return f'{code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{lr}initlr.keras'

    def get_lr_schedule_params(self, learning_rate=None):
        """Get learning rate schedule parameters."""
        if learning_rate is not None:
            lr_init = learning_rate
        else:
            lr_init = self.LEARNING_RATE
        lr_max = lr_init * self.LR_MAX_MULTIPLIER
        lr_min = lr_init * self.LR_MIN_MULTIPLIER
        warmup_steps = int(self.EPOCHS * self.LR_WARMUP_RATIO)
        
        return {
            'initial_lr': lr_init,
            'max_lr': lr_max,
            'min_lr': lr_min,
            'warmup_steps': warmup_steps,
            'total_steps': self.EPOCHS
        }
    
    def print_config(self, head_size=None, num_heads=None, num_transformer_blocks=None,
                     ff_dim=None, lookback=None, forecast=None, learning_rate=None, epochs=None,
                     batch_size=None, early_stop_patience=None, target_code=None, date_cutoff=None):
        """Print current configuration."""
        print("="*50)
        print("TRANSFORMER CONFIGURATION")
        print("="*50)
        print(f"Model Architecture:")
        print(f"  - Head Size: {head_size}")
        print(f"  - Number of Heads: {num_heads}")
        print(f"  - Transformer Blocks: {num_transformer_blocks}")
        print(f"  - Feed Forward Dimension: {ff_dim}")
        print(f"\nTime Series:")
        print(f"  - Lookback: {lookback}")
        print(f"  - Forecast: {forecast}")
        print(f"\nTraining:")
        print(f"  - Learning Rate: {learning_rate}")
        print(f"  - Epochs: {epochs}")
        print(f"  - Batch Size: {batch_size}")
        print(f"  - Early Stop Patience: {early_stop_patience}")
        print(f"\nData:")
        print(f"  - Target Code: {target_code}")
        print(f"  - Date Cutoff: {date_cutoff}")
        print("="*50)


# Pandemic waves configuration
PANDEMIC_WAVES = {
    "Primera Onada": ("2020-03", "2020-06"),
    "Segona Onada": ("2020-10", "2020-12"),
    "Tercera Onada": ("2021-01", "2021-03"),
    "Quarta Onada": ("2021-04", "2021-06"),
    "Cinquena Onada": ("2021-07", "2021-09")
}

# Default configuration instance
default_config = TransformerConfig()


# Utility function to create custom configurations
def create_config(lookback=7, forecast=7, head_size=2, num_heads=2, ff_dim=8, 
                 learning_rate=0.001, epochs=100, target_code="T14", model_dir="models_covid_token",
                 activations_list=['tanh', 'relu'], covid_token_list=[True, False]):
    """
    Create a custom configuration with specified parameters.
    
    Args:
        lookback: Number of past time steps
        forecast: Number of future time steps to predict
        head_size: Dimension of attention heads
        num_heads: Number of attention heads
        ff_dim: Feed-forward layer dimension
        learning_rate: Initial learning rate
        epochs: Number of training epochs
        target_code: Diagnostic code to predict
        
    Returns:
        TransformerConfig: Configured instance
    """
    config = TransformerConfig()
    
    config.LOOKBACK = lookback
    config.FORECAST = forecast
    config.HEAD_SIZE = head_size
    config.NUM_HEADS = num_heads
    config.FF_DIM = ff_dim
    config.LEARNING_RATE = learning_rate
    config.EPOCHS = epochs
    config.TARGET_CODE = target_code
    config.MODEL_DIR = model_dir
    config.ACTIVATIONS_LIST = activations_list
    config.COVID_TOKEN_LIST = covid_token_list

    
    return config