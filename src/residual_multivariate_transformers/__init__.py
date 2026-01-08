"""
Residual Multivariate Transformers Package
==========================================

This package contains modules for implementing residual correction using multivariate
transformer models. It follows a modular structure for better code organization and
maintainability.

Modules:
    - config_residual_transformer: Configuration parameters and settings
    - model_architecture_residual_transformer: Model architectures and learning rate schedulers  
    - training_evaluation_residual_transformer: Training and evaluation utilities
    - utils_residual_transformer: Data processing and utility functions
    - visualization_residual_transformer: Plotting and visualization functions
    - main_training_residual_transformer: Main training and evaluation pipeline
"""

from .model_architecture_residual_transformer import *
from .training_evaluation_residual_transformer import *
from .utils_residual_transformer import *
from .visualization_residual_transformer import *

__version__ = "1.0.0"
__author__ = "TRANSFORMERS_PREDAP Team"