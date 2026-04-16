import os
from datetime import datetime
import pandas as pd
from src.config.base_transformer_config import BaseTransformerConfig as default_config
import json
import matplotlib.pyplot as plt
import gc
import ctypes
from tensorflow.keras import backend as K
import tensorflow as tf

_original_read_csv = pd.read_csv

def cleanup_ram():
    plt.close('all')
    K.clear_session()
    gc.collect()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass

def smart_read(file_path, **kwargs):
    """
    Función que sustituye a pd.read_csv.
    Detecta automáticamente si la extensión es .parquet o .csv
    y llama a la función de lectura apropiada.
    """
    if str(file_path).lower().endswith('.parquet'):
        print(f"-> INFO: Leyendo {file_path} como PARQUET.")
        # Aquí puedes añadir parámetros específicos para Parquet si los necesitas
        return pd.read_parquet(file_path, **kwargs)
    else:
        # Llama a la función original pd.read_csv para CSVs y otros
        print(f"-> INFO: Leyendo {file_path} como CSV (o formato predeterminado).")
        return _original_read_csv(file_path, **kwargs)
    

# Utility function for safe float conversion
def safe_float(value):
    """Convert value to float, handling numpy types and NaN values"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None
    
def initialize_results_tracking(results_dir: str, codes_list) -> dict:
    """Initialize result tracking directories and structures."""
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved in: {results_dir}/")
    
    # Get all possible target codes from default config for initialization
    best_results_per_code = {}
    for code in codes_list:
        best_results_per_code[code] = {
            "best_mse": float('inf'),
            "best_config": None,
            "best_metrics": None,
            "best_run_info": None
        }

    return best_results_per_code

def load_json_codes_list(json_path: str) -> str:
    """Load a list from JSON and return as comma-separated string for Hydra sweep."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    #codes_list = data[key]
    # Return comma-separated string for Hydra sweep parameters
    return ','.join(data)


def compute_dynamic_batch_size(lookback, forecast):
    """
    Computes an appropriate batch size based on lookback and forecast parameters.

    Parameters:
    - lookback (int): Number of past timesteps used for input sequences.
    - forecast (int): Number of timesteps predicted.

    Returns:
    - int: Computed batch size.
    """
    gpus = tf.config.list_physical_devices('GPU')

    if lookback <= 14 and forecast <= 14:
        batch_size = 4096
    if lookback <= 30 and forecast <= 30:
        batch_size = 1024
    elif (30 <= lookback <= 60) and forecast <= 60:
        batch_size = 256
    elif 60 <= lookback <= 128 and forecast<= 128:
        batch_size = 128
    elif 128 < lookback <= 365 and forecast <=365:
        batch_size = 128
    else:
        batch_size = 92
    

    if len(gpus) == 0:
        batch_size = 32
        
    return batch_size


def memory_cleanup():
    """
    Clean up GPU memory between training phases.
    
    NOTE: Do NOT use numba cuda.device.reset() — it destroys the CUDA context
    and makes TensorFlow unable to use the GPU for the rest of the process.
    """
    # Clear Keras session (releases model graphs and cached tensors)
    K.clear_session()
    
    # Reset the default graph (TF1 compat, still useful for freeing resources)
    tf.compat.v1.reset_default_graph()
    
    # Force Python garbage collection
    gc.collect()
    
    # Close all matplotlib figures to free memory
    plt.close('all')
    
    # Free unused C memory back to the OS
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    
    # Reset GPU memory stats (does NOT destroy the CUDA context)
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        try:
            tf.config.experimental.reset_memory_stats(gpu)
        except Exception:
            pass