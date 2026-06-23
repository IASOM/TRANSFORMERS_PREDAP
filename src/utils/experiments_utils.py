import os
from datetime import datetime
from omegaconf import OmegaConf
import pandas as pd
from src.config.base_transformer_config import BaseTransformerConfig as default_config
import json
import matplotlib.pyplot as plt
import gc
import ctypes
from tensorflow.keras import backend as K
import tensorflow as tf
from typing import List
import sys
from pathlib import Path
from omegaconf import DictConfig, OmegaConf

import re

from src.training.training_residual_transformer import load_trained_model


_original_read_csv = pd.read_csv
_in_smart_read = False

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


def save_model_parameters(config, model, model_name: str, model_parameters_path: str):
    """Save model parameters to a JSON file for later reference."""
    params = {
        "target_code": config.code,
        "lookback": config.lookback,
        "forecast_horizon": config.forecast,
        "model_type": "transformer",
        "dataset": config.data_path,
        "activation_function": config.activation_function,
        "covid_token": config.covid_token,
        "cutoff_date": config.cutoff_date,
        "head_size": config.head_size,
        "num_heads": config.num_heads, 
        "ff_dim": config.ff_dim,
        "mlp_units": list(config.mlp_units) if config.mlp_units else [],
        "num_transformer_blocks": config.num_transformer_blocks,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate,
        "positional_encoding": config.positional_encoding,
        "scaler": str(config.scaler),
        "num_layers": len(model.layers),
        "layer_types": [type(layer).__name__ for layer in model.layers],
        # Force TensorFlow integers into standard Python integers
        "total_params": int(model.count_params()),
        "trainable_params": int(sum(tf.keras.backend.count_params(p) for p in model.trainable_weights)),
        "non_trainable_params": int(sum(tf.keras.backend.count_params(p) for p in model.non_trainable_weights))
    }

    # 1. Clean and truncate the model name
    base_name = model_name.removesuffix('.keras').split('_LEARNING')[0]
    base_name = base_name + f"_lb{config.lookback}_fh{config.forecast}"  # Add lookback and forecast to filename for clarity
    
    # 2. Form the full parameters filename
    params_filename = f"{base_name}_parameters.json"
    params_path = os.path.join(model_parameters_path, config.code, params_filename)
    
    # 3. Create directory if it doesn't exist
    os.makedirs(os.path.dirname(params_path), exist_ok=True)
    
    # 4. Save your file safely using the 'params' dictionary directly
    with open(params_path, 'w') as f:
        json.dump(params, f, indent=4)
        
    print(f"✅ Model parameters saved to: {params_path}")

def load_json_codes_list(json_path: str) -> str:
    """Load a list from JSON and return as comma-separated string for Hydra sweep."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    cleaned_codes = []
    for code in data:
        clean_str = str(code).strip()
        
        # Filter out invalid columns

        cleaned_codes.append(f'"{clean_str}"')
            
    # This will return: "DEMAND_DEMANDA_TOTAL","DEMAND_... BARCELONA CIUTAT","..."
    return ",".join(cleaned_codes)

def load_codes(codes_path: str) -> str:
    """Load a list from either JSON or Excel and return as comma-separated string for Hydra sweep."""
    if codes_path.endswith('.json'):
        with open(codes_path, 'r') as f:
            data = json.load(f)
        
        cleaned_codes = []
        for code in data:
            clean_str = str(code).strip()
            
            # Filter out invalid columns

            cleaned_codes.append(f'"{clean_str}"')
        return cleaned_codes
    elif codes_path.endswith('.xlsx') or codes_path.endswith('.xls'):
        return load_excel_codes_list(codes_path)
    else:
        raise ValueError("Unsupported file format. Please provide a JSON or Excel file.")

def load_inference_codes_list(models_dir: str) -> List[str]:
    """Load a list of codes from the quantized models directory for inference."""
    folder_names = [
            entry.name for entry in os.scandir(models_dir) if entry.is_dir()
        ]
    
    cleaned_codes = []
    for code in folder_names:
        clean_str = str(code).strip()
        
        # Filter out invalid columns

        cleaned_codes.append(f'"{clean_str}"')
            
    # This will return: "DEMAND_DEMANDA_TOTAL","DEMAND_... BARCELONA CIUTAT","..."
    return ",".join(cleaned_codes)
    
def get_dates_list(start_date: str, end_date: str) -> List[str]:
    dates_list = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y-%m-%d').tolist()
    return dates_list#','.join([f'"{date}"' for date in dates_list])

def load_codes_list_from_excel(excel_path: str) -> str:
    """Load a list from Excel and return as comma-separated string for Hydra sweep."""
    df = pd.read_excel(excel_path, engine='openpyxl')

    codes_list = df['nom_variable'].tolist()
    
    return codes_list


def excel_to_json_codes_list(excel_path: str, json_output_path: str):
    """
    Reads an Excel file and converts a specific column to a JSON list.
    
    Args:
        excel_path (str): Path to the input Excel file.
        json_output_path (str): Path to save the output JSON file.
    """
    df = pd.read_excel(excel_path, engine='openpyxl')
    
    # Assuming the column of interest is named 'nom_variable'
    if 'nom_variable' not in df.columns:
        raise ValueError("The Excel file must contain a 'nom_variable' column.")
    
    codes_list = df['nom_variable'].dropna().unique().tolist()
    
    # Save to JSON
    with open(json_output_path, 'w') as f:
        json.dump(codes_list, f, indent=4)
    
    print(f"✅ Converted Excel codes list saved to: {json_output_path}")

def load_excel_codes_list(excel_path: str) -> str:
    df = pd.read_excel(excel_path, engine='openpyxl')

    codes_list = df['nom_variable'].tolist()
    
    cleaned_codes = []
    for code in codes_list:
        clean_str = str(code).strip()
        
        # Filter out invalid columns
        if clean_str and clean_str != 'timestamp' and not clean_str.startswith('__index'):
            # CRUCIAL FIX: Wrap the code in escaped quotes so Hydra treats 
            # "BARCELONA CIUTAT" as a single literal string item.
            cleaned_codes.append(f'"{clean_str}"')
            
    # This will return: "DEMAND_DEMANDA_TOTAL","DEMAND_... BARCELONA CIUTAT","..."
    return ",".join(cleaned_codes)

def get_codes_list(input_directory: str) -> str:
    """
    Reads the input data file, extracts unique codes, and returns them
    individually wrapped in quotes as a comma-separated string for Hydra sweeps.
    """
    if input_directory.endswith('.csv'):
        df = pd.read_csv(input_directory, nrows=0)
        codes_list = df.columns.tolist()
    elif input_directory.endswith('.parquet'):
        import pyarrow.parquet as pq
        schema = pq.read_schema(input_directory)
        codes_list = schema.names
    else:
        raise ValueError("Unsupported file format. Please provide a CSV or Parquet file.")

    cleaned_codes = []
    for code in codes_list:
        clean_str = str(code).strip()
        
        # Filter out invalid columns
        if clean_str and clean_str != 'timestamp' and not clean_str.startswith('__index'):
            # CRUCIAL FIX: Wrap the code in escaped quotes so Hydra treats 
            # "BARCELONA CIUTAT" as a single literal string item.
            cleaned_codes.append(f'"{clean_str}"')
            
    # This will return: "DEMAND_DEMANDA_TOTAL","DEMAND_... BARCELONA CIUTAT","..."
    return ",".join(cleaned_codes)


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
    tf.keras.backend.clear_session()
    
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


def extract_model_params(model_name):
    """
    Extract lookback and forecast parameters from model filename.
    
    Expected format: {code}_example_transformer_{forecast}fh_{ff_dim}ff_{lookback}lb_{lr}initlr.keras
    
    Args:
        model_name (str): Model filename
        
    Returns:
        tuple: (lookback, forecast) or (None, None) if not found
    """
    # Pattern to match the model name format
    pattern = r'(\d+)fh_\d+ff_(\d+)lb_'
    
    match = re.search(pattern, model_name)
    if match:
        forecast = int(match.group(1))  # First group is forecast
        lookback = int(match.group(2))  # Second group is lookback
        return lookback, forecast
    else:
        print(f"Could not extract parameters from: {model_name}")
        return None, None



def check_for_help_flag(cfg: DictConfig):
    # Check if user passed help=True or help=true via CLI
    if "help" in cfg and cfg.help:
        print("\n💡 PREDAP Pipeline CLI Parameter Help Guide:")
        print("-" * 50)
        
        # Flatten and isolate groups
        for group_name, group_content in cfg.items():
            if isinstance(group_content, DictConfig):
                print(f"\n[{group_name.upper()} GROUP]")
                for param, value in group_content.items():
                    print(f"  {group_name}.{param:<25} -> current default: {value}")
        
        print("\nUsage example:")
        print("  python script.py model.target_code='M54' model.lookback=90\n")
        sys.exit(0)

