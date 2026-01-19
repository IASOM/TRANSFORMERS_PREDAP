import os
from datetime import datetime
import pandas as pd
from src.config.base_transformer_config import BaseTransformerConfig as default_config
import json
import matplotlib.pyplot as plt
import gc
import ctypes
from tensorflow.keras import backend as K

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