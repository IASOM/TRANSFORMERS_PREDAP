import pandas as pd
import os

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

# --- REEMPLAZO (MONKEY PATCHING) ---

# 1. Guardar la función original de lectura de CSV
#    (Usaremos esta referencia dentro de nuestro wrapper)
_original_read_csv = pd.read_csv

# 2. Reemplazar la función original con nuestra función "inteligente"
pd.read_csv = smart_read

import tensorflow as tf
import numpy as np
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.tensorflow
import mlflow.keras
from datetime import datetime

import tempfile
import json
import pickle
import hydra
from omegaconf import DictConfig, OmegaConf
from hydra.core.global_hydra import GlobalHydra
from tensorflow.keras import backend as K
import gc


#load and visualize data 

#from src.univariate_transformer import default_config
from src import data_preparation



from src.main_train_univ_transformer_class import (
    TransformerUnivConfig,
    UnivariateTransformerPipeline,
)

from src.main_train_diagnostic_residual_transformer_class import(
    DiagnosticResidualTransformerConfig,
    DiagnosticResidualTransformerPipeline,
)

from src.main_train_seasonal_residual_transformer_class import(
    SeasonalResidualTransformerConfig,
    SeasonalResidualTransformerPipeline,
)

from src.univariate_transformer.utils_univ_transformer import load_mlflow_model_history 

from src.config.base_transformer_config import BaseTransformerConfig

default_config = BaseTransformerConfig()




gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


# Utility function for safe float conversion
def safe_float(value):
    """Convert value to float, handling numpy types and NaN values"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

# Global variables for tracking results across runs
best_results_per_code = {}
results_dir = "best_hyperparameters_results"

def initialize_results_tracking():
    """Initialize result tracking directories and structures."""
    global best_results_per_code, results_dir
    
    os.makedirs(results_dir, exist_ok=True)
    print(f"📁 Results will be saved in: {results_dir}/")
    
    # Get all possible target codes from default config for initialization
    codes_list = default_config.CODES_LIST
    best_results_per_code = {}
    for code in codes_list:
        best_results_per_code[code] = {
            "best_mse": float('inf'),
            "best_config": None,
            "best_metrics": None,
            "best_run_info": None
        }

def load_json_codes_list(json_path: str) -> str:
    """Load a list from JSON and return as comma-separated string for Hydra sweep."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    #codes_list = data[key]
    # Return comma-separated string for Hydra sweep parameters
    return ','.join(data)

OmegaConf.register_new_resolver("load_json_codes_list", load_json_codes_list)

@hydra.main(version_base=None, config_path="conf", config_name="grid_search_V1.yaml")
def main_experiment(cfg: DictConfig) -> None:
    """Main experiment function decorated with Hydra for parameter sweeping."""
    
    # Initialize MLflow
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    experiment_name = f"NEW_CODES_{cfg.mlflow.experiment_name}_{datetime.now().strftime('%Y%m%d')}"
    mlflow.set_experiment(experiment_name)
    
    print(f"🎯 MLflow tracking initialized")
    print(f"   • Experiment: {experiment_name}")
    print(f"   • Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   • View results at: http://localhost:5000")
    
    # Initialize results tracking (only once per sweep)
    initialize_results_tracking()
    
    # Extract parameters from Hydra config
    CODE = cfg.model.target_code
    lookback = cfg.model.lookback
    forecast = cfg.model.forecast
    head_size = cfg.model.head_size
    num_heads = cfg.model.num_heads
    ff_dim = cfg.model.ff_dim
    mlp_units = cfg.model.mlp_units
    activation_function = cfg.model.activation
    covid_token = cfg.model.covid_token
    cutoff_date = cfg.training.cutoff_date
    positional_encoding = cfg.training.positional_encoding
    data_path = cfg.data.data_path
    evaluate_model=cfg.training.evaluate_model
    num_transformer_blocks = cfg.model.num_transformer_blocks
    dropout = cfg.model.dropout
    learning_rate = cfg.model.learning_rate
    scaler = default_config.scaler

    
    # Load data
    df = pd.read_csv(data_path)
    
    # Start MLflow run for this specific configuration
    run_name = f"HYDRA_transformer_{CODE}_lb{lookback}_fh{forecast}_{datetime.now().strftime('%H%M%S')}"
    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n🚀 Starting MLflow run: {run_name}")
        print(f"   • Run ID: {run.info.run_id}")
        
        # Log all hyperparameters from config
        mlflow.log_params({
            "target_code": CODE,
            "lookback": lookback,
            "forecast_horizon": forecast,
            "model_type": "transformer",
            "dataset": data_path,
            "activation_function": activation_function,
            "covid_token": covid_token,
            "cutoff_date": cutoff_date,
            "head_size": head_size,
            "num_heads": num_heads, 
            "ff_dim": ff_dim,
            "mlp_units": mlp_units,
            "num_transformer_blocks": num_transformer_blocks,
            "dropout": dropout,
            "learning_rate": learning_rate ,
            "positional_encoding": positional_encoding,
            "causal_masking": False,
            "scaler": scaler,
        })
        
        # Log system information
        mlflow.log_params({
            "tensorflow_version": tf.__version__,
            "python_version": os.sys.version.split()[0],
            "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
            "backend": cfg.system.backend,
            "hydra_config_name": cfg._target_ if hasattr(cfg, '_target_') else "config"
        })

        # ==================== PHASE 1: UNIVARIATE TRANSFORMER ====================
        univ_start_time = datetime.now()
        mlflow.log_param("phase_1_start_time", univ_start_time.isoformat())
        batch_size = data_preparation.compute_dynamic_batch_size(lookback, forecast)
        # Train univariate transformer with parameters from config
        univariate_parameters = TransformerUnivConfig(  
            lookback=lookback,
            forecast=forecast,
            code=CODE,
            activation_function=activation_function,
            covid_token=covid_token,   
            cutoff_date=cutoff_date,
            head_size=head_size, 
            num_heads=num_heads, 
            ff_dim=ff_dim, 
            mlp_units=mlp_units,
            evaluate_model=evaluate_model,
            positional_encoding=positional_encoding,
            num_transformer_blocks=num_transformer_blocks,
            dropout=dropout,
            learning_rate=learning_rate,
            data_path=data_path,
            batch_size = batch_size,
            
        )

        pipeline = UnivariateTransformerPipeline(univariate_parameters)
        #model, model_name, loss, mae, mse = main_training_univ_transformer.main_univ_transformer(**univariate_parameters)
        model, model_name, loss, mae, mse = pipeline.run_complete_pipeline()
        
        #Clear the GPU memory and possible memory garbage
        K.clear_session()
        gc.collect()

        mlflow.keras.log_model(model, artifact_path="univariate_model")
        
        load_mlflow_model_history(model_name, model_type="univariate_transformer")

        univ_end_time = datetime.now()
        univ_duration = (univ_end_time - univ_start_time).total_seconds()
        
        mlflow.log_metrics({
            "duration/phase_1_duration_seconds": univ_duration,
            "duration/phase_1_duration_minutes": univ_duration / 60,
        })

        if loss is not None and mae is not None and mse is not None:
            mlflow.log_metrics({
                "eval/univ_transformer_loss": loss,
                "eval/univ_transformer_mae": mae,
                "eval/univ_transformer_mse": mse
            })
        
        print(f"\n\nRunning for Lookback: {lookback}, Forecast: {forecast}, Code: {CODE}\n")
        
        # ==================== PHASE 2: RESIDUAL DIAGNOSTICS TRANSFORMER ====================
        diag_start_time = datetime.now()
        mlflow.log_param("phase_2_start_time", diag_start_time.isoformat())
        
        diagnostic_parameters = DiagnosticResidualTransformerConfig(
            lookback=lookback,
            forecast=forecast,
            code=CODE,
            activation_function=activation_function,
            covid_token=covid_token,
            cutoff_date=cutoff_date,
            predictions_train_corrected=None,
            predictions_test_corrected=None,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            mlp_units=mlp_units,
            dropout=dropout,
            learning_rate=learning_rate,
            data_path=data_path,
            batch_size = batch_size,
            num_transformer_blocks=num_transformer_blocks,
            
        )

        #predictions_train_corrected, predictions_test_corrected, residual_diagnostics_model, residual_diagnostics_model_name, corrected_diagnostics_mae, corrected_diagnostics_mse, corrected_diagnostics_rmse = main_train_diagnostic_residual_transformer.main_train_diagnostic_residual_transformer(**diagnostic_parameters)
        pipeline = DiagnosticResidualTransformerPipeline(diagnostic_parameters)
        predictions_train_corrected, predictions_test_corrected, residual_diagnostics_model, residual_diagnostics_model_name, corrected_diagnostics_mae, corrected_diagnostics_mse, corrected_diagnostics_rmse = pipeline.run_complete_pipeline()                                                                                                                                                                                                                                                                                                                   
        
        #Clear the GPU memory and possible memory garbage
        K.clear_session()
        gc.collect()

        mlflow.keras.log_model(residual_diagnostics_model, artifact_path="residual_diagnostics_model")
        load_mlflow_model_history(residual_diagnostics_model_name, model_type="residual_diagnostics_transformer")
        
        diag_end_time = datetime.now()
        diag_duration = (diag_end_time - diag_start_time).total_seconds()
        
        mlflow.log_metrics({
            "duration/phase_2_duration_seconds": diag_duration,
            "duration/phase_2_duration_minutes": diag_duration / 60,
            "eval/residual_diagnostics_model_mae": corrected_diagnostics_mae,
            "eval/residual_diagnostics_model_mse": corrected_diagnostics_mse,    
            "eval/residual_diagnostics_model_rmse": corrected_diagnostics_rmse,
        })
        
        # ==================== PHASE 3: RESIDUAL SEASONAL TRANSFORMER ====================
        seasonal_start_time = datetime.now()
        mlflow.log_param("phase_3_start_time", seasonal_start_time.isoformat())

        seasonal_params = SeasonalResidualTransformerConfig(
            lookback=lookback,
            forecast=forecast,
            code=CODE,
            activation_function=activation_function,
            covid_token=covid_token,
            cutoff_date=cutoff_date,
            predictions_train_corrected=predictions_train_corrected,
            predictions_test_corrected=predictions_test_corrected,
            head_size=head_size,
            num_heads=num_heads,
            ff_dim=ff_dim,
            num_transformer_blocks=num_transformer_blocks,
            mlp_units = mlp_units,
            dropout=dropout, #Only applyied to the transformer
            learning_rate=learning_rate,#Not applyied yet
            data_path=data_path,
            batch_size = batch_size,
            
        )

        '''predictions_train_corrected, predictions_test_corrected, residual_seasonal_model, residual_seasonal_model_name, corrected_seasonal_mae, corrected_seasonal_mse, corrected_seasonal_rmse = (
            main_train_seasonal_residual_transformer.main_train_seasonal_residual_transformer(**seasonal_params)
        )'''
        pipeline = SeasonalResidualTransformerPipeline(seasonal_params)
        predictions_train_corrected, predictions_test_corrected, residual_seasonal_model, residual_seasonal_model_name, corrected_seasonal_mae, corrected_seasonal_mse, corrected_seasonal_rmse = pipeline.run_complete_pipeline()
        
        #Clear the GPU memory and possible memory garbage
        K.clear_session()
        gc.collect()

        mlflow.keras.log_model(residual_seasonal_model, artifact_path="residual_seasonal_model")
        load_mlflow_model_history(residual_seasonal_model_name, model_type="residual_seasonal_transformer")
        
        seasonal_end_time = datetime.now()
        seasonal_duration = (seasonal_end_time - seasonal_start_time).total_seconds()
        total_duration = (seasonal_end_time - univ_start_time).total_seconds()
        
        mlflow.log_metrics({
            "duration/phase_3_duration_seconds": seasonal_duration,
            "duration/phase_3_duration_minutes": seasonal_duration / 60,
            "total_training_duration_seconds": total_duration,
            "total_training_duration_minutes": total_duration / 60,
            "eval/residual_seasonal_model_mae": corrected_seasonal_mae,
            "eval/residual_seasonal_model_mse": corrected_seasonal_mse,
            "eval/residual_seasonal_model_rmse": corrected_seasonal_rmse
        })
        
        # ==================== COLLECT AND SAVE BEST RESULTS ====================
        current_run_data = {
            "run_info": {
                "run_id": run.info.run_id,
                "run_name": run_name,
                "timestamp": datetime.now().isoformat(),
                "hydra_job_num": hydra.core.hydra_config.HydraConfig.get().job.num if hydra.core.hydra_config.HydraConfig.initialized() else None
            },
            "hyperparameters": {
                "target_code": CODE,
                "lookback": lookback,
                "forecast_horizon": forecast,
                "activation_function": activation_function,
                "covid_token": covid_token,
                "cutoff_date": cutoff_date,
                "model_type": "transformer",
                "head_size": head_size,
                "num_heads": num_heads,
                "ff_dim": ff_dim,
                "mlp_units": mlp_units,
                "num_transformer_blocks": num_transformer_blocks,
                "dropout": dropout,
                "learning_rate": learning_rate,
                
            },
            "system_info": {
                "tensorflow_version": tf.__version__,
                "python_version": os.sys.version.split()[0],
                "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                "backend": cfg.system.backend
            },
            "training_duration": {
                "phase_1_univariate_seconds": univ_duration,
                "phase_2_diagnostic_seconds": diag_duration,
                "phase_3_seasonal_seconds": seasonal_duration,
                "total_training_seconds": total_duration,
                "total_training_minutes": total_duration / 60
            },
            "evaluation_metrics": {
                "univariate_transformer": {
                    "loss": loss,
                    "mae": mae,
                    "mse": mse
                },
                "diagnostic_residual": {
                    "mae": corrected_diagnostics_mae,
                    "mse": corrected_diagnostics_mse,
                    "rmse": corrected_diagnostics_rmse
                },
                "seasonal_residual": {
                    "mae": corrected_seasonal_mae,
                    "mse": corrected_seasonal_mse,
                    "rmse": corrected_seasonal_rmse
                }
            },
            "model_paths": {
                "univariate_model": model_name,
                "diagnostic_model": residual_diagnostics_model_name,
                "seasonal_model": residual_seasonal_model_name
            }
        }
        
        # Use the final seasonal MSE as the comparison metric
        current_mse = corrected_seasonal_mse
        
        # Initialize or update best results for this code
        if CODE not in best_results_per_code:
            best_results_per_code[CODE] = {
                "best_mse": float('inf'),
                "best_config": None,
                "best_metrics": None,
                "best_run_info": None
            }
        
        # Check if this is the best result for this code
        if current_mse < best_results_per_code[CODE]["best_mse"]:
            print(f"🏆 NEW BEST RESULT for {CODE}!")
            print(f"   • Previous best MSE: {best_results_per_code[CODE]['best_mse']:.6f}")
            print(f"   • New best MSE: {current_mse:.6f}")
            
            # Update best results
            best_results_per_code[CODE] = {
                "best_mse": current_mse,
                "best_config": current_run_data["hyperparameters"],
                "best_metrics": current_run_data["evaluation_metrics"],
                "best_run_info": current_run_data
            }
            
            # Save to JSON file for this code
            json_filename = os.path.join(results_dir, f"best_hyperparameters_{CODE.replace('/', '_')}.json")
            with open(json_filename, 'w') as f:
                json.dump(current_run_data, f, indent=4, default=str)
            
            print(f"💾 Best results saved to: {json_filename}")
            
            # Log as MLflow tag for easy identification
            mlflow.set_tag("is_best_for_code", True)
            mlflow.set_tag("best_mse_for_code", current_mse)
        else:
            print(f"📊 Current MSE: {current_mse:.6f} (Best: {best_results_per_code[CODE]['best_mse']:.6f})")
            mlflow.set_tag("is_best_for_code", False)
        
        # Log final metrics for this run
        mlflow.log_metrics({
            "final/seasonal_mse": current_mse,
            "final/seasonal_mae": corrected_seasonal_mae,
            "final/seasonal_rmse": corrected_seasonal_rmse,
        })
        
        print(f"✅ Completed run for {CODE} - lb:{lookback} fh:{forecast}")
        print(f"   • Seasonal MSE: {current_mse:.6f}")
        print(f"   • Configuration: head_size={head_size}, num_heads={num_heads}, ff_dim={ff_dim}, mlp_units={mlp_units}")

if __name__ == "__main__":
    main_experiment()
    K.clear_session()
    gc.collect()