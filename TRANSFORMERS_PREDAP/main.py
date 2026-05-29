<<<<<<< HEAD
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.tensorflow
import mlflow.keras
from datetime import datetime
import os
import tempfile
import json
import pickle

from src import data_preparation
from src import main_train_diagnostic_residual_transformer
from src import main_train_seasonal_residual_transformer

from src import main_training_univ_transformer
#load and visualize data 

from src.univariate_transformer import default_config

# MAIN TRANSFORMER MODEL WITH MLFLOW TRACKING

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = f"TRANSFORMERS_PREDAP_{datetime.now().strftime('%Y%m%d')}"
mlflow.set_experiment(experiment_name)

print(f"🎯 MLflow tracking initialized")
print(f"   • Experiment: {experiment_name}")
print(f"   • Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   • View results at: http://localhost:5000")

# MAIN TRANSFORMER MODEL 

LOOKBACK_LIST = default_config.LOOKBACK_LIST
FORECAST_LIST = default_config.FORECAST_LIST
CODES_LIST = default_config.CODES_LIST
CUTOFF_DATE = default_config.DATE_CUTOFF
df = pd.read_csv(default_config.DATA_PATH)

# Track overall experiment metrics
total_runs = len(CODES_LIST) * len(LOOKBACK_LIST) * len(FORECAST_LIST)
print(f"📊 Starting training for {total_runs} model configurations")

run_counter = 0


for CODE in CODES_LIST:
    for lb in LOOKBACK_LIST:
        for fh in FORECAST_LIST:
            run_counter += 1
            
            # Start MLflow run for this specific configuration
            run_name = f"Transformer_{CODE}_lb{lb}_fh{fh}_{datetime.now().strftime('%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                print(f"\n🚀 [{run_counter}/{total_runs}] Starting MLflow run: {run_name}")
                print(f"   • Run ID: {run.info.run_id}")
                
                
                # Log hyperparameters
                mlflow.log_params({
                    "target_code": CODE,
                    "lookback": lb,
                    "forecast_horizon": fh,
                    "model_type": "transformer",
                    "run_number": run_counter,
                    "total_runs": total_runs
                })
                
                # Log system information
                mlflow.log_params({
                    "tensorflow_version": tf.__version__,
                    "python_version": os.sys.version.split()[0],
                    "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                    "backend": "Agg"  # matplotlib backend
                })
                lookback = lb
                forecast = fh
                code = CODE

                # ==================== PHASE 1: UNIVARIATE TRANSFORMER ====================
                univ_start_time = datetime.now()
                mlflow.log_param("phase_1_start_time", univ_start_time.isoformat())
                # Train univariate transformer and capture results
                model, model_name, loss, mae, mse = main_training_univ_transformer.main_univ_transformer(forecast=forecast,lookback=lookback, code=code,   cutoff_date=CUTOFF_DATE)
                mlflow.keras.log_model(model, artifact_path="univariate_model")
                
                history_path = f"{model_name}_history.pkl"

                if os.path.exists(history_path):
                    print(f" Found saved history at: {history_path}")
    
                    # Load the history
                    with open(history_path, "rb") as f:
                        history_data = pickle.load(f)
                    
                    # Convert to DataFrame for easier handling
                    history_df = pd.DataFrame(history_data)
                    history_df["epoch"] = range(1, len(history_df) + 1)

                    
                    # --- Log metrics ---
                    for epoch, row in history_df.iterrows():
                        for metric, value in row.items():
                            if metric != "epoch":
                                mlflow.log_metric(metric, float(value), step=int(row["epoch"]))
                    
                    # --- Create and log plots ---
                    metric_groups = {
                        "loss": ["loss", "val_loss"],
                        "accuracy": ["accuracy", "val_accuracy"],
                    }

                    for group_name, keys in metric_groups.items():
                        available = [k for k in keys if k in history_df.columns]
                        if not available:
                            continue

                        plt.figure(figsize=(8, 4))
                        for k in available:
                            plt.plot(history_df["epoch"], history_df[k], label=k, linewidth=2)
                        plt.xlabel("Epoch")
                        plt.ylabel(group_name.capitalize())
                        plt.title(f"Training vs Validation {group_name.capitalize()}")
                        plt.legend()
                        plt.grid(True, linestyle="--", alpha=0.6)
                        plt.tight_layout()

                        plot_path = f"{model_name}_{group_name}_curve.png"
                        plt.savefig(plot_path)
                        plt.close()

                        # Log as artifact
                        mlflow.log_artifact(plot_path, artifact_path="plots")

                        print("✅ History loaded and logged to MLflow successfully.")
                else:
                    print(f"⚠️ No history file found at {history_path}")
                
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
                print(f"\n\nRunning for Lookback: {lookback}, Forecast: {forecast}, Code: {code}\n")
                
                # RESIDUAL DIAGNOSTICS TRANSFORMER
                diag_start_time = datetime.now()
                mlflow.log_param("phase_2_start_time", diag_start_time.isoformat())
                predictions_train_corrected, predictions_test_corrected, residual_diagnostics_model, residual_diagnostics_model_name, corrected_diagnostics_mae, corrected_diagnostics_mse, corrected_diagnostics_rmse = main_train_diagnostic_residual_transformer.main_train_diagnostic_residual_transformer(lookback=lookback, forecast=forecast, code=code, cutoff_date=CUTOFF_DATE, predictions_train_corrected=None, predictions_test_corrected=None)
                mlflow.keras.log_model(residual_diagnostics_model, artifact_path="residual_diagnostics_model")
                
                diag_end_time = datetime.now()
                diag_duration = (diag_end_time - diag_start_time).total_seconds()
                
                mlflow.log_metrics({
                    "duration/phase_2_duration_seconds": diag_duration,
                    "duration/phase_2_duration_minutes": diag_duration / 60,
                    "eval/residual_diagnostics_model_mae": corrected_diagnostics_mae,
                    "eval/residual_diagnostics_model_mse": corrected_diagnostics_mse,    
                    "eval/residual_diagnostics_model_rmse": corrected_diagnostics_rmse,
                    })
                # RESIDUAL SEASONAL TRANSFORMER
                seasonal_start_time = datetime.now()
                mlflow.log_param("phase_3_start_time", seasonal_start_time.isoformat())

                predictions_train_corrected, predictions_test_corrected, residual_seasonal_model, residual_seasonal_model_name, corrected_seasonal_mae, corrected_seasonal_mse, corrected_seasonal_rmse = main_train_seasonal_residual_transformer.main_train_seasonal_residual_transformer(lookback=lookback, forecast=forecast, code=code, cutoff_date=CUTOFF_DATE, predictions_train_corrected=predictions_train_corrected, predictions_test_corrected=predictions_test_corrected)
                mlflow.keras.log_model(residual_seasonal_model, artifact_path="residual_seasonal_model")

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



=======
import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
import mlflow
import mlflow.tensorflow
import mlflow.keras
from datetime import datetime
import os
import tempfile
import json
import pickle

from src import data_preparation
from src import main_train_diagnostic_residual_transformer
from src import main_train_seasonal_residual_transformer

from src import main_training_univ_transformer
#load and visualize data 

from src.univariate_transformer import default_config

# MAIN TRANSFORMER MODEL WITH MLFLOW TRACKING

# Initialize MLflow
mlflow.set_tracking_uri("file:./mlruns")
experiment_name = f"TRANSFORMERS_PREDAP_{datetime.now().strftime('%Y%m%d')}"
mlflow.set_experiment(experiment_name)

print(f"🎯 MLflow tracking initialized")
print(f"   • Experiment: {experiment_name}")
print(f"   • Tracking URI: {mlflow.get_tracking_uri()}")
print(f"   • View results at: http://localhost:5000")

# MAIN TRANSFORMER MODEL 

LOOKBACK_LIST = default_config.LOOKBACK_LIST
FORECAST_LIST = default_config.FORECAST_LIST
CODES_LIST = default_config.CODES_LIST
CUTOFF_DATE = default_config.DATE_CUTOFF
df = pd.read_csv(default_config.DATA_PATH)

# Track overall experiment metrics
total_runs = len(CODES_LIST) * len(LOOKBACK_LIST) * len(FORECAST_LIST)
print(f"📊 Starting training for {total_runs} model configurations")

run_counter = 0


for CODE in CODES_LIST:
    for lb in LOOKBACK_LIST:
        for fh in FORECAST_LIST:
            run_counter += 1
            
            # Start MLflow run for this specific configuration
            run_name = f"Transformer_{CODE}_lb{lb}_fh{fh}_{datetime.now().strftime('%H%M%S')}"
            with mlflow.start_run(run_name=run_name) as run:
                print(f"\n🚀 [{run_counter}/{total_runs}] Starting MLflow run: {run_name}")
                print(f"   • Run ID: {run.info.run_id}")
                
                
                # Log hyperparameters
                mlflow.log_params({
                    "target_code": CODE,
                    "lookback": lb,
                    "forecast_horizon": fh,
                    "model_type": "transformer",
                    "run_number": run_counter,
                    "total_runs": total_runs
                })
                
                # Log system information
                mlflow.log_params({
                    "tensorflow_version": tf.__version__,
                    "python_version": os.sys.version.split()[0],
                    "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                    "backend": "Agg"  # matplotlib backend
                })
                lookback = lb
                forecast = fh
                code = CODE

                # ==================== PHASE 1: UNIVARIATE TRANSFORMER ====================
                univ_start_time = datetime.now()
                mlflow.log_param("phase_1_start_time", univ_start_time.isoformat())
                # Train univariate transformer and capture results
                model, model_name, loss, mae, mse = main_training_univ_transformer.main_univ_transformer(forecast=forecast,lookback=lookback, code=code,   cutoff_date=CUTOFF_DATE)
                mlflow.keras.log_model(model, artifact_path="univariate_model")
                
                history_path = f"{model_name}_history.pkl"

                if os.path.exists(history_path):
                    print(f" Found saved history at: {history_path}")
    
                    # Load the history
                    with open(history_path, "rb") as f:
                        history_data = pickle.load(f)
                    
                    # Convert to DataFrame for easier handling
                    history_df = pd.DataFrame(history_data)
                    history_df["epoch"] = range(1, len(history_df) + 1)

                    
                    # --- Log metrics ---
                    for epoch, row in history_df.iterrows():
                        for metric, value in row.items():
                            if metric != "epoch":
                                mlflow.log_metric(metric, float(value), step=int(row["epoch"]))
                    
                    # --- Create and log plots ---
                    metric_groups = {
                        "loss": ["loss", "val_loss"],
                        "accuracy": ["accuracy", "val_accuracy"],
                    }

                    for group_name, keys in metric_groups.items():
                        available = [k for k in keys if k in history_df.columns]
                        if not available:
                            continue

                        plt.figure(figsize=(8, 4))
                        for k in available:
                            plt.plot(history_df["epoch"], history_df[k], label=k, linewidth=2)
                        plt.xlabel("Epoch")
                        plt.ylabel(group_name.capitalize())
                        plt.title(f"Training vs Validation {group_name.capitalize()}")
                        plt.legend()
                        plt.grid(True, linestyle="--", alpha=0.6)
                        plt.tight_layout()

                        plot_path = f"{model_name}_{group_name}_curve.png"
                        plt.savefig(plot_path)
                        plt.close()

                        # Log as artifact
                        mlflow.log_artifact(plot_path, artifact_path="plots")

                        print("✅ History loaded and logged to MLflow successfully.")
                else:
                    print(f"⚠️ No history file found at {history_path}")
                
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
                print(f"\n\nRunning for Lookback: {lookback}, Forecast: {forecast}, Code: {code}\n")
                
                # RESIDUAL DIAGNOSTICS TRANSFORMER
                diag_start_time = datetime.now()
                mlflow.log_param("phase_2_start_time", diag_start_time.isoformat())
                predictions_train_corrected, predictions_test_corrected, residual_diagnostics_model, residual_diagnostics_model_name, corrected_diagnostics_mae, corrected_diagnostics_mse, corrected_diagnostics_rmse = main_train_diagnostic_residual_transformer.main_train_diagnostic_residual_transformer(lookback=lookback, forecast=forecast, code=code, cutoff_date=CUTOFF_DATE, predictions_train_corrected=None, predictions_test_corrected=None)
                mlflow.keras.log_model(residual_diagnostics_model, artifact_path="residual_diagnostics_model")
                
                diag_end_time = datetime.now()
                diag_duration = (diag_end_time - diag_start_time).total_seconds()
                
                mlflow.log_metrics({
                    "duration/phase_2_duration_seconds": diag_duration,
                    "duration/phase_2_duration_minutes": diag_duration / 60,
                    "eval/residual_diagnostics_model_mae": corrected_diagnostics_mae,
                    "eval/residual_diagnostics_model_mse": corrected_diagnostics_mse,    
                    "eval/residual_diagnostics_model_rmse": corrected_diagnostics_rmse,
                    })
                # RESIDUAL SEASONAL TRANSFORMER
                seasonal_start_time = datetime.now()
                mlflow.log_param("phase_3_start_time", seasonal_start_time.isoformat())

                predictions_train_corrected, predictions_test_corrected, residual_seasonal_model, residual_seasonal_model_name, corrected_seasonal_mae, corrected_seasonal_mse, corrected_seasonal_rmse = main_train_seasonal_residual_transformer.main_train_seasonal_residual_transformer(lookback=lookback, forecast=forecast, code=code, cutoff_date=CUTOFF_DATE, predictions_train_corrected=predictions_train_corrected, predictions_test_corrected=predictions_test_corrected)
                mlflow.keras.log_model(residual_seasonal_model, artifact_path="residual_seasonal_model")

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



>>>>>>> samper_cleaning
