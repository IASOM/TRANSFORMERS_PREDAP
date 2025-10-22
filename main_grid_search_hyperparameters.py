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

# Utility function for safe float conversion
def safe_float(value):
    """Convert value to float, handling numpy types and NaN values"""
    try:
        if pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None

# MAIN TRANSFORMER MODEL 

LOOKBACK_LIST = default_config.LOOKBACK_LIST
FORECAST_LIST = default_config.FORECAST_LIST
CODES_LIST = default_config.CODES_LIST
CUTOFF_DATE = default_config.DATE_CUTOFF
ACTIVATIONS_LIST = default_config.ACTIVATIONS_LIST
COVID_TOKEN_LIST = default_config.COVID_TOKEN_LIST

df = pd.read_csv(default_config.DATA_PATH)

# Track overall experiment metrics
total_runs = len(CODES_LIST) * len(LOOKBACK_LIST) * len(FORECAST_LIST)
print(f"📊 Starting training for {total_runs} model configurations")

run_counter = 0

# Dictionary to track best results for each code
best_results_per_code = {}

# Create results directory
results_dir = "best_hyperparameters_results"
os.makedirs(results_dir, exist_ok=True)
print(f"📁 Results will be saved in: {results_dir}/")

# Initialize best results tracking for each code
for code in CODES_LIST:
    best_results_per_code[code] = {
        "best_mse": float('inf'),
        "best_config": None,
        "best_metrics": None,
        "best_run_info": None
    }


for CODE in CODES_LIST:
    for ACTIVATION_FUNCTION in ACTIVATIONS_LIST:
        for COVID_TOKEN in COVID_TOKEN_LIST:
            for lb in LOOKBACK_LIST:
                for fh in FORECAST_LIST:
                    run_counter += 1
                    
                    # Start MLflow run for this specific configuration
                    run_name = f"Grid_search_transformer_{CODE}_lb{lb}_fh{fh}_{datetime.now().strftime('%H%M%S')}"
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
                        
                        # compact parameter grouping for readability
                        univariate_parameters = dict(  
                            lookback=lookback,
                            forecast=forecast,
                            code=code,
                            activation_function=ACTIVATION_FUNCTION,
                            covid_token= COVID_TOKEN,   
                            cutoff_date=CUTOFF_DATE
                        )

                        model, model_name, loss, mae, mse = main_training_univ_transformer.main_univ_transformer(**univariate_parameters)
                        
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
                        
                        # compact parameter grouping for readability
                        diagnostic_parameters = dict(
                            lookback=lookback,
                            forecast=forecast,
                            code=code,
                            activation_function=ACTIVATION_FUNCTION,
                            covid_token=COVID_TOKEN,
                            cutoff_date=CUTOFF_DATE,
                            predictions_train_corrected=None,
                            predictions_test_corrected=None 
                        )

                        predictions_train_corrected, predictions_test_corrected, residual_diagnostics_model, residual_diagnostics_model_name, corrected_diagnostics_mae, corrected_diagnostics_mse, corrected_diagnostics_rmse = main_train_diagnostic_residual_transformer.main_train_diagnostic_residual_transformer(**diagnostic_parameters)
                                                                                                                                                                                                                                                                                                                       
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

                        # compact parameter grouping for readability
                        seasonal_params = dict(
                            lookback=lookback,
                            forecast=forecast,
                            code=code,
                            activation_function=ACTIVATION_FUNCTION,
                            covid_token=COVID_TOKEN,
                            cutoff_date=CUTOFF_DATE,
                            predictions_train_corrected=predictions_train_corrected,
                            predictions_test_corrected=predictions_test_corrected,
                        )

                        predictions_train_corrected, predictions_test_corrected, residual_seasonal_model, residual_seasonal_model_name, corrected_seasonal_mae, corrected_seasonal_mse, corrected_seasonal_rmse = (
                            main_train_seasonal_residual_transformer.main_train_seasonal_residual_transformer(**seasonal_params)
                        )

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
                        
                        # ==================== COLLECT AND SAVE BEST RESULTS ====================
                        
                        # Prepare current run data
                        current_run_data = {
                            "run_info": {
                                "run_id": run.info.run_id,
                                "run_name": run_name,
                                "timestamp": datetime.now().isoformat(),
                                "run_counter": run_counter,
                                "total_runs": total_runs
                            },
                            "hyperparameters": {
                                "target_code": CODE,
                                "lookback": lb,
                                "forecast_horizon": fh,
                                "activation_function": ACTIVATION_FUNCTION,
                                "covid_token": COVID_TOKEN,
                                "cutoff_date": CUTOFF_DATE,
                                "model_type": "transformer"
                            },
                            "system_info": {
                                "tensorflow_version": tf.__version__,
                                "python_version": os.sys.version.split()[0],
                                "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
                                "backend": "Agg"
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
                        
                        # Use the final seasonal MSE as the comparison metric (best overall performance)
                        current_mse = corrected_seasonal_mse
                        
                        # Check if this is the best result for this code
                        if current_mse < best_results_per_code[CODE]["best_mse"]:
                            print(f"🏆 NEW BEST RESULT for {CODE}!")
                            print(f"   • Previous best MSE: {best_results_per_code[CODE]['best_mse']:.6f}")
                            print(f"   • New best MSE: {current_mse:.6f}")
                            
                            # Calculate improvement percentage (handle infinity case)
                            if best_results_per_code[CODE]["best_mse"] != float('inf') and best_results_per_code[CODE]["best_mse"] > 0:
                                improvement_pct = ((best_results_per_code[CODE]["best_mse"] - current_mse) / best_results_per_code[CODE]["best_mse"] * 100)
                                print(f"   • Improvement: {improvement_pct:.2f}%")
                            else:
                                print(f"   • First successful run for {CODE}!")
                            
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
                        
                        print(f"✅ [{run_counter}/{total_runs}] Completed run for {CODE} - lb:{lb} fh:{fh}")


# ==================== FINAL SUMMARY AND CONSOLIDATED RESULTS ====================

print(f"\n🎉 GRID SEARCH COMPLETED!")
print(f"📊 Total runs: {total_runs}")
print(f"📁 Results saved in: {results_dir}/")

# Create consolidated summary of all best results
summary_data = {
    "experiment_info": {
        "total_runs": total_runs,
        "total_codes": len(CODES_LIST),
        "completed_timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "mlflow_tracking_uri": mlflow.get_tracking_uri()
    },
    "hyperparameter_ranges": {
        "lookback_list": LOOKBACK_LIST,
        "forecast_list": FORECAST_LIST,
        "codes_list": CODES_LIST,
        "activations_list": ACTIVATIONS_LIST,
        "covid_token_list": COVID_TOKEN_LIST
    },
    "best_results_summary": {}
}

print(f"\n🏆 BEST RESULTS SUMMARY:")
print("=" * 80)

for code in CODES_LIST:
    best_result = best_results_per_code[code]
    if best_result["best_mse"] != float('inf'):
        config = best_result["best_config"]
        metrics = best_result["best_metrics"]["seasonal_residual"]
        
        print(f"\n📈 {code}:")
        print(f"   • Best MSE: {best_result['best_mse']:.6f}")
        print(f"   • Best Config: lb={config['lookback']}, fh={config['forecast_horizon']}")
        print(f"   • Activation: {config['activation_function']}, COVID Token: {config['covid_token']}")
        print(f"   • MAE: {metrics['mae']:.6f}, RMSE: {metrics['rmse']:.6f}")
        
        # Add to summary
        summary_data["best_results_summary"][code] = {
            "best_mse": best_result["best_mse"],
            "best_config": best_result["best_config"],
            "best_metrics": best_result["best_metrics"],
            "model_paths": best_result["best_run_info"]["model_paths"]
        }
    else:
        print(f"\n❌ {code}: No successful runs")
        summary_data["best_results_summary"][code] = {
            "status": "no_successful_runs",
            "best_mse": None
        }

# Save consolidated summary
summary_filename = os.path.join(results_dir, "consolidated_best_results_summary.json")
with open(summary_filename, 'w') as f:
    json.dump(summary_data, f, indent=4, default=str)

print(f"\n💾 Consolidated summary saved to: {summary_filename}")

# Create a CSV summary for easy analysis
csv_data = []
for code in CODES_LIST:
    best_result = best_results_per_code[code]
    if best_result["best_mse"] != float('inf'):
        config = best_result["best_config"]
        metrics = best_result["best_metrics"]
        
        csv_row = {
            "code": code,
            "best_mse": best_result["best_mse"],
            "lookback": config["lookback"],
            "forecast_horizon": config["forecast_horizon"],
            "activation_function": config["activation_function"],
            "covid_token": config["covid_token"],
            "univ_mse": metrics["univariate_transformer"]["mse"],
            "diag_mse": metrics["diagnostic_residual"]["mse"],
            "seasonal_mse": metrics["seasonal_residual"]["mse"],
            "seasonal_mae": metrics["seasonal_residual"]["mae"],
            "seasonal_rmse": metrics["seasonal_residual"]["rmse"],
            "total_training_minutes": best_result["best_run_info"]["training_duration"]["total_training_minutes"]
        }
        csv_data.append(csv_row)

# Save CSV summary
if csv_data:
    csv_df = pd.DataFrame(csv_data)
    csv_filename = os.path.join(results_dir, "best_results_summary.csv")
    csv_df.to_csv(csv_filename, index=False)
    print(f"📊 CSV summary saved to: {csv_filename}")
    
    # Display top performers
    print(f"\n🥇 TOP 3 PERFORMERS (by MSE):")
    top_performers = csv_df.nsmallest(3, 'best_mse')
    for idx, row in top_performers.iterrows():
        print(f"   {idx+1}. {row['code']}: MSE={row['best_mse']:.6f} (lb={row['lookback']}, fh={row['forecast_horizon']})")

print(f"\n✨ All results saved in directory: {results_dir}/")
print(f"🌐 View detailed MLflow results at: http://localhost:5000")
print("🔍 Individual best config files: best_hyperparameters_<CODE>.json")
print("📋 Consolidated summary: consolidated_best_results_summary.json")
print("📊 CSV summary: best_results_summary.csv")

