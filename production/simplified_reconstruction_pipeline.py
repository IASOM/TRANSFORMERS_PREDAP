import sys
import os
import importlib
import importlib.util
from pathlib import Path

from data_utils.features import prepare_time_series_features
from data_utils.residual_data_preparation import generate_rolling_sequences_covariates, prepare_residual_data




# Ensure repository root and src are on sys.path so local packages resolve
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from data_utils import data_preparation
from data_utils import data_preparation
from production.data_preparation_in_poduction import prepare_production_prediction_diagnostics_data, prepare_production_prediction_seasonal_data, prepare_production_prediction_univ_data
from production.data_preparation_in_poduction import prepare_production_prediction_univ_data
from training.training_utils import load_diagnostic_covariates
from utils.environment_utils import setup_gpu_memory
from src.config.base_transformer_config import BaseTransformerConfig
from production.model_reconstruction_pipeline import ModelPredictionPipeline, compute_predap_auxiliary_metrics
from src.utils.experiments_utils import get_codes_list, get_dates_list,load_inference_codes_not_done, load_inference_codes_as_list, load_inference_codes_list, get_codes_as_list, memory_cleanup, smart_read
from src.config.config_manager import get_config
import tensorflow as tf

# We do not import the wrapper or optimized runner here because those modules
# perform package-level imports (e.g. `from config.config import ...`) that
# can fail depending on how this script is executed. Only import the specific
# helper we need below (`create_multiyear_sample`).

# Import script helper (create_multiyear_sample)
try:
    cms_mod = importlib.import_module("AQUAS_DATA_RETRIEVAL.scripts.create_multiyear_sample")
    create_multiyear_sample = cms_mod
except Exception:
    # As a fallback import by path
    cms_path = REPO_ROOT / "AQUAS_DATA_RETRIEVAL" / "scripts" / "create_multiyear_sample.py"
    spec = importlib.util.spec_from_file_location("create_multiyear_sample", cms_path)
    cms_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cms_mod)
    create_multiyear_sample = cms_mod

import hydra
from omegaconf import DictConfig, OmegaConf
from src.utils.experiments_utils import load_json_codes_list
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq



def save_final_output_predictions_multirun(
        final_output_df: pd.DataFrame,
        output_path: str = "../production_predictions/final_output_predictions"
        ):
        """
        Saves predictions as individual parquet files partitioned by code.
        Each (code, date, forecast) gets its own uniquely-named file so that:
        - Parallel Hydra instances writing different codes never collide.
        - Sequential date iterations within one instance accumulate safely.
        - A repeated (code, date) write raises an error instead of silently overwriting.

        Args:
            final_output_df: DataFrame with columns including 'code', 'target_date', 'forecast', etc.
            output_path: Root directory for the dataset.
        """
        for code, group_df in final_output_df.groupby("code"):
            # One subdirectory per code — mirrors what PyArrow partitioning would do,
            # but we control the filenames ourselves so we never delete existing files.
            code_dir = os.path.join(output_path, f"{code}")
            os.makedirs(code_dir, exist_ok=True)

            for (init_date, forecast), subgroup_df in group_df.groupby(["init_forecast_date", "forecast"]):
                # Unique, deterministic filename for this (code, init_date, forecast) triple.
                # Using init_forecast_date (the date the prediction was made from) + forecast horizon
                # uniquely identifies a prediction window.
                date_str = pd.Timestamp(init_date).strftime("%Y%m%d")
                filename = f"pred_code{code}_initdate{date_str}_fc{forecast}.parquet"
                filepath = os.path.join(code_dir, filename)

                if os.path.exists(filepath):
                    raise FileExistsError(
                        f"Collision detected: predictions for code='{code}', "
                        f"init_date='{init_date}', forecast={forecast} already exist at:\n"
                        f"  {filepath}\n"
                        f"Delete the file manually if you intend to overwrite it."
                    )

                table = pa.Table.from_pandas(subgroup_df, preserve_index=False)
                pq.write_table(table, filepath)
                print(f"[SAVED] {filepath}")

def create_inference_models(self, code: str, lookback: int, forecast: int, X_univ_data: np.ndarray, X_diagnostics_data: np.ndarray, X_seasonal_data: np.ndarray) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        univ_input_shape = X_univ_data.shape[1:]
        diagnostics_input_shape = X_diagnostics_data.shape[1:]
        seasonal_input_shape = X_seasonal_data.shape
        seasonal_input_shape = tuple((seasonal_input_shape[1], seasonal_input_shape[2] + 1))  # Add 1 to the last dimension to account for the diagnostics predictions that will be concatenated as an additional feature
        

        univ_model, diagnostics_model, seasonal_model = self.reconstruct_full_model(
            code,
            lookback,
            forecast,
            models_directory=self.config.model_folder,
            univ_input_shape=univ_input_shape,
            diagnostics_input_shape=diagnostics_input_shape,
            seasonal_input_shape=seasonal_input_shape,
            head_size=self.config.head_size,
            num_heads=self.config.num_heads,
            ff_dim=self.config.ff_dim,
            num_transformer_blocks=self.config.num_transformer_blocks,
            mlp_units=self.config.mlp_units,
            activation_function=self.config.activation_function,
            dropout=self.config.dropout,
            mlp_dropout=0,
            n_pred=1,
            pos_encoding=True
        )
        return univ_model, diagnostics_model, seasonal_model

def merge_ds_files(input_base_dir: str = "../hydra_production_predictions/final_output_predictions", output_base_dir: str = "../hydra_production_predictions/merged_final_output_predictions"):
        """
        Merges all Parquet files in a dataset directory into a single Parquet file.

        Args:
            dataset_path (str): Path to the dataset directory containing Parquet files.
            output_file (str): Path to the output Parquet file.
        """
        input_base_dir = Path(input_base_dir)
        output_base_dir = Path(output_base_dir)

        # 2. Iterate through each <code> directory
        for code_dir in input_base_dir.iterdir():
            # Ensure we are only looking at directories (skipping hidden files like .DS_Store)
            if code_dir.is_dir():
                code_name = code_dir.name
                print(f"Merging fragments for code: {code_name}...")
                
                # Define and create the specific output directory for this code
                target_dir = output_base_dir / code_name
                target_dir.mkdir(parents=True, exist_ok=True)
                output_file = target_dir / "part-0.parquet"
                
                try:
                    # 3. Load the fragments *only* for this specific code folder
                    dataset = ds.dataset(str(code_dir), format="parquet")
                    scanner = dataset.scanner()
                    
                    # 4. Stream all batches into a single part-0.parquet file
                    with pq.ParquetWriter(output_file, schema=dataset.schema) as writer:
                        for batch in scanner.to_batches():
                            writer.write_batch(batch)
                            
                except Exception as e:
                    print(f"❌ Error processing {code_name}: {e}")

        print("\nAll codes have been successfully merged!")



OmegaConf.register_new_resolver("load_json_codes_list", load_json_codes_list)
OmegaConf.register_new_resolver("get_codes_list", get_codes_list)
OmegaConf.register_new_resolver("get_dates_list", get_dates_list)
OmegaConf
config_name = "parallel_inference_codes.yaml" 




def load_hydra_config():
    @hydra.main(version_base=None, config_path="../conf", config_name=config_name)
    def main(cfg: DictConfig):
        print("Configuration loaded successfully:")
        print(OmegaConf.to_yaml(cfg))
        return cfg
    
    return main()

config = get_config()
# Register custom resolver for loading JSON codes list
#OmegaConf.register_new_resolver("get_codes_list", get_codes_list)
#OmegaConf.register_new_resolver("load_json_codes_list", load_json_codes_list)
OmegaConf.register_new_resolver("load_inference_codes_list", load_inference_codes_list)  # Register a resolver for computing dynamic batch size
OmegaConf.register_new_resolver("load_inference_codes_not_done", load_inference_codes_not_done)  # Register a resolver for getting dates list
@hydra.main(version_base=None, config_path="../conf", config_name=config_name)
def main_inference_pipeline(cfg: DictConfig) -> None:
    setup_gpu_memory()
    config.code = cfg.model.target_code
    config.data_path = cfg.data.data_path
    config.model_folder = cfg.data.models_folder
    #config.diagnostic_covariates_path = load_diagnostic_covariates(config,config.code, config.forecast)
    config.head_size = cfg.model.head_size
    config.num_heads = cfg.model.num_heads
    config.ff_dim = cfg.model.ff_dim
    config.num_transformer_blocks = cfg.model.num_transformer_blocks
    config.mlp_units = cfg.model.mlp_units
    config.activation_function = cfg.model.activation
    config.dropout = cfg.model.dropout
    config.learning_rate = cfg.model.learning_rate
    config.cutoff_date = cfg.training.cutoff_date
    config.covid_token = cfg.model.covid_token
    config.positional_encoding = cfg.training.positional_encoding
    config.evaluate_model = cfg.training.evaluate_model
    config.output_path = cfg.output.output_path
    config.metrics_df_path = cfg.output.metrics_df_path
    config.lookback = cfg.model.lookback
    config.forecast = cfg.model.forecast
    simulation_dates = pd.date_range(start='2025-6-30', end='2026-01-31', freq='D')
    
    final_output_df = pd.DataFrame()  # Initialize an empty DataFrame to store final output predictions
    output_chunks = []
    for max_date in simulation_dates:
        auxiliary_output_df = pd.DataFrame()  # Temporary DataFrame for current iteration
        df = smart_read(config.data_path)
        diagnostics_covariates = load_diagnostic_covariates(config, config.code, config.forecast)
        
        data_parameters = {
            "code": config.code,
            "lookback": config.lookback,
            "forecast": config.forecast,
            "cutoff_date": config.cutoff_date,
            "max_date": str(max_date),
            "covid_token": config.covid_token,
            "scaler": config.scaler,
            "eliminate_covid_data": config.eliminate_covid_data,
            "covid_dates": config.covid_dates
        }



        # Univariate Data Preparation
        X_univ_data_production, Y_univ_data_production, df_timestamp_production = prepare_production_prediction_univ_data(
            df=df,
            **data_parameters
        )
        X_univ, Y_univ = data_preparation.prepare_univariate_data(
            df = df,
            **data_parameters  
        )

        



        # Diagnostics Data Preparation
        X_diagnostics_data_production, Y_diagnostics_data_production = prepare_production_prediction_diagnostics_data(
            df=df,
            **data_parameters
        )
        X_diagnostics, Y_diagnostics = data_preparation.prepare_multivariate_data(
            df = df,
            **data_parameters,
            relevant_feature_cols=diagnostics_covariates,
        )


        X_diagnostics, Y_diagnostics = data_preparation.prepare_multivariate_data(
            df = df,
            **data_parameters,
            relevant_feature_cols=diagnostics_covariates,
        )


        # Seasonal Data Preparation

        date_list = data_preparation.extract_dates(config.data_path, 
                                                    config.code, 
                                                    config.lookback,
                                                    config.forecast, 
                                                    train=False, 
                                                    cutoff_date=config.cutoff_date, 
                                                    max_date = config.max_date, 
                                                    eliminate_covid_data=config.eliminate_covid_data, 
                                                    covid_dates=config.covid_dates)
        
        '''Y_residuals_diagnostics = prepare_residual_data(
            diagnostics_pred,
            Y_univ,
        )'''
        df_seasonal = prepare_time_series_features(
            df = df,
            categorical_vars = config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
            cutoff_date = config.cutoff_date,
            max_date = config.max_date,
            scaler = config.scaler,
            eliminate_covid_data = config.eliminate_covid_data, 
            covid_dates=config.covid_dates,
        )




        #Create models and load weights for the current code, lookback, and forecast
        univ_model, diagnostics_model, seasonal_model = create_inference_models(config.code, 
                                        config.lookback, 
                                        config.forecast, 
                                        X_univ_data_production, 
                                        X_diagnostics_data_production, 
                                        X_seasonal_data_production
                                        )


        # Univariate Predictions
        univ_pred = univ_model.predict(X_univ)
        univ_production_pred = univ_model.predict(X_univ_data_production)
    
        #Diagnostics Predictions
        diagnostics_residuals = diagnostics_model.predict(X_diagnostics)
        diagnostics_pred = univ_pred + diagnostics_residuals

        diagnostics_production_residuals = diagnostics_model.predict(X_diagnostics_data_production)
        diagnostics_production_pred = univ_production_pred + diagnostics_production_residuals

        #Merge the diagnostics predictions with the seasonal covariates for the current max_date
        X_seasonal_data_production = prepare_production_prediction_seasonal_data(
            df=df,
            **data_parameters,
            predictions_train=diagnostics_production_pred,
            seasonal_categorical_vars=config.DEFAULT_SEASONAL_CATEGORICAL_VARS
        )
        
        X_seasonal = generate_rolling_sequences_covariates(
            df_processed = df_seasonal,
            lookback=config.lookback,
            forecast=config.forecast,
            predictions_train = diagnostics_pred,

        )
        #Seasonal Predictions
        seasonal_residuals = seasonal_model.predict(X_seasonal)
        seasonal_pred = diagnostics_pred + seasonal_residuals

        seasonal_production_residuals = seasonal_model.predict(X_seasonal_data_production)
        seasonal_production_pred = diagnostics_production_pred + seasonal_production_residuals


        #Compute Actual metrics with known Y_univ for the current max_date
        mae = np.mean(np.abs(seasonal_pred - Y_univ))
        mse = np.mean((seasonal_pred - Y_univ)**2)
        wape = np.sum(np.abs(seasonal_pred - Y_univ)) / np.sum(np.abs(Y_univ) + 1e-8)
        print(f"MAE for code {config.code} with lookback {config.lookback} and forecast {config.forecast}: {mae}")
        print(f"MSE for code {config.code} with lookback {config.lookback} and forecast {config.forecast}: {mse}")
        print(f"WAPE for code {config.code} with lookback {config.lookback} and forecast {config.forecast}: {wape*100:.2f}%")

        #compute auxiliary metrics for the current unknown data for the max_date
        ci_lower, ci_upper, velocity, acceleration = compute_predap_auxiliary_metrics(
            predictions=seasonal_production_residuals,
            true_past_data=diagnostics_production_pred,
            mae=mae,
            lookback=config.lookback,
            confidence_level=config.confidence_level
        )
        target_date = df_timestamp_production[-config.forecast:].values
        df_timestamp = df['timestamp']
        init_forecast_date = df_timestamp.iloc[-config.forecast-1]
        final_forecast_date = df_timestamp.iloc[-1]

        #Save all the data in an auxiliary output dataframe for the current max_date
        auxiliary_output_df["target_date"] = target_date
        auxiliary_output_df["init_forecast_date"] = init_forecast_date
        auxiliary_output_df["final_forecast_date"] = final_forecast_date
        auxiliary_output_df["code"] = config.code
        auxiliary_output_df["forecast"] = config.forecast
        auxiliary_output_df["predictions"] = seasonal_production_pred.flatten()[:config.forecast]
        auxiliary_output_df["ci_lower"] = ci_lower
        auxiliary_output_df["ci_upper"] = ci_upper
        auxiliary_output_df["velocity"] = velocity
        auxiliary_output_df["acceleration"] = acceleration

        output_chunks.append(auxiliary_output_df)
        final_output_df = pd.concat(output_chunks, ignore_index=True)
        save_final_output_predictions_multirun(final_output_df, output_path="../hydra_production_predictions/final_output_predictions")
    merge_ds_files(input_base_dir="../hydra_production_predictions/final_output_predictions", output_base_dir="../hydra_production_predictions/merged_final_output_predictions")
    #base_pipeline.delete_old_data(predictions_dataset_path=config.output_path, real_data_dataset_path=input_directory, metrics_df_path=config.metrics_df_path)
    memory_cleanup()
    

if __name__ == "__main__":
    
    load_inference_codes_not_done(models_dir="../quantized_models", inference_dir="../hydra_production_predictions/final_output_predictions")
    max_date = "2025-12-31"  # Set a default max date
    input_data_retrieval_directory = 'AQUAS_DATA_RETRIEVAL/data/sample/multilayer_input/'
    output_data_retrieval_directory = 'AQUAS_DATA_RETRIEVAL/data/sample/multilayer_output/'
    custom_args = [
            "--start", "2010-01-01",
            "--end", max_date,
            "--input-dir", input_data_retrieval_directory,
            "--output-dir", output_data_retrieval_directory
            ]

    #create_multiyear_sample.main(custom_args)

    # `create_multiyear_sample.main` expects no arguments and parses
    # `sys.argv` via argparse. Temporarily set `sys.argv` and call it.
    old_argv = sys.argv
    sys.argv = [old_argv[0]] + custom_args
    try:
        create_multiyear_sample.main()
    finally:
        sys.argv = old_argv
    main_inference_pipeline()