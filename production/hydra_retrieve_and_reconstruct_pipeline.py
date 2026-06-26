import sys
import os
import importlib
import importlib.util
from pathlib import Path




# Ensure repository root and src are on sys.path so local packages resolve
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from training.training_utils import load_diagnostic_covariates
from utils.environment_utils import setup_gpu_memory
from src.config.base_transformer_config import BaseTransformerConfig
from production.model_reconstruction_pipeline import ModelPredictionPipeline
from src.utils.experiments_utils import get_codes_list, get_dates_list, load_inference_codes_as_list, load_inference_codes_list, get_codes_as_list, memory_cleanup
from src.config.config_manager import get_config

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

OmegaConf.register_new_resolver("load_json_codes_list", load_json_codes_list)
OmegaConf.register_new_resolver("get_codes_list", get_codes_list)
OmegaConf.register_new_resolver("get_dates_list", get_dates_list)
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


    input_data_retrieval_dir = cfg.data.input_data_retrieval_directory
    out_data_retrieval_dir = cfg.data.output_data_retrieval_directory
    input_directory = config.data_path
    #max_date = "2025-12-31"  # Set a default max date
    base_pipeline = ModelPredictionPipeline(config=BaseTransformerConfig(
                                            code=config.code,
                                            head_size=config.head_size,
                                            num_heads=config.num_heads,
                                            ff_dim=config.ff_dim,
                                            num_transformer_blocks=config.num_transformer_blocks,
                                            mlp_units=config.mlp_units,
                                            activation_function=config.activation_function,
                                            dropout=0,
                                            learning_rate=0.001,
                                            epochs=50,
                                            batch_size=32,
                                            cutoff_date=config.cutoff_date,
                                            covid_token=config.covid_token,
                                            positional_encoding=config.positional_encoding,
                                            evaluate_model=config.evaluate_model,
                                            data_path=config.data_path,
                                            model_folder=config.model_folder,
                                        ))
    simulation_dates = pd.date_range(start='2025-6-30', end='2026-01-31', freq='D')

    (X_univ_data, 
     X_univ_data_production, 
     Y_univ_data,
     Y_univ_data_production, 
     X_diagnostics_data, 
     X_diagnostics_data_production, 
     X_seasonal_data, 
     X_seasonal_data_production, 
     df_timestamp_production,
     df_timestamp
     ) = base_pipeline.create_inference_data(
                                            input_directory=config.data_path,
                                            code=config.code,
                                            lookback=config.lookback,
                                            forecast=config.forecast,
                                            max_date=simulation_dates[-1].strftime('%Y-%m-%d')
                                        )
    univ_model, diagnostics_model, seasonal_model = base_pipeline.create_inference_models(config.code, 
                                          config.lookback, 
                                          config.forecast, 
                                          X_univ_data_production, 
                                          X_diagnostics_data_production, 
                                          X_seasonal_data_production
                                          )

    for max_date in simulation_dates:
        (X_univ_data, 
        X_univ_data_production, 
        Y_univ_data,
        Y_univ_data_production, 
        X_diagnostics_data, 
        X_diagnostics_data_production, 
        X_seasonal_data, 
        X_seasonal_data_production, 
        df_timestamp_production,
        df_timestamp
        ) = base_pipeline.create_inference_data(
                                                input_directory=config.data_path,
                                                code=config.code,
                                                lookback=config.lookback,
                                                forecast=config.forecast,
                                                max_date=str(max_date)
                                            )
        
        final_output_predictions = None
        # rEMOVE THE first "DEMAND_" charachers in front of every code to match the format expected by the reconstruction pipeline
        #CODES_LIST = [code[7:] for code in CODES_LIST ]
        final_output_df = pd.DataFrame()


        '''
        final_output_df = base_pipeline.run_reconstruct_save_results_pipeline(input_directory, 
                                                                              config.code, 
                                                                              [config.lookback], 
                                                                              [config.forecast], 
                                                                              final_output_predictions, 
                                                                              final_output_df,
                                                                              [str(max_date)]
                                                                              )'''
        
        final_output_df = base_pipeline.run_reconstruct_save_results_pipeline(
            config = config,
            final_output_df = final_output_df,
            dates = [str(max_date)],
            univ_model = univ_model,
            diagnostics_model = diagnostics_model,
            seasonal_model = seasonal_model,
            X_univ_data = X_univ_data,
            Y_univ_data = Y_univ_data,
            df_timestamp = df_timestamp,
            X_univ_data_production = X_univ_data_production,
            X_diagnostics_data = X_diagnostics_data,
            X_diagnostics_data_production = X_diagnostics_data_production,
            X_seasonal_data = X_seasonal_data, X_seasonal_data_production = X_seasonal_data_production,
            df_timestamp_production = df_timestamp_production
            ) 
        

        base_pipeline.save_final_output_predictions_multirun(final_output_df, output_path="../hydra_production_predictions/final_output_predictions")
    #base_pipeline.delete_old_data(predictions_dataset_path=config.output_path, real_data_dataset_path=input_directory, metrics_df_path=config.metrics_df_path)
    del base_pipeline  # Elimina la instancia de base_pipeline para liberar memoria
    memory_cleanup()
    

if __name__ == "__main__":
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