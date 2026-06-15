


import sys

from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import hydra

from production.model_reconstruction_pipeline import ModelPredictionPipeline
from src.config.base_transformer_config import BaseTransformerConfig
from src.utils.experiments_utils import get_codes_list, compute_dynamic_batch_size, check_for_help_flag
from src.utils.environment_utils import setup_gpu_memory
from src.config.config_manager import get_config
from AQUAS_DATA_RETRIEVAL import run_pipeline_optimized


config = get_config()
# Register custom resolver for loading JSON codes list
OmegaConf.register_new_resolver("get_codes_list", get_codes_list)
config_name = "config_inference.yaml"  

@hydra.main(version_base=None, config_path="conf", config_name=config_name)
def inference(cfg: DictConfig) -> None:
    check_for_help_flag(cfg)
    setup_gpu_memory()
    # =================== PHASE 0: LOAD CONFIGURATION ===================
    config.code = cfg.model.target_code
    config.lookback = cfg.model.lookback
    config.forecast = cfg.model.forecast
    config.head_size = cfg.model.head_size
    config.num_heads = cfg.model.num_heads
    config.ff_dim = cfg.model.ff_dim
    config.mlp_units = cfg.model.mlp_units
    config.activation_function = cfg.model.activation
    config.covid_token = cfg.model.covid_token
    config.cutoff_date = cfg.training.cutoff_date
    config.positional_encoding = cfg.training.positional_encoding
    config.data_path = cfg.data.data_path
    config.evaluate_model = cfg.training.evaluate_model
    config.num_transformer_blocks = cfg.model.num_transformer_blocks
    config.dropout = cfg.model.dropout
    config.learning_rate = cfg.model.learning_rate
    config.scaler = config.scaler
    config.batch_size = compute_dynamic_batch_size(config.lookback, config.forecast)
    config.model_folder = cfg.data.models_folder
    config.output_path = cfg.output.output_path
    config.metrics_df_path = cfg.output.metrics_df_path

    sys.argv = [
        "run_pipeline.py", 
        "--all", 
        "--start-date", config.cutoff_date, 
        "--end-date", cfg.inference.max_date,
    ]
    
    # Call the main function
    exit_code = run_pipeline_optimized.main()
    
    print(f"Pipeline finished with exit code: {exit_code}")
    # =================== PHASE 0.1: DEFINE PARAMETERS ===================

    final_output_predictions = None
    final_output_df = pd.DataFrame()
    base_pipeline = ModelPredictionPipeline(config=BaseTransformerConfig(
            code=config.code,
            head_size=config.head_size,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            num_transformer_blocks=config.num_transformer_blocks,
            mlp_units=config.mlp_units,
            activation_function=config.activation_function,
            dropout=config.dropout,
            learning_rate=config.learning_rate,
            epochs=50,
            batch_size=config.batch_size,
            cutoff_date=config.cutoff_date,
            covid_token=config.covid_token,
            positional_encoding=config.positional_encoding,
            evaluate_model=config.evaluate_model, 
            data_path=config.data_path, 
            model_folder=config.model_folder,
        )) 

    final_output_df = base_pipeline.run_reconstruct_save_results_pipeline( config.data_path, 
                                                                          config.code, 
                                                                          [config.lookback], 
                                                                          [config.forecast], 
                                                                          final_output_predictions, 
                                                                          final_output_df
                                                                          )
    base_pipeline.save_final_output_predictions(final_output_df)

    base_pipeline.delete_old_data(predictions_dataset_path=config.output_path, real_data_dataset_path=config.data_path, metrics_df_path=config.metrics_df_path)
    print(f"\nFinal output predictions for code {config.code}:\n")


if __name__ == "__main__":


    inference()

    #print(final_output_df)