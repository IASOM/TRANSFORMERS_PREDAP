
import sys

from omegaconf import DictConfig, OmegaConf
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import hydra

from production.model_reconstruction_pipeline import ModelPredictionPipeline
from src.config.base_transformer_config import BaseTransformerConfig
from src.utils.experiments_utils import get_codes_list, compute_dynamic_batch_size, check_for_help_flag, get_dates_list, load_inference_codes_list
from src.utils.environment_utils import setup_gpu_memory
from src.config.config_manager import get_config
from AQUAS_DATA_RETRIEVAL import run_pipeline_optimized


config = get_config()
# Register custom resolver for loading JSON codes list
OmegaConf.register_new_resolver("get_codes_list", get_codes_list)
OmegaConf.register_new_resolver("get_dates_list", get_dates_list)  # Register a resolver for getting dates list
OmegaConf.register_new_resolver("load_inference_codes_list", load_inference_codes_list)  # Register a resolver for computing dynamic batch size
config_name = "config_inference.yaml"  

@hydra.main(version_base=None, config_path="conf", config_name=config_name)
def inference(cfg: DictConfig) -> None:
    check_for_help_flag(cfg)
    setup_gpu_memory()
    # =================== PHASE 0: LOAD CONFIGURATION ===================
    code = cfg.model.target_code
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
    evaluate_model = False  #cfg.training.evaluate_model
    num_transformer_blocks = cfg.model.num_transformer_blocks
    dropout = cfg.model.dropout
    learning_rate = cfg.model.learning_rate
    scaler = config.scaler
    batch_size = compute_dynamic_batch_size(config.lookback, config.forecast)
    model_folder = cfg.data.models_folder
    output_path = cfg.output.output_path
    metrics_df_path = cfg.output.metrics_df_path

    dates_list = get_dates_list("2025-06-30", "2025-12-30")  # Get list of dates from 2025-06-30 to 2025-12-31

    for max_date in dates_list:
        print(f"Running inference for date: {max_date}")

        sys.argv = [
            "run_pipeline.py", 
            "--all", 
            "--start-date", cutoff_date, 
            "--end-date", max_date,
        ]
        
        # Call the main function
        exit_code = run_pipeline_optimized.main()
        
        print(f"Pipeline finished with exit code: {exit_code}")
        print(f'executing inference for date {max_date} code: {code} with lookback: {lookback} and forecast: {forecast}')
        # =================== PHASE 0.1: DEFINE PARAMETERS ===================

        final_output_predictions = None
        final_output_df = pd.DataFrame()
        base_pipeline = ModelPredictionPipeline(config=BaseTransformerConfig(
                code=code,
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformer_blocks=num_transformer_blocks,
                mlp_units=mlp_units,
                activation_function=activation_function,
                dropout=dropout,
                learning_rate=learning_rate,
                epochs=50,
                batch_size=batch_size,
                cutoff_date=cutoff_date,
                covid_token=covid_token,
                positional_encoding=positional_encoding,
                evaluate_model=evaluate_model, 
                data_path=data_path, 
                model_folder=model_folder,
            )) 

        dates_list = [max_date] #Temporary solution to run the inference for the dates between 2025-6-30 and max_date, as the get_dates_list function is not working properly.
        final_output_df = base_pipeline.run_reconstruct_save_results_pipeline( data_path, 
                                                                            code, 
                                                                            [lookback], 
                                                                            [forecast], 
                                                                            final_output_predictions, 
                                                                            final_output_df,
                                                                            dates=dates_list
                                                                            )
        base_pipeline.save_final_output_predictions(final_output_df, output_path  = f"../production_predictions/final_output_predictions/{max_date}")

    base_pipeline.delete_old_data(predictions_dataset_path=output_path, real_data_dataset_path=data_path, metrics_df_path=metrics_df_path, max_date=max_date)
    print(f"\nFinal output predictions for code {code}:\n")


if __name__ == "__main__":


    inference()

    #print(final_output_df)