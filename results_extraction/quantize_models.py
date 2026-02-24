

import keras
import mlflow
from sklearn.pipeline import FunctionTransformer
import tensorflow as tf
import numpy as np
import os
import argparse

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.univariate_transformer.model_architecture_univ_transformer import (
    PositionalEncoding, RevIN
)

from save_forecast_trajectories import univariate_transformer_phase, diagnostics_transformer_phase, seasonal_transformer_phase, load_mlflow_model
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
import data_preparation
from config.base_transformer_config import BaseTransformerConfig

scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)


CUSTOM_OBJECTS = {
    'RevIN': RevIN,
    'PositionalEncoding': PositionalEncoding
}   

class ModelPredictionPipeline:
    def __init__(self, config: BaseTransformerConfig):
        self.config = config
        self.config.print_config()



    def manual_weight_quantization(self, model, model_name,  output_path="_f16.keras"):
        # 1. Create a deep copy so we don't modify the original model
        quant_model = keras.models.clone_model(model)
        quant_model.set_weights(model.get_weights())
        
        for layer in quant_model.layers:
            weights = layer.get_weights()
            if weights:
                # 2. Actually use float16 for compression
                quantized_weights = [w.astype(np.float16) for w in weights]

                layer.set_weights(quantized_weights)
        #save_weights_path =  model_name + "_f16_weights.h5"
        #quant_model.save_weights(save_weights_path)
                
        #quant_model.save(model_name + output_path)
        return quant_model


    def evaluate_model(self, model,model_name, past_preds,  X_test, Y_test, Y_test_orig, original_scale_df, code, forecast, lookback, cutoff_date, max_date, scaler, eliminate_covid_data=False, covid_dates=None):
        # Get predictions
        predictions = model.predict(X_test, verbose=0)
        if past_preds is not None:
            corrected_preds =  past_preds + predictions
        else:
            corrected_preds = predictions

        print("Predicted values shape:", predictions.shape)

        # Inverse transform predictions
        '''predictions_to_plot = data_preparation.inverse_transform_predictions(
            corrected_preds, original_scale_df, code=code, forecast=forecast, lookback=lookback, cutoff_date=cutoff_date, max_date = max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates
        )'''

        non_negative_predictions_to_plot = np.maximum(corrected_preds, 0)  # Ensure no negative predictions

        original_mae = mean_absolute_error(Y_test_orig, non_negative_predictions_to_plot)
        original_mse = mean_squared_error(Y_test_orig, non_negative_predictions_to_plot)
        original_rmse = np.sqrt(original_mse)
        original_wape = np.sum(np.abs(Y_test_orig - non_negative_predictions_to_plot)) / np.sum(np.abs(Y_test_orig)) * 100
        
        print(f"Test Results for {model_name} - MAE: {original_mae:.4f}, MSE: {original_mse:.4f}, RMSE: {original_rmse:.4f}, WAPE: {original_wape:.4f}%")

        return original_mae, original_mse, original_rmse, original_wape, corrected_preds


    def save_quantized_model_weights(self, quant_model, model_name, code, forecast, lookback):
        save_weights_path =  f"../quantized_models/{code}/{model_name}/{code}_{model_name}_{forecast}fh_{lookback}lb_f16_weights.h5"
        if not os.path.exists(os.path.dirname(save_weights_path)):
            os.makedirs(os.path.dirname(save_weights_path))
        quant_model.save_weights(save_weights_path)
        print(f"Quantized weights saved to {save_weights_path}")

    def load_mlflow_run_id_by_name(self, exp_names, code, forecast, lookback, model_type, lr = 1e-5):

        run_name_prefix = f"full_TRANSFORMER3_transformer_{code}_lb{lookback}_fh{forecast}"

        filter_string = f"attributes.run_name LIKE '{run_name_prefix}%'"
        
        runs = mlflow.search_runs(experiment_names = exp_names,
                                filter_string=filter_string, 
                                order_by=["start_time DESC"],
                                max_results=1)
        
        if runs.empty:
            raise ValueError(f"No run found starting with '{run_name_prefix}' in '{exp_names}'")
        
        return runs.iloc[0].run_id
    
    def eval_quantization_impact(self, input_directory, code, lookback, forecast,cutoff_date, 
                                 max_date,scaler, univ_model, diagnostics_model, seasonal_model, 
                                 quant_univ_model, quant_diagnostics_model, quant_seasonal_model):
        
        # --- 1. DATA PREPARATION PHASE ---
        # Load all required test features
        X_test, Y_test, Y_test_orig, date_list = univariate_transformer_phase(
            input_directory, code, lookback, forecast, cutoff_date, max_date, 
            relevant_feature_cols=None, scaler=scaler, eliminate_covid_data=eliminate_covid_data
            )
        
        X_train_covs, X_test_covs = diagnostics_transformer_phase(code, lookback, forecast, max_date, scaler)
        
        categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation"]
        original_scale_df = pd.read_parquet(input_directory)

        # --- 2. INFERENCE PHASE ---
        predictions_univ = univ_model.predict(X_test, verbose=0)
        pred_diagnostics_residuals = diagnostics_model.predict(X_test_covs, verbose=0)

        predictions_test = predictions_univ + pred_diagnostics_residuals 

        X_train_seasonal_covs, X_test_seasonal_covs = seasonal_transformer_phase(
            code, forecast, lookback,cutoff_date, max_date, categorical_vars=categorical_vars, 
            predictions_train=None, predictions_test=predictions_test, scaler=scaler)
        X_train_seasonal_covs = X_train_seasonal_covs.astype(float)
        X_test_seasonal_covs = X_test_seasonal_covs.astype(float)

        pred_seasonal_residuals = seasonal_model.predict(X_test_seasonal_covs, verbose=0)

        pred_corrected_diagnostics = predictions_univ + pred_diagnostics_residuals
        pred_corrected_seasonal = pred_corrected_diagnostics + pred_seasonal_residuals

        non_negative_predictions_to_plot = np.maximum(pred_corrected_seasonal, 0)  # Ensure no negative predictions

        original_mae = mean_absolute_error(Y_test_orig, non_negative_predictions_to_plot)
        original_mse = mean_squared_error(Y_test_orig, non_negative_predictions_to_plot)
        original_rmse = np.sqrt(original_mse)
        original_wape = np.sum(np.abs(Y_test_orig - non_negative_predictions_to_plot)) / np.sum(np.abs(Y_test_orig)) * 100

        # Evaluate model
        print(f"\nTest Results for Original Models")
        original_mae, original_mse, original_rmse, original_wape, preds_univ = self.evaluate_model(univ_model, "Univariate Transformer", None, X_test, Y_test, Y_test_orig, original_scale_df, code, forecast, lookback, cutoff_date, max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
        diagnostics_mae, diagnostics_mse, diagnostics_rmse, diagnostics_wape, preds_diagnostics = self.evaluate_model(diagnostics_model, "Diagnostics Residuals Transformer", preds_univ, X_test_covs, Y_test, Y_test_orig, original_scale_df, code, forecast, lookback, cutoff_date, max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
        seasonal_mae, seasonal_mse, seasonal_rmse, seasonal_wape, preds_seasonal = self.evaluate_model(seasonal_model, "Seasonal Residuals Transformer", preds_diagnostics , X_test_seasonal_covs, Y_test, Y_test_orig, original_scale_df, code, forecast, lookback,cutoff_date=cutoff_date , max_date=max_date , scaler=scaler , eliminate_covid_data=eliminate_covid_data , covid_dates=covid_dates)

        #Evaluate quantized model
        quant_predictions_univ = quant_univ_model.predict(X_test, verbose=0)
        quant_predictions_diagnostics_residuals = quant_diagnostics_model.predict(X_test_covs, verbose=0)
        quant_predictions_seasonal_residuals = quant_seasonal_model.predict(X_test_seasonal_covs, verbose=0)

        quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
        quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals
        print(f"\nTest Results for Quantized Models")
        original_quant_mae, original_quant_mse, original_quant_rmse, original_quant_wape, _ = self.evaluate_model(quant_univ_model, "Quantized Univariate Transformer", None, X_test, Y_test, Y_test_orig, original_scale_df, code, forecast, lookback, cutoff_date, max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
        diagnostics_quant_mae, diagnostics_quant_mse, diagnostics_quant_rmse, diagnostics_quant_wape, _ = self.evaluate_model(quant_diagnostics_model, "Quantized Diagnostics Residuals Transformer", quant_predictions_univ, X_test_covs, Y_test, Y_test_orig, original_scale_df, code, forecast, lookback, cutoff_date, max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
        seasonal_quant_mae, seasonal_quant_mse, seasonal_quant_rmse, seasonal_quant_wape, _ = self.evaluate_model(quant_seasonal_model, "Quantized Seasonal Residuals Transformer", quant_pred_corrected_diagnostics, X_test_seasonal_covs, Y_test, Y_test_orig, original_scale_df, code, forecast, lookback,cutoff_date=cutoff_date , max_date=max_date , scaler=scaler , eliminate_covid_data=eliminate_covid_data , covid_dates=covid_dates)

        print("\nComplete vs quantized model comparison:")
        print(f"Original MAE: {original_mae:.4f}, Quantized MAE: {original_quant_mae:.4f}")
        print(f"Original MSE: {original_mse:.4f}, Quantized MSE: {original_quant_mse:.4f}")
        print(f"Original RMSE: {original_rmse:.4f}, Quantized RMSE: {original_quant_rmse:.4f}")
        print(f"Original WAPE: {seasonal_wape:.4f}%, Quantized WAPE: {original_quant_wape:.4f}%")
        print("================================================================\n")
    
    def run_quantization_pipeline(self,exp_names, input_directory, code, lookback, forecast, cutoff_date, max_date, scaler, eliminate_covid_data=False, covid_dates=None):
        univ_model_name_in_run = "univariate_model"
        diag_model_name_in_run = "residual_diagnostics_model"
        seasonal_model_name_in_run = "residual_seasonal_model"

        run_id = self.load_mlflow_run_id_by_name(exp_names=exp_names, code=code, forecast=forecast, lookback=lookback, model_type=None, lr=1e-5)

        univ_model = load_mlflow_model(run_id, univ_model_name_in_run, custom_objects= CUSTOM_OBJECTS) 
        diagnostics_model = load_mlflow_model(run_id, diag_model_name_in_run, custom_objects= CUSTOM_OBJECTS)
        seasonal_model = load_mlflow_model(run_id, seasonal_model_name_in_run, custom_objects= CUSTOM_OBJECTS)
            

        quant_univ_model = self.manual_weight_quantization(univ_model, model_name="univariate_model")
        quant_diagnostics_model = self.manual_weight_quantization(diagnostics_model, model_name="diagnostics_model")
        quant_seasonal_model = self.manual_weight_quantization(seasonal_model, model_name="seasonal_model")

        self.save_quantized_model_weights(quant_univ_model, "univariate_model", code, forecast, lookback)
        self.save_quantized_model_weights(quant_diagnostics_model, "diagnostics_model", code, forecast, lookback)
        self.save_quantized_model_weights(quant_seasonal_model, "seasonal_model", code, forecast, lookback)

        return univ_model, diagnostics_model, seasonal_model, quant_univ_model, quant_diagnostics_model, quant_seasonal_model



if __name__ == "__main__":
    CODES_LIST = ['demanda__TOTAL', 'demanda__SERVEI_CODI__URG', 'B34','J00', 'I10', 'M54','Ch01#subch01#A00-A09']
    LOOKBACK_LIST = [7, 14, 60, 60, 182,182]
    FORECAST_LIST = [7, 14, 30, 60, 182,365]


    input_directory = '../data/FINAL_DB/full_CAT1.parquet'
    models_directory = '../transformer_outputs/models_covid_token'
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    max_date = '2025-09-30'
    cutoff_date = '2008-01-01'
    eliminate_covid_data = False
    covid_dates = None

    for code in CODES_LIST:
        for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST):
            print(f"Processing code {code} with lookback {lookback} and forecast {forecast}")
            default_config = BaseTransformerConfig()

            run_quantization_pipeline = ModelPredictionPipeline(config=default_config)
            exp_names = ["full_TRANSFORMER3_EXPERIMENTS_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260210", "full_TRANSFORMER3_TRANSFORMERS_PREDAP_HYDRA_GRID_SEARCH_20260212"]

            univ_model, diagnostics_model, seasonal_model, quant_univ_model, quant_diagnostics_model, quant_seasonal_model = run_quantization_pipeline.run_quantization_pipeline(
                exp_names=exp_names,
                input_directory=input_directory,
                code=code,
                lookback=lookback,
                forecast=forecast,
                cutoff_date=cutoff_date,
                max_date=max_date,
                scaler=scaler,
                eliminate_covid_data=eliminate_covid_data,
                covid_dates=covid_dates
            )
            run_quantization_pipeline.eval_quantization_impact(
                input_directory=input_directory,
                code=code,
                lookback=lookback,
                forecast=forecast,
                cutoff_date=cutoff_date,
                max_date=max_date,
                scaler=scaler,
                univ_model=univ_model,
                diagnostics_model=diagnostics_model,
                seasonal_model=seasonal_model,
                quant_univ_model=quant_univ_model,
                quant_diagnostics_model=quant_diagnostics_model,
                quant_seasonal_model=quant_seasonal_model,
            )

            