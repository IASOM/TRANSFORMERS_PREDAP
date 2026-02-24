

import sys


from attrs import field
import pandas as pd
import numpy as np 

from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.univariate_transformer.model_architecture_univ_transformer import (
    PositionalEncoding, RevIN
)

from src.univariate_transformer import model_architecture_univ_transformer
from src.residual_multivariate_transformers import model_architecture_residual_transformer
from sklearn.pipeline import FunctionTransformer
from save_forecast_trajectories import univariate_transformer_phase, diagnostics_transformer_phase, seasonal_transformer_phase, load_mlflow_model

import data_preparation



def load_model_weights(model, code, lookback, forecast, models_directory, model_type="univariate"):
    """
    Loads the weights of a Keras model from a specified path.
    
    Args:
        model: A compiled Keras model instance.
        model_path: The file path to the saved Keras model weights (e.g., .keras or .h5 file).
    Returns:
        The Keras model instance with loaded weights.
    """

    weights_path =  f"{models_directory}/{code}/{model_type}/{code}_{model_type}_{forecast}fh_{lookback}lb_f16_weights.h5"
    print(model.summary())
   
    
    # Try to identify mismatch before loading
    
    try:
        model.load_weights(weights_path)
        print(f"Successfully loaded weights from {weights_path}")
    except Exception as e:
        print(f"Error loading weights: {e}")
        raise e  # Re-raise after logging
    return model


def create_univariate_transformer_model(input_shape, forecast, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function="tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding=True):
    model = model_architecture_univ_transformer.build_model(
        input_shape=input_shape,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        mlp_units=mlp_units,
        activation_function=activation_function,
        dropout=dropout,
        mlp_dropout=mlp_dropout,
        n_pred=forecast,
        pos_encoding=pos_encoding
    )
    return model

def create_residual_transformer_model(input_shape, forecast, transformer_params, activation_function="tanh"):
    model = model_architecture_residual_transformer.hybrid_lstm_transformer_model(
        input_shape=input_shape,
        forecast = forecast,
        transformer_params=transformer_params,
        activation_function=activation_function
    )
    return model

def reconstruct_full_model(code, lookback, forecast, models_directory, univ_input_shape, diagnostics_input_shape, seasonal_input_shape, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function="tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding=True):
    DEFAULT_RESIDUAL_TRANSFORMER_PARAMS =  {
        'head_size': 16,
        'num_heads': 16,
        'ff_dim': 512,
        'mlp_units': [256, 128],
        'num_transformer_blocks': 2,
        'dropout': 0
    }
    
    
    univ_model = create_univariate_transformer_model(
        input_shape= univ_input_shape,
        forecast=forecast,
        head_size=head_size,
        num_heads=num_heads,
        ff_dim=ff_dim,
        num_transformer_blocks=num_transformer_blocks,
        mlp_units=mlp_units,
        activation_function=activation_function,
        dropout=dropout,
        mlp_dropout=mlp_dropout,
        n_pred=forecast,
        pos_encoding=pos_encoding,


    )

    diagnostics_model = create_residual_transformer_model(
        input_shape= diagnostics_input_shape,
        forecast = forecast,
        transformer_params=DEFAULT_RESIDUAL_TRANSFORMER_PARAMS, 
        activation_function=activation_function
    )



    seasonal_model = create_residual_transformer_model(
        input_shape= seasonal_input_shape,
        forecast = forecast,
        transformer_params=DEFAULT_RESIDUAL_TRANSFORMER_PARAMS,
        activation_function=activation_function
    )


    univ_model = load_model_weights(univ_model, code, lookback, forecast, models_directory, model_type="univariate_model")
    diagnostics_model = load_model_weights(diagnostics_model, code, lookback, forecast, models_directory, model_type="diagnostics_model")
    seasonal_model = load_model_weights(seasonal_model, code, lookback, forecast, models_directory, model_type="seasonal_model")

    return univ_model, diagnostics_model, seasonal_model

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
    models_directory = "../quantized_models"
    head_size = 32
    num_heads = 8
    ff_dim = 512
    num_transformer_blocks = 2
    mlp_units = [512,256]
    activation_function = "gelu"

    
    for code in CODES_LIST:
        for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST):
            print(f" \nProcessing code {code} with lookback {lookback} and forecast {forecast}")
            print("================================================================")

            X_test, Y_test, Y_test_orig, date_list = univariate_transformer_phase(input_directory, code, forecast, lookback, cutoff_date, max_date, relevant_feature_cols=None, scaler=scaler, eliminate_covid_data=eliminate_covid_data)
            X_train_covs, X_test_covs = diagnostics_transformer_phase(code, forecast, lookback, max_date, scaler)
            categorical_vars = ["Day_of_Week", 
                                "Month", 
                                "Season", 
                                "Holiday", 
                                "School_Vacation",
                                #"Is_Weekend",
                                ]
            X_train_seasonal_covs, X_test_seasonal_covs = seasonal_transformer_phase(code, forecast, lookback,cutoff_date, max_date, categorical_vars=categorical_vars, predictions_train=None, predictions_test= None, scaler=scaler)
            X_train_seasonal_covs = X_train_seasonal_covs.astype(float)
            X_test_seasonal_covs = X_test_seasonal_covs.astype(float)

            univ_input_shape = X_test.shape[1:]
            diagnostics_input_shape = X_test_covs.shape[1:]
            seasonal_input_shape = X_test_seasonal_covs.shape[1:]
            seasonal_input_shape = tuple((seasonal_input_shape[0], seasonal_input_shape[1]+1))

            univ_model, diagnostics_model, seasonal_model = reconstruct_full_model(
                code,
                lookback,
                forecast,
                models_directory,
                univ_input_shape,
                diagnostics_input_shape,
                seasonal_input_shape,
                head_size,
                num_heads,
                ff_dim,
                num_transformer_blocks,
                mlp_units,
                activation_function=activation_function,
                dropout=0,
                mlp_dropout=0,
                n_pred=1,
                pos_encoding=True
            )

            #Evaluate quantized model
            original_scale_df = pd.read_parquet(input_directory)
            quant_predictions_univ = univ_model.predict(X_test, verbose=0)
            quant_predictions_diagnostics_residuals = diagnostics_model.predict(X_test_covs, verbose=0)

            quant_pred_seasonal_corrected = quant_predictions_univ + quant_predictions_diagnostics_residuals
            X_train_seasonal_covs, X_test_seasonal_covs = seasonal_transformer_phase(code, forecast, lookback,cutoff_date, max_date, categorical_vars=categorical_vars, predictions_train=None, predictions_test=quant_pred_seasonal_corrected, scaler=scaler)
            X_train_seasonal_covs = X_train_seasonal_covs.astype(float)
            X_test_seasonal_covs = X_test_seasonal_covs.astype(float)
            quant_predictions_seasonal_residuals = seasonal_model.predict(X_test_seasonal_covs, verbose=0)

            quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
            quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals

            quant_predictions_to_plot = data_preparation.inverse_transform_predictions(
                quant_predictions_univ, original_scale_df, code=code, forecast=forecast, lookback=lookback, cutoff_date=cutoff_date, max_date = max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates
            )

            '''quant_seasonal_predictions_to_plot = data_preparation.inverse_transform_predictions(
                quant_pred_corrected_seasonal, original_scale_df, code=code, forecast=forecast, lookback=lookback, cutoff_date=cutoff_date, max_date = max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates
            )'''
            quant_non_negative_predictions_to_plot = np.maximum(quant_pred_corrected_seasonal, 0)
            quant_original_mae = mean_absolute_error(Y_test_orig, quant_non_negative_predictions_to_plot)
            quant_original_mse = mean_squared_error(Y_test_orig, quant_non_negative_predictions_to_plot)
            quant_original_rmse = np.sqrt(quant_original_mse)
            quant_original_wape = np.sum(np.abs(Y_test_orig - quant_non_negative_predictions_to_plot)) / np.sum(np.abs(Y_test_orig)) * 100
            print(f"Quantized Model Test Results - MAE: {quant_original_mae:.4f}, MSE: {quant_original_mse:.4f}, RMSE: {quant_original_rmse:.4f}, WAPE: {quant_original_wape:.4f}%")
            print("================================================================\n")






