import shutil

from sklearn.pipeline import FunctionTransformer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Optional, Dict, List, Tuple, Any
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.utils.experiments_utils import smart_read, get_codes_as_list
from model_architechture import model_architecture_univ_transformer
from model_architechture import model_architecture_residual_transformer
from config.base_transformer_config import BaseTransformerConfig
from production.data_preparation_in_poduction import DataPreparationInProduction
from src.training.training_utils import load_diagnostic_covariates

np.read_csv = smart_read

default_config = BaseTransformerConfig()


def compute_predap_auxiliary_metrics(predictions: np.ndarray, true_past_data: np.ndarray, mae: np.ndarray, lookback: int, confidence_level: float=0.95) -> Dict[str, float]:
        """
        Computes auxiliary evaluation metrics (MAE, MSE, WAPE) for the given predictions and true values.

        Args:
            predictions (np.ndarray): The array of predicted values.
            true_past_data (np.ndarray): The array of true past values.
            mae (np.ndarray): The mean absolute error.
            lookback (int): The lookback period.
            confidence_level (float): The confidence level for the confidence interval.
        Returns:
            A dictionary containing the computed auxiliary metrics.
        """

        pred_series = pd.Series(predictions.flatten())
        true_past_data = pd.Series(true_past_data.flatten())
        full_series = pd.concat([true_past_data, pred_series], ignore_index=True)

        # 4. Calculate the Laplace multiplier (k) instead of Z-score
        k_multiplier = -np.log(2 * (1.0 - confidence_level))

        # 5. Compute the final bounds centered around your current predictions
        ci_margin = k_multiplier * mae

        ci_lower = pred_series - ci_margin
        ci_upper = pred_series + ci_margin

        #compute 1st and 2nd derivatives (velocity and acceleration)
        day_delta = 1.0
        velocity = full_series.diff().div(day_delta)
        acceleration = full_series.diff().div(day_delta)

        velocity = velocity.iloc[-len(pred_series):]  # Keep only the velocity for the predicted points
        acceleration = acceleration.iloc[-len(pred_series):]  # Keep only the acceleration for the predicted points

        return ci_lower.values, ci_upper.values, velocity.values, acceleration.values

class ModelPredictionPipeline(DataPreparationInProduction):
    def __init__(self, config: BaseTransformerConfig):
        self.config = config
        self.config.print_config()

    def create_univariate_transformer_model(self, input_shape, forecast, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function="tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding=True):
        """Creates a univariate transformer model for time series forecasting.
        Args:
            input_shape (tuple): The shape of the input data (lookback, num_features).
            forecast (int): The number of future time steps to predict.
            head_size (int): The dimensionality of the attention heads.
            num_heads (int): The number of attention heads.
            ff_dim (int): The dimensionality of the feed-forward layer.
            num_transformer_blocks (int): The number of transformer blocks to stack.
            mlp_units (list of int): A list specifying the number of units in each MLP layer after the transformer blocks.
            activation_function (str): The activation function to use in the transformer and MLP layers.
            dropout (float): The dropout rate for regularization in the transformer blocks.
            mlp_dropout (float): The dropout rate for regularization in the MLP layers.
            n_pred (int): The number of future time steps to predict (forecast horizon).
            pos_encoding (bool): Whether to use positional encoding in the model.
        Returns:
            A compiled Keras model instance representing the univariate transformer.
        """
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

    def create_residual_transformer_model(self, input_shape, forecast, transformer_params, activation_function="tanh"):
        """Creates a residual transformer model for either diagnostics or seasonal components.
        Args:
            input_shape (tuple): The shape of the input data (lookback, num_features).
            forecast (int): The number of future time steps to predict.
            transformer_params (dict): A dictionary containing the parameters for the transformer architecture, including:
                - head_size (int): The dimensionality of the attention heads.
                - num_heads (int): The number of attention heads.
                - ff_dim (int): The dimensionality of the feed-forward layer.
                - mlp_units (list of int): A list specifying the number of units in each MLP layer after the transformer blocks.
                - num_transformer_blocks (int): The number of transformer blocks to stack.
            activation_function (str): The activation function to use in the transformer and MLP layers.
        Returns:
            A compiled Keras model instance representing the residual transformer."""
        model = model_architecture_residual_transformer.hybrid_lstm_transformer_model(
            input_shape=input_shape,
            forecast = forecast,
            transformer_params=transformer_params,
            activation_function=activation_function
        )
        return model
    
    def load_model_weights(self,model, code, lookback, forecast, models_directory, model_type="univariate"):
        """
        Loads the weights of a Keras model from a specified path.
        
        Args:
            model: A compiled Keras model instance.
            model_path: The file path to the saved Keras model weights (e.g., .keras or .h5 file).
        Returns:
            The Keras model instance with loaded weights.
        """

        weights_path =  f"{models_directory}/{code}/{model_type}/{code}_{model_type}_{forecast}fh_{lookback}lb_f16_weights.h5"
        
        #print(model.summary())
        # Try to identify mismatch before loading
        
        try:
            model.load_weights(weights_path)
            print(f"Successfully loaded weights from {weights_path}")
        except Exception as e:
            print(f"Error loading weights: {e}")
            raise e  # Re-raise after logging
        return model

    def reconstruct_full_model(self, code: str, lookback: int, forecast: int, models_directory: str, univ_input_shape: Tuple[int], diagnostics_input_shape: Tuple[int], seasonal_input_shape: Tuple[int], head_size: int, num_heads: int, ff_dim: int, num_transformer_blocks: int, mlp_units: List[int], activation_function: str="tanh", dropout: float=0, mlp_dropout: float=0, n_pred: int=1, pos_encoding=True):
        """Reconstructs the full model architecture for univariate, diagnostics residual, and seasonal residual components, and loads the corresponding weights for each sub-model.
        
        Returns:
            A tuple containing the reconstructed univariate model, diagnostics residual model, and seasonal residual model with loaded weights.
        """
        
        DEFAULT_RESIDUAL_TRANSFORMER_PARAMS =  {
            'head_size': 16,
            'num_heads': 16,
            'ff_dim': 512,
            'mlp_units': [256, 128],
            'num_transformer_blocks': 2,
            'dropout': 0
        }
        
        
        univ_model = self.create_univariate_transformer_model(
            input_shape= univ_input_shape,
            forecast= forecast,
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

        diagnostics_model = self.create_residual_transformer_model(
            input_shape= diagnostics_input_shape,
            forecast = forecast,
            transformer_params=DEFAULT_RESIDUAL_TRANSFORMER_PARAMS, 
            activation_function=activation_function
        )



        seasonal_model = self.create_residual_transformer_model(
            input_shape= seasonal_input_shape,
            forecast = forecast,
            transformer_params=DEFAULT_RESIDUAL_TRANSFORMER_PARAMS,
            activation_function=activation_function
        )


        univ_model = self.load_model_weights(univ_model, code, lookback, forecast, models_directory, model_type="univariate_model")
        diagnostics_model = self.load_model_weights(diagnostics_model, code, lookback, forecast, models_directory, model_type="diagnostics_model")
        seasonal_model = self.load_model_weights(seasonal_model, code, lookback, forecast, models_directory, model_type="seasonal_model")

        return univ_model, diagnostics_model, seasonal_model

    def compute_predap_auxiliary_metrics(self, predictions: np.ndarray, true_past_data: np.ndarray, mae: np.ndarray, lookback: int, confidence_level: float=0.95) -> Dict[str, float]:
        """
        Computes auxiliary evaluation metrics (MAE, MSE, WAPE) for the given predictions and true values.

        Args:
            predictions (np.ndarray): The array of predicted values.
            true_past_data (np.ndarray): The array of true past values.
            mae (np.ndarray): The mean absolute error.
            lookback (int): The lookback period.
            confidence_level (float): The confidence level for the confidence interval.
        Returns:
            
        """

        pred_series = pd.Series(predictions.flatten())
        true_past_data = pd.Series(true_past_data.flatten())
        full_series = pd.concat([true_past_data, pred_series], ignore_index=True)

        # 4. Calculate the Laplace multiplier (k) instead of Z-score
        k_multiplier = -np.log(2 * (1.0 - confidence_level))

        # 5. Compute the final bounds centered around your current predictions
        ci_margin = k_multiplier * mae

        ci_lower = pred_series - ci_margin
        ci_upper = pred_series + ci_margin

        #compute 1st and 2nd derivatives (velocity and acceleration)
        day_delta = 1.0
        velocity = full_series.diff().div(day_delta)
        acceleration = full_series.diff().div(day_delta)

        velocity = velocity.iloc[-len(pred_series):]  # Keep only the velocity for the predicted points
        acceleration = acceleration.iloc[-len(pred_series):]  # Keep only the acceleration for the predicted points

        return ci_lower.values, ci_upper.values, velocity.values, acceleration.values

    
    def individual_run_reconstruct_save_results_pipeline(
            self, 
            input_directory: str,
            code: str, 
            LOOKBACK_LIST: List[int], 
            FORECAST_LIST: List[int], 
            final_output_predictions: Optional[np.ndarray], 
            final_output_df: pd.DataFrame,
            dates: List[str],
            
            ) -> pd.DataFrame:
        """Runs the full pipeline to reconstruct the model, make predictions, and save results for a given code and list of lookback and forecast combinations. 
        Args:
            code (str): The code for which to run the pipeline (e.g., 'demanda__TOTAL').
            LOOKBACK_LIST (List[int]): A list of lookback periods to iterate
            FORECAST_LIST (List[int]): A list of forecast horizons to iterate.
            final_output_predictions (Optional[np.ndarray]): An optional array to store final output predictions across iterations
            final_output_df (pd.DataFrame): A DataFrame to store the final output predictions along with corresponding dates.
        Returns:
            A DataFrame containing the final output predictions for each forecast horizon along with corresponding dates.
        """
        
        #dates = ["2025-12-23","2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31"]
        output_chunks = []

        for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST): 
        
            for max_date in dates:
            
                auxiliary_output_df = pd.DataFrame()  # Temporary DataFrame for current iteration
    
                X_univ_data_production, Y_univ_data_production, df_timestamp_production = self.prepare_prediction_univ_data(
                    data_path=input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    eliminate_covid_data=self.config.eliminate_covid_data,
                    covid_token=self.config.covid_token,
                    production_mode=True
                )

                X_diagnostics_data_production, Y_diagnostics_data_production = self.prepare_prediction_diagnostics_data(
                    data_path=input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date = self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    covid_token=self.config.covid_token,
                    production_mode=True
                )

                X_seasonal_data_production = self.prepare_prediction_seasonal_data(
                    data_path=input_directory,
                    code=code,
                    forecast=forecast,
                    lookback=lookback,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
                    predictions_train=None,
                    predictions_test=None,
                    scaler=self.config.scaler,
                )

                X_univ_data, Y_univ_data, df_timestamp = self.prepare_prediction_univ_data(
                    data_path=input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    eliminate_covid_data=self.config.eliminate_covid_data,
                    covid_token=self.config.covid_token,
                    production_mode=False
                )

                X_diagnostics_data, Y_diagnostics_data = self.prepare_prediction_diagnostics_data(
                    data_path=input_directory, 
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    covid_token=self.config.covid_token,
                    production_mode=False
                )

                X_seasonal_data = self.prepare_prediction_seasonal_data(
                    data_path=input_directory,
                    code=code,
                    forecast=forecast,
                    lookback=lookback,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
                    predictions_train=None,
                    predictions_test=None,
                    scaler=self.config.scaler,
                )
                
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
                #Make predictions for the known historical data
                quant_predictions_univ = univ_model(X_univ_data, training=False).numpy()
                quant_predictions_diagnostics_residuals = diagnostics_model(X_diagnostics_data, training=False).numpy()
                
                preds_reshaped = quant_predictions_diagnostics_residuals[:, -forecast:, np.newaxis]
                X_seasonal_with_preds = np.concatenate((X_seasonal_data, preds_reshaped), axis=2)
                #X_seasonal_with_preds = X_seasonal_data.copy() #temporary put only the original seasonal data, without the diagnostics predictions, to avoid potential issues with input shape or model expectations.
                quant_predictions_seasonal_residuals = seasonal_model(X_seasonal_with_preds, training=False).numpy()


                quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
                quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals
                #Clip predictions to be non-negative, as we are predicting production values that cannot be negative
                quant_pred_corrected_seasonal = np.clip(quant_pred_corrected_seasonal, 0, +np.inf)

                mae = np.mean(np.abs(quant_pred_corrected_seasonal - Y_univ_data))
                mse = np.mean((quant_pred_corrected_seasonal - Y_univ_data)**2)
                wape = np.sum(np.abs(quant_pred_corrected_seasonal - Y_univ_data)) / np.sum(np.abs(Y_univ_data) + 1e-8)
                print(f"MAE for code {code} with lookback {lookback} and forecast {forecast}: {mae}")
                print(f"MSE for code {code} with lookback {lookback} and forecast {forecast}: {mse}")
                print(f"WAPE for code {code} with lookback {lookback} and forecast {forecast}: {wape*100:.2f}%")

                #Make predictions for the unknown historical data
                quant_predictions_univ_production = univ_model(X_univ_data_production, training=False).numpy()
                quant_predictions_diagnostics_residuals_production = diagnostics_model(X_diagnostics_data_production, training=False).numpy()

                preds_reshaped_production = quant_predictions_diagnostics_residuals_production[:, -forecast:, np.newaxis]
                X_seasonal_with_preds = np.concatenate((X_seasonal_data_production, preds_reshaped_production), axis=2)
                #X_seasonal_with_preds = X_seasonal_data.copy() #temporary put only the original seasonal data, without the diagnostics predictions, to avoid potential issues with input shape or model expectations.
                quant_predictions_seasonal_residuals_production = seasonal_model(X_seasonal_with_preds, training=False).numpy()


                quant_pred_corrected_diagnostics_production = quant_predictions_univ_production + quant_predictions_diagnostics_residuals_production
                quant_pred_corrected_seasonal_production = quant_pred_corrected_diagnostics_production + quant_predictions_seasonal_residuals_production
                #Clip predictions to be non-negative, as we are predicting production values that cannot be negative
                quant_pred_corrected_seasonal_production = np.clip(quant_pred_corrected_seasonal_production, 0, +np.inf)

                ci_lower, ci_upper, velocity, acceleration = self.compute_predap_auxiliary_metrics(quant_pred_corrected_seasonal_production.flatten(), Y_univ_data.flatten(), mae, lookback=lookback, confidence_level=0.95)

                target_date = df_timestamp_production[-forecast:].values
                init_forecast_date = df_timestamp.iloc[-forecast-1]
                final_forecast_date = df_timestamp.iloc[-1]

                auxiliary_output_df["target_date"] = target_date
                auxiliary_output_df["init_forecast_date"] = init_forecast_date
                auxiliary_output_df["final_forecast_date"] = final_forecast_date
                auxiliary_output_df["code"] = code
                auxiliary_output_df["forecast"] = forecast
                auxiliary_output_df["predictions"] = quant_pred_corrected_seasonal_production.flatten()[:forecast]
                auxiliary_output_df["ci_lower"] = ci_lower
                auxiliary_output_df["ci_upper"] = ci_upper
                auxiliary_output_df["velocity"] = velocity
                auxiliary_output_df["acceleration"] = acceleration

                init_time_concatenation = pd.Timestamp.now()
                output_chunks.append(auxiliary_output_df)
                #final_output_df = pd.concat([final_output_df, auxiliary_output_df], ignore_index=True)
                end_time_concatenation = pd.Timestamp.now()
                print(f"Time taken to concatenate data: {end_time_concatenation - init_time_concatenation}")

        #final_output_df.to_csv(f"../quantized_models/{code}/final_output_predictions_{code.replace(':', '#')}.csv", index=False)
        if output_chunks:
            output_chunks_df = pd.concat(output_chunks, ignore_index=True)

        final_output_df = pd.concat([final_output_df, output_chunks_df], ignore_index=True)
        return final_output_df
    

    def create_inference_data(self, input_directory: str, code: str, lookback: int, forecast: int, max_date: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepares the input data for inference by loading and processing the univariate, diagnostics, and seasonal datasets.

        Args:
            input_directory (str): The directory where the input data is stored.
            code (str): The code for which to prepare the inference data.
            lookback (int): The number of past time steps to consider for making predictions.
            forecast (int): The number of future time steps to predict.
            max_date (str): The maximum date for which to prepare the inference data.


        """
        auxiliary_output_df = pd.DataFrame()  # Temporary DataFrame for current iteration
    
        X_univ_data_production, Y_univ_data_production, df_timestamp_production = self.prepare_prediction_univ_data(
            data_path=input_directory,
            code=code,
            lookback=lookback,
            forecast=forecast,
            cutoff_date=self.config.cutoff_date,
            max_date=max_date,
            scaler=self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data,
            covid_token=self.config.covid_token,
            production_mode=True
        )

        X_diagnostics_data_production, Y_diagnostics_data_production = self.prepare_prediction_diagnostics_data(
            data_path=input_directory,
            code=code,
            lookback=lookback,
            forecast=forecast,
            cutoff_date = self.config.cutoff_date,
            max_date=max_date,
            scaler=self.config.scaler,
            covid_token=self.config.covid_token,
            production_mode=True
        )

        X_seasonal_data_production = self.prepare_prediction_seasonal_data(
            data_path=input_directory,
            code=code,
            forecast=forecast,
            lookback=lookback,
            cutoff_date=self.config.cutoff_date,
            max_date=max_date,
            categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
            predictions_train=None,
            predictions_test=None,
            scaler=self.config.scaler,
        )

        X_univ_data, Y_univ_data, df_timestamp = self.prepare_prediction_univ_data(
            data_path=input_directory,
            code=code,
            lookback=lookback,
            forecast=forecast,
            cutoff_date=self.config.cutoff_date,
            max_date=max_date,
            scaler=self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data,
            covid_token=self.config.covid_token,
            production_mode=False
        )

        X_diagnostics_data, Y_diagnostics_data = self.prepare_prediction_diagnostics_data(
            data_path=input_directory, 
            code=code,
            lookback=lookback,
            forecast=forecast,
            cutoff_date=self.config.cutoff_date,
            max_date=max_date,
            scaler=self.config.scaler,
            covid_token=self.config.covid_token,
            production_mode=False
        )

        X_seasonal_data = self.prepare_prediction_seasonal_data(
            data_path=input_directory,
            code=code,
            forecast=forecast,
            lookback=lookback,
            cutoff_date=self.config.cutoff_date,
            max_date=max_date,
            categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
            predictions_train=None,
            predictions_test=None,
            scaler=self.config.scaler,
        )
        
        

        return (X_univ_data, 
                X_univ_data_production,
                Y_univ_data, 
                Y_univ_data_production, 
                X_diagnostics_data, 
                X_diagnostics_data_production, 
                X_seasonal_data, 
                X_seasonal_data_production, 
                df_timestamp_production,
                df_timestamp)
    

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
    def run_reconstruct_save_results_pipeline(
            self, 
            config: BaseTransformerConfig,
            final_output_df: pd.DataFrame,
            dates: List[str],
            univ_model: tf.keras.Model,
            diagnostics_model: tf.keras.Model,
            seasonal_model: tf.keras.Model,
            X_univ_data: np.ndarray, Y_univ_data: np.ndarray, df_timestamp: pd.Series,
            X_univ_data_production: np.ndarray,
            X_diagnostics_data: np.ndarray, X_diagnostics_data_production: np.ndarray,
            X_seasonal_data: np.ndarray, X_seasonal_data_production: np.ndarray,
            df_timestamp_production: pd.Series,
            
            ) -> pd.DataFrame:
        """Runs the full pipeline to reconstruct the model, make predictions, and save results for a given code and list of lookback and forecast combinations. 
        Args:
            config (BaseTransformerConfig): The configuration for the transformer model.
            input_directory (str): The directory containing the input data.
            final_output_predictions (Optional[np.ndarray]): An optional array to store final output predictions across iterations
            final_output_df (pd.DataFrame): A DataFrame to store the final output predictions along with corresponding dates.
            dates (List[str]): A list of dates for which to make predictions.
            univ_model (tf.keras.Model): The universal model.
            diagnostics_model (tf.keras.Model): The diagnostics model.
            seasonal_model (tf.keras.Model): The seasonal model.
            X_univ_data (np.ndarray): The universal input data.
            Y_univ_data (np.ndarray): The universal target data.
            df_timestamp (pd.Series): The timestamp series for the training data.
            X_univ_data_production (np.ndarray): The universal input data for production.
            X_diagnostics_data (np.ndarray): The diagnostics input data.
            X_diagnostics_data_production (np.ndarray): The diagnostics input data for production.
            X_seasonal_data (np.ndarray): The seasonal input data.
            X_seasonal_data_production (np.ndarray): The seasonal input data for production.
            df_timestamp_production (pd.Series): The timestamp series for the production data.
            final_output_predictions (Optional[np.ndarray]): An optional array to store final output predictions across iterations
            final_output_df (pd.DataFrame): A DataFrame to store the final output predictions along with corresponding dates.
        Returns:
            A DataFrame containing the final output predictions for each forecast horizon along with corresponding dates.
        """
        
        #dates = ["2025-12-23","2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31"]
        output_chunks = []

        for max_date in dates:
            print(f"Processing code: {config.code}, lookback: {config.lookback}, forecast: {config.forecast}, max_date: {max_date}")
            auxiliary_output_df = pd.DataFrame()  # Temporary DataFrame for current iteration

            #Make predictions for the known historical data
            quant_predictions_univ = univ_model(X_univ_data, training=False).numpy()
            quant_predictions_diagnostics_residuals = diagnostics_model(X_diagnostics_data, training=False).numpy()
            
            preds_reshaped = quant_predictions_diagnostics_residuals[:, -config.forecast:, np.newaxis]
            X_seasonal_with_preds = np.concatenate((X_seasonal_data, preds_reshaped), axis=2)
            #X_seasonal_with_preds = X_seasonal_data.copy() #temporary put only the original seasonal data, without the diagnostics predictions, to avoid potential issues with input shape or model expectations.
            quant_predictions_seasonal_residuals = seasonal_model(X_seasonal_with_preds, training=False).numpy()


            quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
            quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals
            #Clip predictions to be non-negative, as we are predicting production values that cannot be negative
            quant_pred_corrected_seasonal = np.clip(quant_pred_corrected_seasonal, 0, +np.inf)

            mae = np.mean(np.abs(quant_pred_corrected_seasonal - Y_univ_data))
            mse = np.mean((quant_pred_corrected_seasonal - Y_univ_data)**2)
            wape = np.sum(np.abs(quant_pred_corrected_seasonal - Y_univ_data)) / np.sum(np.abs(Y_univ_data) + 1e-8)
            print(f"MAE for code {config.code} with lookback {config.lookback} and forecast {config.forecast}: {mae}")
            print(f"MSE for code {config.code} with lookback {config.lookback} and forecast {config.forecast}: {mse}")
            print(f"WAPE for code {config.code} with lookback {config.lookback} and forecast {config.forecast}: {wape*100:.2f}%")

            #Make predictions for the unknown historical data
            quant_predictions_univ_production = univ_model(X_univ_data_production, training=False).numpy()
            quant_predictions_diagnostics_residuals_production = diagnostics_model(X_diagnostics_data_production, training=False).numpy()

            preds_reshaped_production = quant_predictions_diagnostics_residuals_production[:, -config.forecast:, np.newaxis]
            X_seasonal_with_preds = np.concatenate((X_seasonal_data_production, preds_reshaped_production), axis=2)
            #X_seasonal_with_preds = X_seasonal_data.copy() #temporary put only the original seasonal data, without the diagnostics predictions, to avoid potential issues with input shape or model expectations.
            quant_predictions_seasonal_residuals_production = seasonal_model(X_seasonal_with_preds, training=False).numpy()


            quant_pred_corrected_diagnostics_production = quant_predictions_univ_production + quant_predictions_diagnostics_residuals_production
            quant_pred_corrected_seasonal_production = quant_pred_corrected_diagnostics_production + quant_predictions_seasonal_residuals_production
            #Clip predictions to be non-negative, as we are predicting production values that cannot be negative
            quant_pred_corrected_seasonal_production = np.clip(quant_pred_corrected_seasonal_production, 0, +np.inf)

            ci_lower, ci_upper, velocity, acceleration = self.compute_predap_auxiliary_metrics(quant_pred_corrected_seasonal_production.flatten(), Y_univ_data.flatten(), mae, lookback=config.lookback, confidence_level=0.95)

            target_date = df_timestamp_production[-config.forecast:].values
            init_forecast_date = df_timestamp.iloc[-config.forecast-1]
            final_forecast_date = df_timestamp.iloc[-1]

            auxiliary_output_df["target_date"] = target_date
            auxiliary_output_df["init_forecast_date"] = init_forecast_date
            auxiliary_output_df["final_forecast_date"] = final_forecast_date
            auxiliary_output_df["code"] = config.code
            auxiliary_output_df["forecast"] = config.forecast
            auxiliary_output_df["predictions"] = quant_pred_corrected_seasonal_production.flatten()[:config.forecast]
            auxiliary_output_df["ci_lower"] = ci_lower
            auxiliary_output_df["ci_upper"] = ci_upper
            auxiliary_output_df["velocity"] = velocity
            auxiliary_output_df["acceleration"] = acceleration

            init_time_concatenation = pd.Timestamp.now()
            output_chunks.append(auxiliary_output_df)
            #final_output_df = pd.concat([final_output_df, auxiliary_output_df], ignore_index=True)
            end_time_concatenation = pd.Timestamp.now()
            print(f"Time taken to concatenate data: {end_time_concatenation - init_time_concatenation}")

        #final_output_df.to_csv(f"../quantized_models/{code}/final_output_predictions_{code.replace(':', '#')}.csv", index=False)
        if output_chunks:
            output_chunks_df = pd.concat(output_chunks, ignore_index=True)

        final_output_df = pd.concat([final_output_df, output_chunks_df], ignore_index=True)
        return final_output_df
    '''def run_reconstruct_save_results_pipeline(
            self, 
            input_directory: str,
            code: str, 
            LOOKBACK_LIST: List[int], 
            FORECAST_LIST: List[int], 
            final_output_predictions: Optional[np.ndarray], 
            final_output_df: pd.DataFrame,
            dates: List[str],
            
            ) -> pd.DataFrame:
        """Runs the full pipeline to reconstruct the model, make predictions, and save results for a given code and list of lookback and forecast combinations. 
        Args:
            code (str): The code for which to run the pipeline (e.g., 'demanda__TOTAL').
            LOOKBACK_LIST (List[int]): A list of lookback periods to iterate
            FORECAST_LIST (List[int]): A list of forecast horizons to iterate.
            final_output_predictions (Optional[np.ndarray]): An optional array to store final output predictions across iterations
            final_output_df (pd.DataFrame): A DataFrame to store the final output predictions along with corresponding dates.
        Returns:
            A DataFrame containing the final output predictions for each forecast horizon along with corresponding dates.
        """
        
        #dates = ["2025-12-23","2025-12-24", "2025-12-25", "2025-12-26", "2025-12-27", "2025-12-28", "2025-12-29", "2025-12-30", "2025-12-31"]
        output_chunks = []

        for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST): 

            for max_date in dates:
                print(f"Processing code: {code}, lookback: {lookback}, forecast: {forecast}, max_date: {max_date}")
                auxiliary_output_df = pd.DataFrame()  # Temporary DataFrame for current iteration
    
                X_univ_data_production, Y_univ_data_production, df_timestamp_production = self.prepare_prediction_univ_data(
                    data_path=input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    eliminate_covid_data=self.config.eliminate_covid_data,
                    covid_token=self.config.covid_token,
                    production_mode=True
                )

                X_diagnostics_data_production, Y_diagnostics_data_production = self.prepare_prediction_diagnostics_data(
                    data_path=input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date = self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    covid_token=self.config.covid_token,
                    production_mode=True
                )

                X_seasonal_data_production = self.prepare_prediction_seasonal_data(
                    data_path=input_directory,
                    code=code,
                    forecast=forecast,
                    lookback=lookback,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
                    predictions_train=None,
                    predictions_test=None,
                    scaler=self.config.scaler,
                )

                X_univ_data, Y_univ_data, df_timestamp = self.prepare_prediction_univ_data(
                    data_path=input_directory,
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    eliminate_covid_data=self.config.eliminate_covid_data,
                    covid_token=self.config.covid_token,
                    production_mode=False
                )

                X_diagnostics_data, Y_diagnostics_data = self.prepare_prediction_diagnostics_data(
                    data_path=input_directory, 
                    code=code,
                    lookback=lookback,
                    forecast=forecast,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    scaler=self.config.scaler,
                    covid_token=self.config.covid_token,
                    production_mode=False
                )

                X_seasonal_data = self.prepare_prediction_seasonal_data(
                    data_path=input_directory,
                    code=code,
                    forecast=forecast,
                    lookback=lookback,
                    cutoff_date=self.config.cutoff_date,
                    max_date=max_date,
                    categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
                    predictions_train=None,
                    predictions_test=None,
                    scaler=self.config.scaler,
                )
                
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
                #Make predictions for the known historical data
                quant_predictions_univ = univ_model(X_univ_data, training=False).numpy()
                quant_predictions_diagnostics_residuals = diagnostics_model(X_diagnostics_data, training=False).numpy()
                
                preds_reshaped = quant_predictions_diagnostics_residuals[:, -forecast:, np.newaxis]
                X_seasonal_with_preds = np.concatenate((X_seasonal_data, preds_reshaped), axis=2)
                #X_seasonal_with_preds = X_seasonal_data.copy() #temporary put only the original seasonal data, without the diagnostics predictions, to avoid potential issues with input shape or model expectations.
                quant_predictions_seasonal_residuals = seasonal_model(X_seasonal_with_preds, training=False).numpy()


                quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
                quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals
                #Clip predictions to be non-negative, as we are predicting production values that cannot be negative
                quant_pred_corrected_seasonal = np.clip(quant_pred_corrected_seasonal, 0, +np.inf)

                mae = np.mean(np.abs(quant_pred_corrected_seasonal - Y_univ_data))
                mse = np.mean((quant_pred_corrected_seasonal - Y_univ_data)**2)
                wape = np.sum(np.abs(quant_pred_corrected_seasonal - Y_univ_data)) / np.sum(np.abs(Y_univ_data) + 1e-8)
                print(f"MAE for code {code} with lookback {lookback} and forecast {forecast}: {mae}")
                print(f"MSE for code {code} with lookback {lookback} and forecast {forecast}: {mse}")
                print(f"WAPE for code {code} with lookback {lookback} and forecast {forecast}: {wape*100:.2f}%")

                #Make predictions for the unknown historical data
                quant_predictions_univ_production = univ_model(X_univ_data_production, training=False).numpy()
                quant_predictions_diagnostics_residuals_production = diagnostics_model(X_diagnostics_data_production, training=False).numpy()

                preds_reshaped_production = quant_predictions_diagnostics_residuals_production[:, -forecast:, np.newaxis]
                X_seasonal_with_preds = np.concatenate((X_seasonal_data_production, preds_reshaped_production), axis=2)
                #X_seasonal_with_preds = X_seasonal_data.copy() #temporary put only the original seasonal data, without the diagnostics predictions, to avoid potential issues with input shape or model expectations.
                quant_predictions_seasonal_residuals_production = seasonal_model(X_seasonal_with_preds, training=False).numpy()


                quant_pred_corrected_diagnostics_production = quant_predictions_univ_production + quant_predictions_diagnostics_residuals_production
                quant_pred_corrected_seasonal_production = quant_pred_corrected_diagnostics_production + quant_predictions_seasonal_residuals_production
                #Clip predictions to be non-negative, as we are predicting production values that cannot be negative
                quant_pred_corrected_seasonal_production = np.clip(quant_pred_corrected_seasonal_production, 0, +np.inf)

                ci_lower, ci_upper, velocity, acceleration = self.compute_predap_auxiliary_metrics(quant_pred_corrected_seasonal_production.flatten(), Y_univ_data.flatten(), mae, lookback=lookback, confidence_level=0.95)

                target_date = df_timestamp_production[-forecast:].values
                init_forecast_date = df_timestamp.iloc[-forecast-1]
                final_forecast_date = df_timestamp.iloc[-1]

                auxiliary_output_df["target_date"] = target_date
                auxiliary_output_df["init_forecast_date"] = init_forecast_date
                auxiliary_output_df["final_forecast_date"] = final_forecast_date
                auxiliary_output_df["code"] = code
                auxiliary_output_df["forecast"] = forecast
                auxiliary_output_df["predictions"] = quant_pred_corrected_seasonal_production.flatten()[:forecast]
                auxiliary_output_df["ci_lower"] = ci_lower
                auxiliary_output_df["ci_upper"] = ci_upper
                auxiliary_output_df["velocity"] = velocity
                auxiliary_output_df["acceleration"] = acceleration

                init_time_concatenation = pd.Timestamp.now()
                output_chunks.append(auxiliary_output_df)
                #final_output_df = pd.concat([final_output_df, auxiliary_output_df], ignore_index=True)
                end_time_concatenation = pd.Timestamp.now()
                print(f"Time taken to concatenate data: {end_time_concatenation - init_time_concatenation}")

        #final_output_df.to_csv(f"../quantized_models/{code}/final_output_predictions_{code.replace(':', '#')}.csv", index=False)
        if output_chunks:
            output_chunks_df = pd.concat(output_chunks, ignore_index=True)

        final_output_df = pd.concat([final_output_df, output_chunks_df], ignore_index=True)
        return final_output_df'''

    def save_final_output_predictions(self, final_output_df: pd.DataFrame, output_path: str = "../production_predictions/final_output_predictions"):
        """
        Saves the final output predictions DataFrame to a parquet file.

        Args:
            final_output_df (pd.DataFrame): The DataFrame containing the final output predictions along with corresponding dates.
            code (str): The code for which the predictions were made, used for naming the output file.
        """
        
        table = pa.Table.from_pandas(final_output_df, preserve_index=False)
        #output_path = self.config.production_predictions_dir

        ds.write_dataset(
            table,
            base_dir=output_path,
            format="parquet",
            partitioning=["code"],
            existing_data_behavior="delete_matching"  # This will overwrite existing partitions with the same code
        )

    def save_final_output_predictions_multirun(
        self,
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

    def merge_ds_files(self, input_base_dir: str = "../hydra_production_predictions/final_output_predictions", output_base_dir: str = "../hydra_production_predictions/merged_final_output_predictions"):
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


    def compute_code_prediction_features(
        self,
        final_output_df: pd.DataFrame,
        code: str,
        lookback: int,
        real_data_df: Optional[pd.DataFrame] = None,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Computes prediction features for a single code from final_output_df:
        1) Confidence interval using the last rolling window of size lookback.
        2) WAPE for predictions (when real values are available).
        3) First and second derivatives (velocity and acceleration) of predictions.

        Args:
            final_output_df (pd.DataFrame): DataFrame with at least ['code', 'target_date', 'predictions'].
            code (str): Single code to process.
            lookback (int): Rolling window size used for confidence interval estimation.
            real_data_df (Optional[pd.DataFrame]): Optional DataFrame with real values.
                Expected columns: ['timestamp', <real_code_column>], where <real_code_column>
                is code with '#' replaced by ':'.
            confidence_level (float): Confidence level for CI (default 0.95).

        Returns:
            pd.DataFrame: Code-filtered DataFrame enriched with CI, WAPE, velocity,
                and acceleration columns.
        """
        required_cols = {"code", "target_date", "predictions"}
        missing_cols = required_cols - set(final_output_df.columns)
        if missing_cols:
            raise ValueError(
                f"Missing required columns in final_output_df: {sorted(missing_cols)}"
            )
        if lookback <= 0:
            raise ValueError("lookback must be a positive integer")

        df_code = final_output_df.loc[final_output_df["code"] == code].copy()
        if df_code.empty:
            raise ValueError(f"No rows found for code '{code}' in final_output_df")

        df_code["target_date"] = pd.to_datetime(df_code["target_date"], errors="coerce")
        if "forecast_date" in df_code.columns:
            df_code["forecast_date"] = pd.to_datetime(df_code["forecast_date"], errors="coerce")
            df_code = df_code.sort_values(["forecast_date", "target_date"]).reset_index(drop=True)
        else:
            df_code = df_code.sort_values("target_date").reset_index(drop=True)

        # CI is computed from the last lookback predictions (excluding current point).
        pred_series = pd.to_numeric(df_code["predictions"], errors="coerce")
        rolling_mean = pred_series.shift(1).rolling(window=lookback, min_periods=1).mean()
        rolling_std = pred_series.shift(1).rolling(window=lookback, min_periods=2).std(ddof=1).fillna(0.0)

        alpha_tail = (1.0 + confidence_level) / 2.0
        z_score = NormalDist().inv_cdf(alpha_tail)

        ci_margin = z_score * rolling_std
        df_code["ci_window_mean"] = rolling_mean
        df_code["ci_window_std"] = rolling_std
        df_code["ci_lower"] = pred_series - ci_margin
        df_code["ci_upper"] = pred_series + ci_margin

        # First and second derivatives over time (daily units).
        day_delta = (
            df_code["target_date"].diff().dt.total_seconds().div(86400.0).replace(0, np.nan)
        )
        df_code["velocity"] = pred_series.diff().div(day_delta)
        df_code["acceleration"] = df_code["velocity"].diff().div(day_delta)

        # WAPE computation when actuals are available.
        if "actual" in df_code.columns:
            df_code["actual"] = pd.to_numeric(df_code["actual"], errors="coerce")
        else:
            df_code["actual"] = np.nan

        if real_data_df is not None:
            real_code = code.replace("#", ":")
            if "timestamp" not in real_data_df.columns:
                raise ValueError("real_data_df must contain a 'timestamp' column")
            if real_code not in real_data_df.columns:
                raise ValueError(
                    f"real_data_df must contain column '{real_code}' for code '{code}'"
                )

            real_slice = real_data_df[["timestamp", real_code]].copy()
            real_slice["timestamp"] = pd.to_datetime(real_slice["timestamp"], errors="coerce")
            real_slice = real_slice.rename(columns={"timestamp": "target_date", real_code: "actual"})

            df_code = df_code.merge(real_slice, on="target_date", how="left", suffixes=("", "_real"))
            if "actual_real" in df_code.columns:
                df_code["actual"] = df_code["actual_real"]
                df_code = df_code.drop(columns=["actual_real"])

        df_code["wape_row"] = np.nan
        df_code["wape_code"] = np.nan
        valid_actuals = df_code["actual"].notna()
        if valid_actuals.any():
            abs_error = np.abs(
                pd.to_numeric(df_code.loc[valid_actuals, "predictions"], errors="coerce")
                - df_code.loc[valid_actuals, "actual"]
            )
            abs_actual = np.abs(df_code.loc[valid_actuals, "actual"]) + 1e-8
            df_code.loc[valid_actuals, "wape_row"] = abs_error / abs_actual

            denom = np.abs(df_code.loc[valid_actuals, "actual"]).sum() + 1e-8
            df_code["wape_code"] = abs_error.sum() / denom

        return df_code
    

    def compute_evaluation_metrics(self, predictions_df: np.ndarray, real_data_df: np.ndarray, metrics_df: pd.DataFrame) -> Dict[str, float]:
        """
        Computes evaluation metrics (MAE, MSE, WAPE) between predictions and real data.

        Args:
            predictions_df (np.ndarray): The array of predicted values.
            real_data_df (np.ndarray): The array of actual values.
        Returns:
            A dictionary containing the computed evaluation metrics.
        """

        codes = np.unique(predictions_df['code'].values)
        dates = np.unique(predictions_df['target_date'].values)
        forecasts = np.unique(predictions_df['forecast'].values[0])
        results_accumulator = []
        for code in codes:
            for date in dates:
                for forecast in forecasts:
                    real_code = code.replace('#', ':')
                    preds_mask = (predictions_df['code'] == code) & (predictions_df['target_date'] == date) & (predictions_df['forecast'] == forecast)
                    preds = predictions_df.loc[preds_mask, 'predictions'].values
                    
                    real_data_df['timestamp'] = pd.to_datetime(real_data_df['timestamp'])
                    real_mask = real_data_df[real_code][real_data_df['timestamp'] == date].index
                    real = real_data_df.loc[real_mask, real_code].values

                    mae = np.mean(np.abs(preds - real))
                    mse = np.mean((preds - real)**2)
                    rmse = np.sqrt(mse)
                    real_lb_data = real_data_df[(real_data_df['timestamp'] < date) & (real_data_df['timestamp'] >= (pd.to_datetime(date) - pd.Timedelta(days= forecast)))]
                    wape = np.sum(np.abs(preds - real)) / np.sum(np.abs(real_lb_data[real_code]) + 1e-8)

                    print(f"Evaluation for code {code} on date {date}: MAE={mae}, MSE={mse}, WAPE={wape*100:.2f}%")
                    results_accumulator.append({
                        'code': code,
                        'date': date,
                        'forecast': forecast,
                        'MAE': mae,
                        'MSE': mse,
                        'RMSE': rmse,
                        'WAPE': wape *100   

                    })
        metrics_df = pd.concat([metrics_df, pd.DataFrame(results_accumulator)], ignore_index=True)
        output_path = self.config.production_metrics_file
        metrics_df.to_parquet(output_path, index=False)
        return metrics_df

        

    def delete_old_data(self, predictions_dataset_path: str, real_data_dataset_path: Optional[str] = None, metrics_df_path: str = "../production_predictions/production_evaluation_metrics.parquet", max_date: Optional[str] = None) -> Optional[str]:
        """
        Deletes rows from a parquet file where the day difference between
        target_date and forecast_date equals the value in forecast.

        Args:
            predictions_dataset_path (str): Full parquet dataset path created by save_final_output_predictions.
            real_data_dataset_path (Optional[str]): Optional path to the real data parquet file for additional processing.
            metrics_df_path (str): Path to the existing metrics dataframe. If it doesn't exist, a new one will be created.
            max_date (Optional[str]): The maximum date to consider for deletion.
        """
        if not os.path.exists(predictions_dataset_path):
            print(f"No data file found at: {predictions_dataset_path}")
            return None
        if not os.path.exists(real_data_dataset_path):
            print(f"No real data file found at: {real_data_dataset_path}")
        else:
            print(f"Real data file found at: {real_data_dataset_path}. It will be used for additional processing.")

        if os.path.exists(metrics_df_path):
            metrics_df = pd.read_parquet(metrics_df_path)
            print(f"Existing metrics dataframe loaded from: {metrics_df_path}")
        else:
            print(f"No existing metrics dataframe found at: {metrics_df_path}. A new one will be created.")
            metrics_df = pd.DataFrame(columns=['code', 'date', 'MAE', 'MSE', 'RMSE', 'WAPE'])
            
        
        real_data_dataset = smart_read(real_data_dataset_path) if real_data_dataset_path else None
        if max_date is not None:
            real_data_dataset = real_data_dataset[real_data_dataset['timestamp'] < pd.to_datetime(max_date)] if real_data_dataset is not None else None
        my_schema = pa.schema([("code", pa.string())])

        # 2. Create the partitioning object WITH the flavor
        partition_schema = ds.partitioning(my_schema)
        dataset = ds.dataset(predictions_dataset_path, format="parquet", partitioning=partition_schema)

        
        table = dataset.to_table()
        df = table.to_pandas()

        required_cols = {"target_date", "final_forecast_date", "forecast"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {predictions_dataset_path}: {sorted(missing)}"
            )

        # Ensure datetime + numeric types
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
        df["final_forecast_date"] = pd.to_datetime(df["final_forecast_date"], errors="coerce")
        forecast_days = pd.to_numeric(df["forecast"], errors="coerce")  # Add 1 to match the day difference logic (inclusive of start date)

        # Rows to delete:
        # difference in days equals forecast value in that row
        #diff_days = (df["target_date"] - df["final_forecast_date"]).dt.days
        # 1. Gather ALL unique forecast dates from the entire dataset
        all_forecast_dates = pd.concat(
            [df["init_forecast_date"], df["final_forecast_date"]]
        ).dropna().unique()

        # 2. Condition A: Does the target_date match ANY forecast date in the dataset?
        matches_forecast = df["target_date"].isin(all_forecast_dates)

        # 4. Combine both conditions using a bitwise AND (&)
        delete_mask = matches_forecast 

        deleted_rows = int(delete_mask.sum())
        df_to_delete = df.loc[delete_mask].copy()

        if real_data_dataset is not None and not df_to_delete.empty:
            self.compute_evaluation_metrics(predictions_df=df_to_delete, real_data_df=real_data_dataset, metrics_df=metrics_df)
        else:
            print("No rows to delete or real data dataset not available. Skipping evaluation metrics computation.")

        df_clean = df.loc[~delete_mask].copy()

        if os.path.isdir(predictions_dataset_path):
            # 2. Delete the directory to clear old partitions
            shutil.rmtree(predictions_dataset_path)

        table = pa.Table.from_pandas(df_clean)
        write_partitioning = ds.partitioning(pa.schema([("code", pa.string())]))
        ds.write_dataset(
            table, 
            base_dir=predictions_dataset_path, 
            format="parquet", 
            partitioning=write_partitioning,
            existing_data_behavior="overwrite_or_ignore"
        )

        '''df_clean.to_parquet(
            predictions_dataset_path, 
            engine='pyarrow', 
            partition_cols=['code'], 
            index=False
        )'''
        print(f"Deleted {deleted_rows} rows from: {predictions_dataset_path}")
        print(f"Remaining rows: {len(df_clean)}")

        return predictions_dataset_path
    

if __name__ == "__main__":
    CODES_LIST = ["DEMAND_demanda__TOTAL_UP_00185", "DEMAND_demanda__TOTAL_RS_CATALUNYA CENTRAL"] 
                 #["demanda__SERVEI_CODI__INF",
                 #   "demanda__SERVEI_CODI__INFP",
                 #   "demanda__SERVEI_CODI__MF",
                 #   "demanda__SERVEI_CODI__PED",
                 #   "demanda__SERVEI_CODI__URG",
                 #   "demanda__TIPUS_CLASS__9T",
                 #   "demanda__TIPUS_CLASS__C9C",
                 #   "demanda__TIPUS_CLASS__C9R",
                 #   ]
    
                    #['demanda__TOTAL', 'demanda__SERVEI_CODI__URG', 'B34','J00', 'I10', 'M54','Ch01#subch01#A00-A09']
    LOOKBACK_LIST = [7, 14, 60, 60, 182,182]
    FORECAST_LIST = [7, 14, 30, 60, 182,365]
    FINAL_LOOKBACK = 182
    FINAL_FORECAST = 365

    #input_directory = '../data/FINAL_DB/full_CAT1.parquet'
    input_directory = '../data/FINAL_DB/demand_diagnosis_joined.parquet' #'../data/FINAL_DB/finals_combined.csv'
    #model_folder = '../transformer_outputs/models_covid_token'
    output_path = f"../production_predictions/final_output_predictions"
    metrics_df_path = "../production_predictions/production_evaluation_metrics.parquet"
    scaler = FunctionTransformer(func=lambda x: x, inverse_func=lambda x: x)
    max_date = '2026-01-01'
    cutoff_date = '2008-01-01'
    eliminate_covid_data = False
    covid_dates = None
    model_folder = "../quantized_models"
    head_size = 32
    num_heads = 8
    ff_dim = 512
    num_transformer_blocks = 2
    mlp_units = [512,256]
    activation_function = "gelu"


    final_output_df = pd.DataFrame()
    for code in CODES_LIST:
        final_output_predictions = None
        
        #final_output_df = generate_future_dates_df(input_directory, num_days=FORECAST_LIST[-1])
        
        base_pipeline = ModelPredictionPipeline(config=BaseTransformerConfig(
                code=code,
                head_size=head_size,
                num_heads=num_heads,
                ff_dim=ff_dim,
                num_transformer_blocks=num_transformer_blocks,
                mlp_units=mlp_units,
                activation_function=activation_function,
                dropout=0,
                learning_rate=0.001,
                epochs=50,
                batch_size=32,
                cutoff_date=cutoff_date,
                covid_token=True,
                positional_encoding=True,
                evaluate_model=False, 
                data_path=input_directory, 
                model_folder=model_folder,
            ))
        final_output_df = base_pipeline.run_reconstruct_save_results_pipeline(input_directory,code, LOOKBACK_LIST, FORECAST_LIST, final_output_predictions, final_output_df)

        base_pipeline.save_final_output_predictions(final_output_df)
    #base_pipeline.delete_old_data(predictions_dataset_path=output_path, real_data_dataset_path=input_directory, metrics_df_path=metrics_df_path)
    print(f"\nFinal output predictions for code {code}:\n")
    #print(final_output_df)