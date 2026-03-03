from sklearn.pipeline import FunctionTransformer
import numpy as np
import pandas as pd
from sqlalchemy import between
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any
import pyarrow as pa
import pyarrow.dataset as ds

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.univariate_transformer import model_architecture_univ_transformer
from src.residual_multivariate_transformers import model_architecture_residual_transformer
from utils import data_preparation
from config.base_transformer_config import BaseTransformerConfig
from production.data_preparation_in_poduction import DataPreparationInProduction

default_config = BaseTransformerConfig()


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
        print(model.summary())
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

    
    def run_reconstruct_save_results_pipeline(self, code: str, LOOKBACK_LIST: List[int], FORECAST_LIST: List[int], final_output_predictions: Optional[np.ndarray], final_output_df: pd.DataFrame):
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
        for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST):
            auxiliary_output_df = pd.DataFrame()  # Temporary DataFrame for current iteration
            X_univ_data, Y_univ_data, df_timestamp = self.prepare_prediction_univ_data(
                data_path=input_directory,
                code=code,
                lookback=lookback,
                forecast=forecast,
                cutoff_date=cutoff_date,
                max_date=max_date,
                scaler=scaler,
                eliminate_covid_data=eliminate_covid_data,
                covid_token=self.config.covid_token,
                production_mode=False
            )

            X_diagnostics_data, Y_diagnostics_data = self.prepare_prediction_diagnostics_data(
                data_path=input_directory,
                code=code,
                lookback=lookback,
                forecast=forecast,
                cutoff_date =cutoff_date,
                max_date=max_date,
                scaler=scaler,
                covid_token=self.config.covid_token,
                production_mode=False
            )

            X_seasonal_data = self.prepare_prediction_seasonal_data(
                data_path=input_directory,
                code=code,
                forecast=forecast,
                lookback=lookback,
                cutoff_date=cutoff_date,
                max_date=max_date,
                categorical_vars=self.config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
                predictions_train=None,
                predictions_test=None,
                scaler=scaler,
            )
            
            univ_input_shape = X_univ_data.shape[1:]
            diagnostics_input_shape = X_diagnostics_data.shape[1:]
            seasonal_input_shape = X_seasonal_data.shape
            seasonal_input_shape = tuple((seasonal_input_shape[1], seasonal_input_shape[2]+1))

            
            univ_model, diagnostics_model, seasonal_model = self.reconstruct_full_model(
                code,
                lookback,
                forecast,
                models_directory="../quantized_models",
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

            #
            quant_predictions_univ = univ_model.predict(X_univ_data, verbose=0)
            quant_predictions_diagnostics_residuals = diagnostics_model.predict(X_diagnostics_data, verbose=0)
            
            preds_reshaped = quant_predictions_diagnostics_residuals[:, -forecast:, np.newaxis]
            X_seasonal_with_preds = np.concatenate((X_seasonal_data, preds_reshaped), axis=2)
            quant_predictions_seasonal_residuals = seasonal_model.predict(X_seasonal_with_preds, verbose=0)


            quant_pred_corrected_diagnostics = quant_predictions_univ + quant_predictions_diagnostics_residuals
            quant_pred_corrected_seasonal = quant_pred_corrected_diagnostics + quant_predictions_seasonal_residuals

            mae = np.mean(np.abs(quant_pred_corrected_seasonal - Y_univ_data))
            mse = np.mean((quant_pred_corrected_seasonal - Y_univ_data)**2)
            wape = np.mean(np.abs(quant_pred_corrected_seasonal - Y_univ_data) / (np.abs(Y_univ_data) + 1e-8))
            print(f"MAE for code {code} with lookback {lookback} and forecast {forecast}: {mae}")
            print(f"MSE for code {code} with lookback {lookback} and forecast {forecast}: {mse}")
            print(f"WAPE for code {code} with lookback {lookback} and forecast {forecast}: {wape*100:.2f}%")

            target_date = df_timestamp[-forecast:].values
            forecast_date = df_timestamp.iloc[-forecast-1]

            auxiliary_output_df["target_date"] = target_date
            auxiliary_output_df["forecast_date"] = forecast_date
            auxiliary_output_df["code"] = code
            auxiliary_output_df["forecast"] = forecast
            auxiliary_output_df["predictions"] = quant_pred_corrected_seasonal.flatten()[:forecast]

            final_output_df = pd.concat([final_output_df, auxiliary_output_df], ignore_index=True)
            
        #final_output_df.to_csv(f"../quantized_models/{code}/final_output_predictions_{code.replace(':', '#')}.csv", index=False)

        return final_output_df

    def save_final_output_predictions(self, final_output_df: pd.DataFrame, output_path: str = "../production_predictions/final_output_predictions"):
        """
        Saves the final output predictions DataFrame to a CSV file.

        Args:
            final_output_df (pd.DataFrame): The DataFrame containing the final output predictions along with corresponding dates.
            code (str): The code for which the predictions were made, used for naming the output file.
        """
        
        table = pa.Table.from_pandas(final_output_df, preserve_index=False)

        ds.write_dataset(
            table,
            base_dir=output_path,
            format="parquet",
            partitioning=["code"]
        )

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

                    real_mask = real_data_df[real_code][real_data_df['timestamp'] == date].index
                    real = real_data_df.loc[real_mask, real_code].values

                    mae = np.mean(np.abs(preds - real))
                    mse = np.mean((preds - real)**2)
                    rmse = np.sqrt(mse)
                    wape = np.mean(np.abs(preds - real) / (np.abs(real) + 1e-8))

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
        metrics_df.to_parquet("../production_predictions/production_evaluation_metrics.parquet", index=False)
        return metrics_df

        

    def delete_old_data(self, predictions_dataset_path: str, real_data_dataset_path: Optional[str] = None, metrics_df_path: str = "../production_predictions/production_evaluation_metrics.csv"):
        """
        Deletes rows from a parquet file where the day difference between
        target_date and forecast_date equals the value in forecast.

        Args:
            predictions_dataset_path (str): Full parquet dataset path created by save_final_output_predictions.
            real_data_dataset_path (Optional[str]): Optional path to the real data parquet file for additional processing.
            metrics_df_path (str): Path to the existing metrics dataframe. If it doesn't exist, a new one will be created.
        
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
            
        
        real_data_dataset = pd.read_parquet(real_data_dataset_path) if real_data_dataset_path else None

        my_schema = pa.schema([("code", pa.string())])

        # 2. Create the partitioning object WITH the flavor
        partition_schema = ds.partitioning(my_schema)
        dataset = ds.dataset(predictions_dataset_path, format="parquet", partitioning=partition_schema)

        
        table = dataset.to_table()
        df = table.to_pandas()

        required_cols = {"target_date", "forecast_date", "forecast"}
        missing = required_cols - set(df.columns)
        if missing:
            raise ValueError(
                f"Missing required columns in {predictions_dataset_path}: {sorted(missing)}"
            )

        # Ensure datetime + numeric types
        df["target_date"] = pd.to_datetime(df["target_date"], errors="coerce")
        df["forecast_date"] = pd.to_datetime(df["forecast_date"], errors="coerce")
        forecast_days = pd.to_numeric(df["forecast"], errors="coerce")  # Add 1 to match the day difference logic (inclusive of start date)

        # Rows to delete:
        # difference in days equals forecast value in that row
        diff_days = (df["target_date"] - df["forecast_date"]).dt.days
        delete_mask = diff_days == forecast_days

        deleted_rows = int(delete_mask.sum())
        df_to_delete = df.loc[delete_mask].copy()

        self.compute_evaluation_metrics(predictions_df=df_to_delete, real_data_df=real_data_dataset, metrics_df=metrics_df)
        df_clean = df.loc[~delete_mask].copy()

        df_clean.to_parquet(
            predictions_dataset_path, 
            engine='pyarrow', 
            partition_cols=['code'], 
            index=False
        )
        print(f"Deleted {deleted_rows} rows from: {predictions_dataset_path}")
        print(f"Remaining rows: {len(df_clean)}")

        return predictions_dataset_path


if __name__ == "__main__":
    CODES_LIST = ['demanda__TOTAL', 'demanda__SERVEI_CODI__URG', 'B34','J00', 'I10', 'M54','Ch01#subch01#A00-A09']
    LOOKBACK_LIST = [7, 14, 60, 60, 182,182]
    FORECAST_LIST = [7, 14, 30, 60, 182,365]
    FINAL_LOOKBACK = 182
    FINAL_FORECAST = 365

    input_directory = '../data/FINAL_DB/full_CAT1.parquet'
    models_directory = '../transformer_outputs/models_covid_token'
    output_path = f"../production_predictions/final_output_predictions"
    metrics_df_path = "../production_predictions/production_evaluation_metrics.parquet"
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
                evaluate_model=True, 
                data_path=input_directory
            ))
        #final_output_df = base_pipeline.run_reconstruct_save_results_pipeline(code, LOOKBACK_LIST, FORECAST_LIST, final_output_predictions, final_output_df)
    #base_pipeline.save_final_output_predictions(final_output_df)
    base_pipeline.delete_old_data(predictions_dataset_path=output_path, real_data_dataset_path=input_directory, metrics_df_path=metrics_df_path)
    print(f"\nFinal output predictions for code {code}:\n")
    print(final_output_df)