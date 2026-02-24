from sklearn.pipeline import FunctionTransformer
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os
import sys
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.univariate_transformer import model_architecture_univ_transformer
from src.residual_multivariate_transformers import model_architecture_residual_transformer
import data_preparation
from config.base_transformer_config import BaseTransformerConfig

default_config = BaseTransformerConfig()




@dataclass
class TransformerPredictionConfig(BaseTransformerConfig):
    """Configuration class for transformer training parameters"""
    
    # Configuration object reference
    config_object: Optional[Any] = field(default=None)
    categorical_vars: List[str] = field(default_factory=lambda: ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation",])
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "lookback": self.lookback,
            "forecast": self.forecast,
            "code": self.code,
            "head_size": self.head_size,
            "num_heads": self.num_heads,
            "ff_dim": self.ff_dim,
            "num_transformer_blocks": self.num_transformer_blocks,
            "mlp_units": self.mlp_units,
            "dropout": self.dropout,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "cutoff_date": self.cutoff_date,
            "covid_token": self.covid_token,
            "positional_encoding": self.positional_encoding,
            "activation_function": self.activation_function,
            "evaluate_model": self.evaluate_model, 
            "data_path": self.data_path,
        }
    
    def print_config(self):
        """Print configuration in a readable format"""
        print("\n" + "="*60)
        print("TRANSFORMER TRAINING CONFIGURATION")
        print("="*60)
        for key, value in self.to_dict().items():
            print(f"{key:25}: {value}")
        print("="*60)
    

    

class ModelPredictionPipeline:
    def __init__(self, config: TransformerPredictionConfig):
        self.config = config
        self.config.print_config()

    def load_diagnostic_covariates(self,diagnostic_covariates_path,code, forecast):
        diagnostic_covariates_df = pd.read_excel(diagnostic_covariates_path + code + ".xlsx", engine='openpyxl')
        diagnostic_covariates_list = list(diagnostic_covariates_df[diagnostic_covariates_df['LAG'] ==  forecast]['predictors'])[0].split(',')
    
        return diagnostic_covariates_list


    def prepare_prediction_univ_data(self, data_path: str, code: str, lookback: int, forecast: int, cutoff_date: str, max_date: str, scaler: FunctionTransformer, eliminate_covid_data: bool, covid_token: bool, production_mode: bool = False):
        code = code.replace("#", ":")
        # Load CSV
        df = pd.read_parquet(data_path)
        if eliminate_covid_data:
            assert covid_dates is not None
            df = data_preparation.eliminate_covid_dates(df, covid_dates)
        
        df = data_preparation.cut_dataframe(df, cutoff_date,max_date, data_path)

        categorical_vars = ["Day_of_Week", "Month", "Season", "Holiday", "School_Vacation","Is_Weekend"]
        df_dates = data_preparation.prepare_time_series_features(df, categorical_vars=categorical_vars, cutoff_date=cutoff_date, max_date=max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
        df_timestamp = df['timestamp']
        df_dates = df_dates.drop(columns=['timestamp']) 
         
        # univariate scenario ...................................................
        idx_code = df.columns.get_loc(code)
        feature_cols = df.columns[idx_code]  
        target_col = df.columns[idx_code]  # Get the target column 
        
        # Convert to numpy arrays
        X_raw = df[feature_cols].values.reshape(-1, 1)  
        X_raw = np.hstack((X_raw, df_dates.values.astype(np.float32)))
        Y_raw = df[target_col].values # Target values
        
        if covid_token:
            df_covid = data_preparation.add_covid_token(df)
            covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
            X_raw = np.hstack((X_raw, covid_feature))

        if production_mode:
            X_raw = X_raw[-lookback:].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
            Y_raw = np.zeros_like(Y_raw)  # Replace with zeros for production mode as we don't have true future values
            
            # Generate future timestamps for the forecast horizon and append to df_timestamp
            last_date = df_timestamp.iloc[-1]
            new_dates = pd.date_range(
                start=last_date + pd.Timedelta(days=1), 
                periods=forecast, 
                freq='D'
            )
            df_forecast = pd.Series(new_dates)
            #df_timestamp = df_timestamp.iloc[-lookback:].reset_index(drop=True)
            df_timestamp = pd.concat([df_timestamp, df_forecast], ignore_index=True)
            

        else:
            X_raw = X_raw[-(lookback + forecast):-forecast].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
            #df_timestamp = df_timestamp#.iloc[-(lookback + forecast):].reset_index(drop=True)
        return X_raw, Y_raw, df_timestamp


    def prepare_prediction_diagnostics_data(self, data_path: str, code: str, lookback: int, forecast: int, max_date: str, scaler: FunctionTransformer, covid_token: bool = False, production_mode: bool = False): 
        
        relevant_feature_cols = self.load_diagnostic_covariates(default_config.diagnostic_covariates_path, code, forecast)
        code = code.replace("#", ":")
        # Load CSV
        df = pd.read_parquet(data_path)
        if eliminate_covid_data:
            assert covid_dates is not None
            df = data_preparation.eliminate_covid_dates(df, covid_dates)
        df = data_preparation.cut_dataframe(df, cutoff_date,max_date, data_path)

        categorical_vars = ["Day_of_Week", 
                                "Month", 
                                "Season", 
                                "Holiday", 
                                "School_Vacation",
                                "Is_Weekend",
                                ]
        df_dates = data_preparation.prepare_time_series_features(df, categorical_vars=categorical_vars, cutoff_date=cutoff_date, max_date=max_date, scaler=scaler, eliminate_covid_data=eliminate_covid_data, covid_dates=covid_dates)
        df_dates = df_dates.drop(columns=['timestamp']) 
        
        
        # multivariate scenario ..................................................
        # Select feature columns (exclude timestamp & target)
        idx_code = df.columns.get_loc(code)
        df_features = df.drop(columns = ['timestamp'])
        target_col = df.columns[idx_code]  # Target code column
        # Convert DataFrame to numpy arrays
        if relevant_feature_cols is not None:
            X_raw = df_features[relevant_feature_cols].values  
            X_raw = np.hstack((X_raw, df_dates.values.astype(np.float32)))
        else:
            X_raw = df_features.values

        Y_raw = df[target_col].values
        if covid_token:
            df_covid = data_preparation.add_covid_token(df)
            covid_feature = df_covid['covid_token'].values.reshape(-1, 1)
            
            X_raw = np.hstack((X_raw, covid_feature))

        if production_mode:
            X_raw = X_raw[-lookback:].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
            Y_raw = np.zeros_like(Y_raw)  # Replace with zeros for production mode as we don't have true future values
        else:
            X_raw = X_raw[-(lookback + forecast):-forecast].reshape(1, lookback, -1)
            Y_raw = Y_raw[-forecast:].reshape(1, forecast)
        return X_raw, Y_raw
    

    def prepare_prediction_seasonal_data(self, data_path: str, code: str, forecast: int, lookback: int, cutoff_date: str, max_date: str, categorical_vars: List[str], predictions_train: Optional[np.ndarray], predictions_test: Optional[np.ndarray], scaler: FunctionTransformer):
        df = pd.read_parquet(data_path)
        # Prepare seasonal features for training data

        print("Preparing seasonal features for training data...")
        df_processed = data_preparation.prepare_time_series_features(
            df, 
            self.config.categorical_vars, 
            cutoff_date=self.config.cutoff_date,
            max_date = self.config.final_cutoff_date,
            scaler = self.config.scaler,
            eliminate_covid_data=self.config.eliminate_covid_data, 
            covid_dates=self.config.covid_dates,)

        X_seasonal_covs = df_processed.drop(columns=['timestamp']).values[-forecast:].reshape(1, -1, df_processed.shape[1]-1)
        X_seasonal_covs = X_seasonal_covs.astype(float)
        return X_seasonal_covs
    
    def create_univariate_transformer_model(self, input_shape, forecast, head_size, num_heads, ff_dim, num_transformer_blocks, mlp_units, activation_function="tanh", dropout=0, mlp_dropout=0, n_pred=1, pos_encoding=True):
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

        for lookback, forecast in zip(LOOKBACK_LIST, FORECAST_LIST):

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
                categorical_vars=None,
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

            if final_output_predictions is None:
                    final_output_predictions = np.zeros(FORECAST_LIST[-1])

                
            if FORECAST_LIST.index(forecast) == 0:
                final_output_predictions[:forecast] += quant_pred_corrected_seasonal[:forecast][0]
                vals = final_output_predictions[ :forecast]
                padded_vals = np.full(365, np.nan)
                padded_vals[:forecast] = vals
                
                final_output_df[forecast] = padded_vals

            else:
                final_output_predictions[FORECAST_LIST[FORECAST_LIST.index(forecast)-1]:forecast] += quant_pred_corrected_seasonal[0][FORECAST_LIST[FORECAST_LIST.index(forecast)-1]:forecast]
                vals = final_output_predictions[:FORECAST_LIST[FORECAST_LIST.index(forecast)]]
                padded_vals = np.full(365, np.nan)
                padded_vals[:FORECAST_LIST[FORECAST_LIST.index(forecast)]] = vals
                final_output_df[forecast] = padded_vals
            
        final_output_df['date'] = df_timestamp[-FORECAST_LIST[-1]:].values
        final_output_df.to_csv(f"../quantized_models/{code}/final_output_predictions_{code.replace(':', '#')}.csv", index=False)

        return final_output_df

if __name__ == "__main__":
    CODES_LIST = ['demanda__TOTAL', 'demanda__SERVEI_CODI__URG', 'B34','J00', 'I10', 'M54','Ch01#subch01#A00-A09']
    LOOKBACK_LIST = [7, 14, 60, 60, 182,182]
    FORECAST_LIST = [7, 14, 30, 60, 182,365]
    FINAL_LOOKBACK = 182
    FINAL_FORECAST = 365

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
        final_output_predictions = None
        final_output_df = pd.DataFrame()
        #final_output_df = generate_future_dates_df(input_directory, num_days=FORECAST_LIST[-1])
        
        base_pipeline = ModelPredictionPipeline(config=TransformerPredictionConfig(
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
        base_pipeline.run_reconstruct_save_results_pipeline(code, LOOKBACK_LIST, FORECAST_LIST, final_output_predictions, final_output_df)
        
    print(f"\nFinal output predictions for code {code}:\n")
    print(final_output_df)