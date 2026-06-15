
import hydra
import numpy as np
import pandas as pd
import tensorflow as tf
import os
import json
from datetime import datetime
from omegaconf import DictConfig, OmegaConf
import mlflow
import mlflow.tensorflow
import mlflow.keras
import gc
from tf_keras import backend as K

from sklearn.metrics import mean_absolute_error, mean_squared_error

from CCLR_PREDAP.main import CCLR_pipeline
from production.model_quantization_pipeline import ModelQuantizationPipeline
from src.config.config_manager import get_config
from src.model_architechture.model_architecture_univ_transformer import build_model
from src.data_utils import data_preparation
from src.data_utils.residual_data_preparation import prepare_residual_data, generate_rolling_sequences_covariates
from training.training_univ_transformer import train_univariate_model
from training.training_utils import compile_model, load_base_model_transformer, load_diagnostic_covariates, setup_callbacks

from model_architechture.model_architecture_residual_transformer import build_residual_transformer_model
from utils.environment_utils import setup_gpu_memory
from utils.experiments_utils import compute_dynamic_batch_size, get_codes_list, check_for_help_flag , memory_cleanup, save_model_parameters, smart_read
from training.training_residual_transformer import train_residual_model
from data_utils.features import prepare_time_series_features
from evaluation.evaluate_transformer import evaluate_transformer_models, save_performance_results, save_univ_performance_results
from utils.mlflow_logger import load_mlflow_model_history
from visualization_func.visualization_transformer import plt_model, plot_residuals_analysis



def setup_mlflow_tracking(cnfg,config, run_name=None):
    mlflow.set_tracking_uri(cnfg.mlflow.tracking_uri)
    experiment_name = f"{cnfg.mlflow.experiment_name}_{datetime.now().strftime('%Y%m%d')}"
    mlflow.set_experiment(experiment_name)
    print(f"🎯 MLflow tracking initialized")
    print(f"   • Experiment: {experiment_name}")
    print(f"   • Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"   • View results at: http://localhost:5000")

    run_name = f"1.0_Production_TRANSFORMER_{config.code}_lb{config.lookback}_fh{config.forecast}_{datetime.now().strftime('%H%M%S')}"
    
    return run_name

def log_initial_mlflow_params(config, cnfg):

    mlflow.log_params({
        "target_code": config.code,
        "lookback": config.lookback,
        "forecast_horizon": config.forecast,
        "model_type": "transformer",
        "dataset": config.data_path,
        "activation_function": config.activation_function,
        "covid_token": config.covid_token,
        "cutoff_date": config.cutoff_date,
        "head_size": config.head_size,
        "num_heads": config.num_heads, 
        "ff_dim": config.ff_dim,
        "mlp_units": config.mlp_units,
        "num_transformer_blocks": config.num_transformer_blocks,
        "dropout": config.dropout,
        "learning_rate": config.learning_rate ,
        "positional_encoding": config.positional_encoding,
        "scaler": config.scaler,
    })
    
    # Log system information
    mlflow.log_params({
        "tensorflow_version": tf.__version__,
        "python_version": os.sys.version.split()[0],
        "gpu_available": len(tf.config.list_physical_devices('GPU')) > 0,
        "backend": cnfg.system.backend,
        "hydra_config_name": cnfg._target_ if hasattr(cnfg, '_target_') else "config"
    })


def univ_model_training_pipeline(config,X_train_univ, Y_train_univ, X_test_univ, Y_test_univ, base_model_name, model_params, training_parameters):

    print("\n" + "="*50)
    print("MODEL BUILDING PHASE")
    print("="*50)

    input_univ_shape = X_train_univ.shape[1:]  # (lookback, num_features) # config.lookback, X.shape[-1]
    univ_model = build_model(
        input_univ_shape,
        **model_params
    )
        
    #Compile models and setup callbacks for training
    print("Model Univariate Architecture:")
    univ_model.summary()
    univ_model = compile_model(config,univ_model)
    callbacks = setup_callbacks(config,
                                config.learning_rate,
                                config.early_stop_patience)

    # Train models
    training_univ_history = train_univariate_model(
        model=univ_model,
        X=X_train_univ,
        Y=Y_train_univ,
        **training_parameters,
        model_name=base_model_name,
        callbacks=callbacks,
    )

    #Phase 2.1: Predict with base model 
    train_univ_predictions = univ_model.predict(X_train_univ)
    test_univ_predictions = univ_model.predict(X_test_univ)

    loss_univ, mae_univ, mse_univ = univ_model.evaluate(X_test_univ, Y_test_univ, verbose=0, batch_size=config.batch_size)

    del training_univ_history
    
    return univ_model, train_univ_predictions, test_univ_predictions, loss_univ, mae_univ, mse_univ

def diagnostics_model_training_pipeline(config, 
                                        X_train_diagnostics, Y_train_diagnostics, 
                                        X_test_diagnostics,Y_test_diagnostics, 
                                        train_univ_predictions, test_univ_predictions, 
                                        Y_train_univ, Y_test_univ, 
                                        diagnostics_model_name, 
                                        residual_model_params, training_parameters):
    
    input_shape_diagnostics = X_train_diagnostics.shape[1:]
    diagnostics_model = build_residual_transformer_model(
        input_shape_diagnostics,
        **residual_model_params
    )

    print("Model Diagnostic Architecture:")
    diagnostics_model.summary()
    diagnostics_model = compile_model(config,diagnostics_model)
    callbacks_diagnostics = setup_callbacks(config,
                                config.learning_rate,
                                config.early_stop_patience)

    Y_train_residuals_diagnostics = prepare_residual_data(
        train_univ_predictions,
        Y_train_univ,
    
    )

    Y_test_residuals_diagnostics = prepare_residual_data(
        test_univ_predictions,
        Y_test_univ
    )


    training_diagnostics_history = train_residual_model(
        model=diagnostics_model,
        X=X_train_diagnostics,
        Y=Y_train_residuals_diagnostics,
        **training_parameters,
        model_name=diagnostics_model_name,
        callbacks=callbacks_diagnostics,
    )

    # Predict with diagnostic models
    train_residual_diagnostics_predictions = diagnostics_model.predict(X_train_diagnostics)
    test__residual_diagnostics_predictions = diagnostics_model.predict(X_test_diagnostics)
    loss_diagnostics, mae_diagnostics, mse_diagnostics = diagnostics_model.evaluate(X_test_diagnostics, Y_test_diagnostics, verbose=0, batch_size=config.batch_size)

    train_corrected_diagnostics_predictions = train_univ_predictions + train_residual_diagnostics_predictions
    test_corrected_diagnostics_predictions = test_univ_predictions + test__residual_diagnostics_predictions


    del training_diagnostics_history

    return diagnostics_model, train_corrected_diagnostics_predictions, test_corrected_diagnostics_predictions, loss_diagnostics, mae_diagnostics, mse_diagnostics

def seasonal_model_training_pipeline(config, 
                                     X_train_seasonal, Y_train_residuals_diagnostics, 
                                     X_test_seasonal, Y_test_residuals_diagnostics, 
                                     train_corrected_diagnostics_predictions, test_corrected_diagnostics_predictions, 
                                     seasonal_model_name,
                                     residual_model_params, training_parameters
                                     ):
    
    input_shape_seasonal = X_train_seasonal.shape[1:]
    
    seasonal_model = build_residual_transformer_model(
        input_shape_seasonal,
        **residual_model_params
    )

    print("Model Seasonal Architecture:")
    seasonal_model.summary()
    seasonal_model = compile_model(config,seasonal_model)
    callbacks_seasonal = setup_callbacks(config,
                                config.learning_rate,
                                config.early_stop_patience)

    training_seasonal_history = train_residual_model(
        model=seasonal_model,
        X=X_train_seasonal,
        Y=Y_train_residuals_diagnostics,
        **training_parameters,
        model_name=seasonal_model_name,
        callbacks=callbacks_seasonal,
    )

    # Predict with seasonal model
    train_seasonal_predictions = seasonal_model.predict(X_train_seasonal)
    test_seasonal_predictions = seasonal_model.predict(X_test_seasonal)
    loss_seasonal, mae_seasonal, mse_seasonal = seasonal_model.evaluate(X_test_seasonal, Y_test_residuals_diagnostics, verbose=0, batch_size=config.batch_size)

    train_final_predictions = train_corrected_diagnostics_predictions + train_seasonal_predictions
    test_final_predictions = test_corrected_diagnostics_predictions + test_seasonal_predictions

    del training_seasonal_history

    return seasonal_model, train_final_predictions, test_final_predictions, loss_seasonal, mae_seasonal, mse_seasonal


config = get_config()
# Register custom resolver for loading JSON codes list
OmegaConf.register_new_resolver("get_codes_list", get_codes_list)
config_name = "config_production_quantization.yaml"  

@hydra.main(version_base=None, config_path="conf", config_name=config_name)
def main(cfg: DictConfig) -> None:
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

    base_model_name = config.get_model_name()
    diagnostics_model_name = config.get_diagnostic_residual_model_name()
    seasonal_model_name = config.get_seasonal_residual_model_name()
    diagnostic_covariates_list = load_diagnostic_covariates(config,config.code, config.forecast)

    # =================== PHASE 0.1: DEFINE PARAMETERS ===================
    data_parameters = {
        "code": config.code,
        "lookback": config.lookback,
        "forecast": config.forecast,
        "cutoff_date": config.cutoff_date,
        "max_date": config.max_date,
        "covid_token": config.covid_token,
        "scaler": config.scaler,
        "eliminate_covid_data": config.eliminate_covid_data,
        "covid_dates": config.covid_dates
    }

    model_params = {
        "head_size": config.head_size,
        "num_heads": config.num_heads,
        "ff_dim": config.ff_dim,
        "num_transformer_blocks": config.num_transformer_blocks,
        "mlp_units": config.mlp_units,
        "mlp_dropout": config.dropout,
        "dropout": config.dropout,
        "activation_function": config.activation_function,
        "n_pred": config.forecast,
    }

    residual_model_params = {
        "forecast": config.forecast,
        "lstm_params": config.DEFAULT_RESIDUAL_LSTM_PARAMS,
        "transformer_params": config.DEFAULT_RESIDUAL_TRANSFORMER_PARAMS,
        "activation_function": config.activation_function,
    }

    training_parameters = {
        "batch_size": config.batch_size,
        "epochs": config.epochs,
        "save_history": config.DEFAULT_RESIDUAL_SAVE_PARAMS["save_history"],
        "save_model": config.DEFAULT_RESIDUAL_SAVE_PARAMS["save_model"],
        "save_memory": config.DEFAULT_RESIDUAL_SAVE_PARAMS["save_memory"],
        "shuffle": config.DEFAULT_RESIDUAL_TRAINING_PARAMS["shuffle"],
        "validation_split": config.DEFAULT_RESIDUAL_TRAINING_PARAMS["validation_split"],
    }


    # Check if best code file exists, if not compute CCLR pipeline to generate it
    diagnostic_covariates_path = config.diagnostic_covariates_path
    best_code_name = diagnostic_covariates_path + config.code + '.xlsx'
    best_code_folder = '../data/best_features/'+ best_code_name
    if not os.path.exists(best_code_name):
        print(f"⚠️  Warning: Best code file not found at {best_code_folder}. Computing CCLR pipeline.")
        CCLR_pipeline(
            data_paths=[config.data_path],
            CODES_LIST=[config.code],
            BEST_FEATURES_PATH = config.diagnostic_covariates_path,
        )

    run_name = setup_mlflow_tracking(cfg,config)

    with mlflow.start_run(run_name=run_name) as run:
        print(f"\n🚀 Starting MLflow run: {run_name}")
        print(f"   • Run ID: {run.info.run_id}")
        log_initial_mlflow_params(config, cfg)
        
        # Log all hyperparameters from config
        # =================== PHASE 1: DATA PREPARATION ===================
        run_quantization_pipeline = ModelQuantizationPipeline(config=config)
        df= smart_read(config.data_path)
        
        df_seasonal = prepare_time_series_features(
            df = df,
            categorical_vars = config.DEFAULT_SEASONAL_CATEGORICAL_VARS,
            cutoff_date = config.cutoff_date,
            max_date = config.max_date,
            scaler = config.scaler,
            eliminate_covid_data = config.eliminate_covid_data, 
            covid_dates=config.covid_dates,
        )

        df_train, df_test = data_preparation.split_train_test(df, config.default_split_ratio)
        df_train_seasonal, df_test_seasonal = data_preparation.split_train_test(df_seasonal, config.default_split_ratio)



        X_train_univ, Y_train_univ = data_preparation.prepare_univariate_data(
            df = df_train,
            **data_parameters   
        )
        X_test_univ, Y_test_univ = data_preparation.prepare_univariate_data(
            df = df_test,
            **data_parameters
        )

        X_train_diagnostics, Y_train_diagnostics = data_preparation.prepare_multivariate_data(
            df = df_train,
            **data_parameters,
            relevant_feature_cols=diagnostic_covariates_list
        )
        X_test_diagnostics, Y_test_diagnostics = data_preparation.prepare_multivariate_data(
            df = df_test,
            **data_parameters,
            relevant_feature_cols=diagnostic_covariates_list
        )

        date_list = data_preparation.extract_dates(config.data_path, 
                                                    config.code, 
                                                    config.lookback,
                                                    config.forecast, 
                                                    train=False, 
                                                    cutoff_date=config.cutoff_date, 
                                                    max_date = config.max_date, 
                                                    eliminate_covid_data=config.eliminate_covid_data, 
                                                    covid_dates=config.covid_dates)



        # =================== PHASE 2: TRAIN BASE MODEL AND GET PREDICTIONS ===================
        univ_start_time = datetime.now()
        
        univ_model, train_univ_predictions, test_univ_predictions, loss_univ, mae_univ, mse_univ = univ_model_training_pipeline(config, 
                                                                                                                    X_train_univ, 
                                                                                                                    Y_train_univ, 
                                                                                                                    X_test_univ, 
                                                                                                                    Y_test_univ, 
                                                                                                                    base_model_name, 
                                                                                                                    model_params, 
                                                                                                                    training_parameters
                                                                                                                    )

        univ_duration = float((datetime.now() - univ_start_time).total_seconds())
        mlflow.log_metrics({"duration/phase_1_univariate_transformer_duration_seconds": univ_duration,})
        mlflow.keras.log_model(univ_model, artifact_path="univariate_model")
        load_mlflow_model_history(base_model_name, model_type="univariate_transformer")
        save_model_parameters(config, univ_model, base_model_name, config.model_parameters_path)

        #Quantize univariate model
        quant_univ_model = run_quantization_pipeline.manual_weight_quantization(univ_model, model_name="univariate_model")
        run_quantization_pipeline.save_quantized_model_weights(quant_univ_model, "univariate_model", config.code, config.forecast, config.lookback)
        
        #Clean up memory after univariate model training and quantization
        memory_cleanup()
        del univ_model, 

        

        # =================== PHASE 3: PREPARE DATA AND TRAIN DIAGNOSTIC MODELS ===================

        diagnostics_start_time = datetime.now()
        diagnostics_model, train_corrected_diagnostics_predictions, test_corrected_diagnostics_predictions, loss_diagnostics, mae_diagnostics, mse_diagnostics = diagnostics_model_training_pipeline(
            config,
            X_train_diagnostics, Y_train_diagnostics, 
            X_test_diagnostics,Y_test_diagnostics, 
            train_univ_predictions, test_univ_predictions, 
            Y_train_univ, Y_test_univ, 
            diagnostics_model_name, 
            residual_model_params, training_parameters
        )

        diagnostics_duration = float((datetime.now() - diagnostics_start_time).total_seconds())
        mlflow.log_metrics({"duration/phase_2_diagnostics_duration_seconds": diagnostics_duration,})
        mlflow.keras.log_model(diagnostics_model, artifact_path="diagnostic_model")
        load_mlflow_model_history(diagnostics_model_name, model_type="diagnostic_transformer")
        save_model_parameters(config, diagnostics_model, diagnostics_model_name, config.model_parameters_path)
        # Quantize diagnostic model
        quant_diagnostics_model = run_quantization_pipeline.manual_weight_quantization(diagnostics_model, model_name="diagnostics_model")
        run_quantization_pipeline.save_quantized_model_weights(quant_diagnostics_model, "diagnostics_model", config.code, config.forecast, config.lookback)
        # Clean up memory after diagnostic model training and quantization
        memory_cleanup()
        del diagnostics_model


        # =================== PHASE 4: PREPARE DATA AND TRAIN SEASONAL MODELS ===================
        # Prepare residuals for seasonal model training
        
        Y_train_residuals_diagnostics = prepare_residual_data(
            train_corrected_diagnostics_predictions,
            Y_train_univ,
        )
        Y_test_residuals_diagnostics = prepare_residual_data(
            test_corrected_diagnostics_predictions,
            Y_test_univ,
        )
        
        X_train_seasonal = generate_rolling_sequences_covariates(
            df_processed = df_train_seasonal,
            lookback=config.lookback,
            forecast=config.forecast,
            predictions_train = train_corrected_diagnostics_predictions,

        )
        X_test_seasonal = generate_rolling_sequences_covariates(
            df_processed = df_test_seasonal,
            lookback=config.lookback,
            forecast=config.forecast,
            predictions_train = test_corrected_diagnostics_predictions,
        )

        # =================== PHASE 4.1: TRAIN SEASONAL MODELS ===================
        seasonal_start_time = datetime.now()
        seasonal_model, train_final_predictions, test_final_predictions, loss_seasonal, mae_seasonal, mse_seasonal = seasonal_model_training_pipeline(
            config,
            X_train_seasonal, Y_train_residuals_diagnostics, 
            X_test_seasonal, Y_test_residuals_diagnostics, 
            train_corrected_diagnostics_predictions, test_corrected_diagnostics_predictions, 
            seasonal_model_name,
            residual_model_params, training_parameters
        )

        seasonal_duration = float((datetime.now() - seasonal_start_time).total_seconds())
        mlflow.log_metrics({"duration/phase_3_seasonal_duration_seconds": seasonal_duration,})
        mlflow.keras.log_model(seasonal_model, artifact_path="seasonal_model")
        load_mlflow_model_history(seasonal_model_name, model_type="seasonal_transformer")
        save_model_parameters(config, seasonal_model, seasonal_model_name, config.model_parameters_path)
        # Quantize seasonal model
        quant_seasonal_model = run_quantization_pipeline.manual_weight_quantization(seasonal_model, model_name="seasonal_model")
        run_quantization_pipeline.save_quantized_model_weights(quant_seasonal_model, "seasonal_model", config.code, config.forecast, config.lookback)
        # Clean up memory after seasonal model training and quantization
        memory_cleanup()
        del seasonal_model

        # =================== PHASE 5: EVALUATE MODELS ===================
        evaluate_transformer_models(config,
                                    base_model_name, diagnostics_model_name, seasonal_model_name,
                                    Y_test_univ, test_univ_predictions, loss_univ,
                                    test_corrected_diagnostics_predictions, loss_diagnostics,
                                    test_final_predictions, loss_seasonal, date_list
                                    )


        # ==================== PHASE 6: MODEL QUANTIZATION PIPELINE ===================
        
        



        
       
        
        
        # Test quantized models predictions
        '''quant_univ_predictions = quant_univ_model.predict(X_test_univ)
        quant_diagnostics_residuals = quant_diagnostics_model.predict(X_test_diagnostics)
        quant_corrected_diagnostics_predictions = quant_univ_predictions + quant_diagnostics_residuals
        quant_seasonal_residuals = quant_seasonal_model.predict(X_test_seasonal)
        quant_final_predictions = quant_corrected_diagnostics_predictions + quant_seasonal_residuals

        quant_loss_univ, quant_mae_univ, quant_mse_univ = quant_univ_model.evaluate(X_test_univ, Y_test_univ, verbose=0, batch_size=config.batch_size)
        quant_loss_diagnostics, quant_mae_diagnostics, quant_mse_diagnostics = quant_diagnostics_model.evaluate(X_test_diagnostics, Y_test_diagnostics, verbose=0, batch_size=config.batch_size)
        quant_loss_seasonal, quant_mae_seasonal, quant_mse_seasonal = quant_seasonal_model.evaluate(X_test_seasonal, Y_test_residuals_diagnostics, verbose=0, batch_size=config.batch_size)     

        quantized_base_model_name = base_model_name + "_quantized"
        quantized_diagnostics_model_name = diagnostics_model_name + "_quantized"
        quantized_seasonal_model_name = seasonal_model_name + "_quantized"
        evaluate_transformer_models(config,
                                    quantized_base_model_name, quantized_diagnostics_model_name, quantized_seasonal_model_name,
                                    Y_test_univ, quant_univ_predictions, quant_loss_univ,
                                    quant_corrected_diagnostics_predictions, quant_loss_diagnostics,
                                    quant_final_predictions, quant_loss_seasonal, date_list
                                    )'''
        
        # ==================== PHASE 7: inference ====================





if __name__ == "__main__":
    main()
    