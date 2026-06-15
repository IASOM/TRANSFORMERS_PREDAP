
import os
import json
import sys
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.metrics import mean_absolute_error
import tensorflow as tf
import pandas as pd
from datetime import datetime
import numpy as np

from training.training_residual_transformer import load_trained_model
from utils.mlflow_logger import MLflowLogger
from visualization_func.visualization_transformer import plot_residuals_analysis, plt_model

def save_performance_results(model_name, loss, 
                             original_mae, original_mse, 
                             original_rmse, original_wape,
                             corrected_mae, corrected_mse, 
                             corrected_rmse, corrected_wape,
                             forecast, lookback, code, output_dir="results"):
    """
    Save performance comparison results to a JSON file.
    
    Parameters:
    -----------
    model_name : str
        Name of the residual model being evaluated
    loss : float
        Loss value for the model
    original_mae, original_mse, original_rmse : float
        Performance metrics for the original base model
    corrected_mae, corrected_mse, corrected_rmse : float
        Performance metrics for the residual-corrected model
    forecast : int
        Forecast horizon used
    lookback : int
        Lookback window used
    code : str
        Target diagnostic code
    output_dir : str
        Directory to save the results file
    
    Returns:
    --------
    str
        Path to the saved JSON file
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists('../' +output_dir):
        os.makedirs('../' + output_dir)
        print(f"Created directory: {output_dir}")
    
    # Calculate improvements
    mae_improvement = float((original_mae - corrected_mae) / original_mae * 100) if original_mae != 0 else 0
    mse_improvement = float((original_mse - corrected_mse) / original_mse * 100) if original_mse != 0 else 0
    rmse_improvement =float((original_rmse - corrected_rmse) / original_rmse * 100) if original_rmse != 0 else 0
    wape_improvement =float((original_wape - corrected_wape) / original_wape * 100) if original_wape != 0 else 0
    
    # Create results dictionary
    results = {
        "model_info": {
            "residual_model_name": model_name,
            "target_code": code,
            "forecast_horizon": forecast,
            "lookback_window": lookback,
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_type": "Residual Multivariate Transformer"
        },
        "original_model_performance": {
            "MAE": round(float(original_mae), 6),
            "MSE": round(float(original_mse), 6),
            "RMSE": round(float(original_rmse), 6),
            "WAPE": round(float(original_wape), 6)
        },
        "corrected_model_performance": {
            "MAE": round(float(corrected_mae), 6),
            "MSE": round(float(corrected_mse), 6),
            "RMSE": round(float(corrected_rmse), 6),
            "WAPE": round(float(corrected_wape), 6)

        },
        "improvements": {
            "MAE_improvement_percent": round(float(mae_improvement), 2),
            "MSE_improvement_percent": round(float(mse_improvement), 2),
            "RMSE_improvement_percent": round(float(rmse_improvement), 2),
            "WAPE_improvement_percent": round(float(wape_improvement), 2),
            "overall_assessment": "positive" if float(mae_improvement) > 0 else "negative"
        },
        "summary": {
            "best_metric": "MAE" if abs(mae_improvement) >= max(abs(mse_improvement), abs(rmse_improvement)) else 
                         "MSE" if abs(mse_improvement) >= abs(rmse_improvement) else "RMSE",
            "best_improvement": max(mae_improvement, mse_improvement, rmse_improvement),
            "average_improvement": round((mae_improvement + mse_improvement + rmse_improvement) / 3, 2)
        }
    }

    MLflowLogger(active=True).log_metrics({
        f"eval/{model_name}_loss": loss,
        f"eval/{model_name}_mae": corrected_mae,
        f"eval/{model_name}_mse": corrected_mse,
        f"eval/{model_name}_rmse": corrected_rmse,
        f"eval/{model_name}_wape": corrected_wape
    })
    
    # Generate filename based on model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = model_name.replace('.keras', '').replace('/', '_').replace('\\', '_')
    filename = f"performance_{clean_model_name}_{timestamp}.json"
    filepath = os.path.join('../' +output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n📊 Performance results saved to: {filepath}")
    
    return filepath


def save_univ_performance_results(model_name, loss, 
                             original_mae, original_mse, 
                             original_rmse, original_wape,
                             forecast, lookback, code, output_dir="results"):
    """
    Save performance comparison results to a JSON file.
    
    Parameters:
    -----------
    model_name : str
        Name of the residual model being evaluated
    loss : float
        Loss value for the model
    original_mae, original_mse, original_rmse : float
        Performance metrics for the original base model
    corrected_mae, corrected_mse, corrected_rmse : float
        Performance metrics for the residual-corrected model
    forecast : int
        Forecast horizon used
    lookback : int
        Lookback window used
    code : str
        Target diagnostic code
    output_dir : str
        Directory to save the results file
    
    Returns:
    --------
    str
        Path to the saved JSON file
    """
    
    # Create output directory if it doesn't exist
    if not os.path.exists('../' +output_dir):
        os.makedirs('../' + output_dir)
        print(f"Created directory: {output_dir}")
    
    # Create results dictionary
    results = {
        "model_info": {
            "univ_model_name": model_name,
            "target_code": code,
            "forecast_horizon": forecast,
            "lookback_window": lookback,
            "evaluation_timestamp": datetime.now().isoformat(),
            "model_type": "Residual Multivariate Transformer"
        },
        "univ_model_performance": {
            "MAE": round(float(original_mae), 6),
            "MSE": round(float(original_mse), 6),
            "RMSE": round(float(original_rmse), 6),
            "WAPE": round(float(original_wape), 6)
        },
        
    }

    MLflowLogger(active=True).log_metrics({
        f"eval/{model_name}_loss": loss,
        f"eval/{model_name}_mae": original_mae,
        f"eval/{model_name}_mse": original_mse,
        f"eval/{model_name}_rmse": original_rmse,
        f"eval/{model_name}_wape": original_wape
    })
    
    # Generate filename based on model name and timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_model_name = model_name.replace('.keras', '').replace('/', '_').replace('\\', '_')
    filename = f"performance_{clean_model_name}_{timestamp}.json"
    filepath = os.path.join('../' +output_dir, filename)
    
    # Save to JSON file
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
    
    print(f"\n📊 Performance results saved to: {filepath}")
    
    return filepath


def evaluate_transformer_models(config, 
                                base_model_name, diagnostics_model_name, seasonal_model_name,
                                Y_test_univ, test_univ_predictions, loss_univ,
                                test_corrected_diagnostics_predictions, loss_diagnostics,
                                test_final_predictions, loss_seasonal, date_list
                                ):
    #Phase 4: Evaluate base model
    print("\n" + "="*50)
    print("EVALUATION PHASE")
    print("="*50)

    non_negative_univ_predictions = np.maximum(test_univ_predictions, 0)  # Ensure no negative predictions

    original_mae_univ = mean_absolute_error(Y_test_univ, non_negative_univ_predictions)
    original_mse_univ = mean_squared_error(Y_test_univ, non_negative_univ_predictions)
    original_rmse_univ = np.sqrt(original_mse_univ)
    original_wape_univ = np.sum(np.abs(Y_test_univ - non_negative_univ_predictions)) / np.sum(np.abs(Y_test_univ)) * 100
    print(f"Test Results - Loss: {loss_univ:.4f}, MAE: {original_mae_univ:.4f}, MSE: {original_mse_univ:.4f}, RMSE: {original_rmse_univ:.4f}, WAPE: {original_wape_univ:.4f}%")
    
    plt_model(Y_test_univ, non_negative_univ_predictions, date_list, model_name=base_model_name, show_plt=False)

    

    
    non_negative_diagnostics_predictions = np.maximum(test_corrected_diagnostics_predictions, 0)  # Ensure no negative predictions
    
    original_mae_diagnostics = mean_absolute_error(Y_test_univ, non_negative_diagnostics_predictions)
    original_mse_diagnostics = mean_squared_error(Y_test_univ , non_negative_diagnostics_predictions)
    original_rmse_diagnostics = np.sqrt(original_mse_diagnostics)
    original_wape_diagnostics = np.sum(np.abs(Y_test_univ - non_negative_diagnostics_predictions)) / np.sum(np.abs(Y_test_univ)) * 100
    print(f"Diagnostics Test Results - Loss: {loss_diagnostics:.4f}, MAE: {original_mae_diagnostics:.4f}, MSE: {original_mse_diagnostics:.4f}, RMSE: {original_rmse_diagnostics:.4f}, WAPE: {original_wape_diagnostics:.4f}%")
    plt_model(Y_test_univ, non_negative_diagnostics_predictions, date_list, model_name=diagnostics_model_name, show_plt=False)
    plot_residuals_analysis(non_negative_univ_predictions, 
                            non_negative_diagnostics_predictions,
                            Y_test_univ,
                            f"{config.code} Diagnostics Residual Correction", 
                            model_name=diagnostics_model_name,
                            timestamp=date_list,
                            show_plt=False)
    
    non_negative_seasonal_predictions = np.maximum(test_final_predictions, 0)  # Ensure no negative predictions

    original_mae_seasonal = mean_absolute_error(Y_test_univ, non_negative_seasonal_predictions)
    original_mse_seasonal = mean_squared_error(Y_test_univ, non_negative_seasonal_predictions)
    original_rmse_seasonal = np.sqrt(original_mse_seasonal)
    original_wape_seasonal = np.sum(np.abs(Y_test_univ - non_negative_seasonal_predictions)) / np.sum(np.abs(Y_test_univ)) * 100
    print(f"Seasonal Test Results - Loss: {loss_seasonal:.4f}, MAE: {original_mae_seasonal:.4f}, MSE: {original_mse_seasonal:.4f}, RMSE: {original_rmse_seasonal:.4f}, WAPE: {original_wape_seasonal:.4f}%")
    plt_model(Y_test_univ, non_negative_seasonal_predictions, date_list, model_name=seasonal_model_name, show_plt=False)
    plot_residuals_analysis(non_negative_diagnostics_predictions, 
                            non_negative_seasonal_predictions, 
                            Y_test_univ, 
                            f"{config.code} Seasonal Residual Correction",
                            model_name=diagnostics_model_name,
                            timestamp=date_list,
                            show_plt=False)
    
    save_univ_performance_results(
        model_name=base_model_name,
        original_mae=original_mae_univ,
        original_mse=original_mse_univ,
        original_rmse=original_rmse_univ,
        original_wape=original_wape_univ,
        forecast=config.forecast,
        lookback=config.lookback,
        loss =loss_seasonal,
        code=config.code    
    )


    save_performance_results(
        model_name=diagnostics_model_name,
        original_mae=original_mae_univ,
        original_mse=original_mse_univ,
        original_rmse=original_rmse_univ,   
        original_wape=original_wape_univ,
        corrected_mae=original_mae_diagnostics,
        corrected_mse=original_mse_diagnostics,
        corrected_rmse=original_rmse_diagnostics,
        corrected_wape=original_wape_diagnostics,
        forecast=config.forecast,
        lookback=config.lookback,
        loss=loss_diagnostics,
        code=config.code
    )

    save_performance_results(
        model_name=seasonal_model_name,
        original_mae=original_mae_univ,
        original_mse=original_mse_univ,
        original_rmse=original_rmse_univ,
        original_wape=original_wape_univ,
        corrected_mae=original_mae_seasonal,
        corrected_mse=original_mse_seasonal,
        corrected_rmse=original_rmse_seasonal,
        corrected_wape=original_wape_seasonal,
        forecast=config.forecast,
        lookback=config.lookback,
        loss=loss_seasonal,
        code=config.code
    )
