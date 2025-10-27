
import os 
import tensorflow as tf
import data_preparation
from univariate_transformer import plt_model, plot_predictions_with_waves, extract_model_params
from residual_multivariate_transformers import load_base_model_transformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import pandas as pd
import numpy as np

def evaluate_univ_transformer(model_name, input_directory, code,  cutoff_date, covid_token = False, MODEL_FOLDER='models_univariate_transformer', df_waves=None):
    print(f"\n--- Evaluating model: {model_name} ---")
        
    # Extract parameters from filename
    lookback, forecast = extract_model_params(model_name)
    
    if lookback is None or forecast is None:
        print(f"Skipping {model_name} - could not extract parameters, the lookback or forecast value is None.")
        return  # Stop evaluation for this model

    # Load model
    model_path = os.path.join(MODEL_FOLDER, model_name)
    model = tf.keras.models.load_model(model_path, compile=True)

    # Prepare test data
    X_test, Y_test = data_preparation.prepare_data(
        input_directory, code, lookback, forecast,covid_token=covid_token,  cutoff_date=cutoff_date, train=False, debug=True, univariate=True
    )

    X_test_orig, Y_test_orig = data_preparation.prepare_data_not_normalized(
        input_directory, code, lookback, forecast,covid_token=covid_token,  cutoff_date=cutoff_date, train=False, debug=True, univariate=True
    )

    date_list = data_preparation.extract_dates(input_directory, code, lookback, forecast, train=False)

    original_scale_df = pd.read_csv(input_directory)
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    print("Predicted values shape:", predictions.shape)
    
      # Inverse transform predictions
    predictions_to_plot = data_preparation.inverse_transform_predictions(
        predictions, original_scale_df, code, cutoff_date=cutoff_date
    )
    
    # Evaluate model
    loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)
    original_mae = mean_absolute_error(Y_test_orig, predictions_to_plot)
    original_mse = mean_squared_error(Y_test_orig, predictions_to_plot)

    original_rmse = np.sqrt(original_mse)
    print(f"Test Results - Loss: {loss:.4f}, MAE: {original_mae:.4f}, MSE: {original_mse:.4f}")
    


    
    # Generate plots
    model_display_name = model_name.replace('.keras', '')


    plt_model(Y_test_orig, predictions_to_plot, date_list, model_name=model_display_name, show_plt=False)

    # Plot predictions with pandemic waves
    plot_predictions_with_waves(Y_test_orig, predictions_to_plot, date_list, df_waves, model_display_name)

    # Sliding window evaluation (optional)
    # evaluate_model_sliding_window(model, model_display_name, X_test, Y_test, date_list, df_waves, sliding_window=forecast)
    return loss, original_mae, original_mse