
import os 
import tensorflow as tf
import data_preparation
from univariate_transformer import plt_model, plot_predictions_with_waves, extract_model_params

def evaluate_univ_transformer(model_name, input_directory, code, MODEL_FOLDER='models_univariate_transformer', df_waves=None):
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
        input_directory, code, lookback, forecast, train=False, debug=True, univariate=True
    )
    date_list = data_preparation.extract_dates(input_directory, code, lookback, forecast, train=False)
    
    # Evaluate model
    loss, mae, mse = model.evaluate(X_test, Y_test, verbose=0)
    print(f"Test Results - Loss: {loss:.4f}, MAE: {mae:.4f}, MSE: {mse:.4f}")
    
    # Get predictions
    predictions = model.predict(X_test, verbose=0)
    print("Predicted values shape:", predictions.shape)

    # Generate plots
    model_display_name = model_name.replace('.keras', '')
    plt_model(Y_test, predictions, model_name=model_display_name, col_idx=0, show_plt=False)
    
    # Plot predictions with pandemic waves
    plot_predictions_with_waves(Y_test, predictions, date_list, df_waves, model_display_name)
    
    # Sliding window evaluation (optional)
    # evaluate_model_sliding_window(model, model_display_name, X_test, Y_test, date_list, df_waves, sliding_window=forecast)
    return loss, mae, mse