import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # non-interactive backend (no GUI)
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

from src import main_train_diagnostic_residual_transformer
from src import main_train_seasonal_residual_transformer

from src import main_training_univ_transformer
#load and visualize data 

from src.univariate_transformer import default_config
# MAIN TRANSFORMER MODEL 

LOOKBACK_LIST = default_config.LOOKBACK_LIST
FORECAST_LIST = default_config.FORECAST_LIST
CODES_LIST = default_config.CODES_LIST

for CODE in CODES_LIST:
    for lb in LOOKBACK_LIST:
        for fh in FORECAST_LIST:
            lookback = lb
            forecast = fh
            code = CODE
            main_training_univ_transformer.main_univ_transformer(forecast=forecast,lookback=lookback, code=code)
            print(f"\n\nRunning for Lookback: {lookback}, Forecast: {forecast}, Code: {code}\n")
            # RESIDUAL DIAGNOSTICS TRANSFORMER
            predictions_train_corrected, predictions_test_corrected = main_train_diagnostic_residual_transformer.main_train_diagnostic_residual_transformer(lookback=lookback, forecast=forecast, code=code, predictions_train_corrected=None, predictions_test_corrected=None)

            # RESIDUAL SEASONAL TRANSFORMER
            predictions_train_corrected, predictions_test_corrected = main_train_seasonal_residual_transformer.main_train_seasonal_residual_transformer(lookback=lookback, forecast=forecast, code=code, predictions_train_corrected=predictions_train_corrected, predictions_test_corrected=predictions_test_corrected)





