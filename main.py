import pandas as pd
import tensorflow as tf
import numpy as np
import pandas as pd
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
CODE = default_config.TARGET_CODE



for lb in LOOKBACK_LIST:
    for fh in FORECAST_LIST:
        lookback = lb
        forecast = fh
        code = CODE
        main_training_univ_transformer.main_univ_transformer(lookback=lookback, forecast=forecast, code=code)
        print(f"\n\nRunning for Lookback: {lookback}, Forecast: {forecast}, Code: {code}\n")
        corrected_forecast_values = main_train_diagnostic_residual_transformer.main_train_diagnostic_residual_transformer(lookback=lookback, forecast=forecast, code=code)
        corrected_forecast_values = main_train_seasonal_residual_transformer.main_train_seasonal_residual_transformer(lookback=lookback, forecast=forecast, code=code, corrected_forecast_values=corrected_forecast_values)

# RESIDUAL DIAGNOSTICS TRANSFORMER



# RESIDUAL SEASONAL TRANSFORMER 