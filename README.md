# WAIKATO_FORECAST_demand

![Alt text](WAIKATO.drawio.png)

This notebook provides a comprehensive approach to forecasting demand for health diagnostic visits using deep learning models, including vanilla Transformer, MultiheadAttention mechanisms models and Recurrent Neural Networks (RNNs) with feature selection methods.

The forecasting algorithm consists of three key phases:

- Initial Forecasting – A univariate model generates the baseline forecast.
- Residual Prediction – In subsequent phases, the forecast is refined by modeling the residuals.
- Incorporating Seasonality & Lagged Features – The final predictions integrate seasonal patterns and lagged variables that influence demand, enhancing accuracy.

This approach leverages both sequential modeling techniques and residual learning to improve forecasting performance, making it a powerful tool for time series prediction in healthcare demand.
