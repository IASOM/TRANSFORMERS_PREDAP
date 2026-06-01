import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def crps(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)

def smape(y_true, y_pred):
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred)
    return np.mean(np.where(denominator == 0, 0, diff / denominator)) * 100

def pinball_loss(y_true, y_pred, tau=0.5):
    error = y_true - y_pred
    return np.mean(np.maximum(tau * error, (tau - 1) * error))

def mean_absolute_percentage_error(y_true, y_pred, epsilon=1e-8):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = np.abs(y_true) > epsilon
    if mask.sum() == 0:
        return float('nan')
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
