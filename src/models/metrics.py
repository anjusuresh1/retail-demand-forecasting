import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def mean_absolute_percentage_error(
    y_true,
    y_pred,
    epsilon: float = 1e-8
) -> float:
    """Calculate MAPE"""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    
    denominator = np.maximum(np.abs(actual), epsilon)
    
    return float(np.mean(np.abs((actual - predicted) / denominator)) * 100)

def weighted_absolute_percentage_error(y_true, y_pred, epsilon: float = 1e-8) -> float:
    """Caclulate WAPE"""
    actual = np.asarray(y_true)
    predicted = np.asarray(y_pred)
    
    numerator = np.sum(abs(actual - predicted))
    denominator = max(np.sum(abs(actual)), epsilon)
    
    return float(numerator / denominator * 100)

def regression_metrics(y_true, y_pred) -> dict:
    """Return a group of forecasting metrics"""
    return {
        'MAE' : mean_absolute_error(y_true, y_pred),
        'RMSE': float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'MAPE': mean_absolute_percentage_error(y_true, y_pred),
        'WAPE': weighted_absolute_percentage_error(y_true, y_pred)
        
    }