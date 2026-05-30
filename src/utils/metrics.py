"""
Evaluation metrics for demand forecasting.
"""
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error


def rmse(y_true, y_pred):
    """Root Mean Squared Error."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


def mae(y_true, y_pred):
    """Mean Absolute Error."""
    return mean_absolute_error(y_true, y_pred)


def mape(y_true, y_pred):
    """Mean Absolute Percentage Error."""
    return mean_absolute_percentage_error(y_true, y_pred)


def evaluate_forecast(y_true, y_pred, dataset_name='Test'):
    """
    Calculate comprehensive forecast metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        dataset_name: Name for printing
        
    Returns:
        Dictionary of metrics
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    
    metrics = {
        'rmse': rmse(y_true, y_pred),
        'mae': mae(y_true, y_pred),
        'mape': mape(y_true, y_pred),
    }
    
    print(f"\n{dataset_name} Metrics:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  MAPE: {metrics['mape']:.4f}")
    
    return metrics
