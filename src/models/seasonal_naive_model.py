"""
Seasonal Naive-based demand forecasting model.
"""
import numpy as np
from .base_model import BaseModel


class SeasonalNaiveModel(BaseModel):
    """Seasonal Naive implementation for demand forecasting."""
    
    def __init__(self, name='seasonal_naive_forecast', **params):
        super().__init__(name)
        self.seasonal_lag_col = 'sales_lag_7'
        self.fallback_value = 0.0
        self.feature_importance_type = 'gain'
        
    
    def train(self, X_train, y_train):
        """Train a seasonal naive t-7 model."""
        print(f"Training {self.name} with {X_train.shape[0]} samples and {X_train.shape[1]} features...")

        y_values = np.asarray(y_train, dtype=float)
        self.fallback_value = float(np.nanmean(y_values)) if y_values.size else 0.0
        if np.isnan(self.fallback_value):
            self.fallback_value = 0.0

        self.feature_names = X_train.columns.tolist() if hasattr(X_train, 'columns') else None
        self.model = {
            'type': 'seasonal_naive_t7',
            'lag_column': self.seasonal_lag_col,
            'fallback_value': self.fallback_value,
        }
        self.is_fitted = True

        print(f"[+] Seasonal naive t-7 ready using '{self.seasonal_lag_col}'")
        return self
    
    def predict(self, X):
        """Predict using sales from t-7."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} has not been trained yet!")

        predictions = np.full(X.shape[0], self.fallback_value, dtype=float)

        if hasattr(X, 'columns') and self.seasonal_lag_col in X.columns:
            lag_values = X[self.seasonal_lag_col].to_numpy(dtype=float, copy=True)
            lag_values[np.isnan(lag_values)] = self.fallback_value
            predictions = lag_values

        return predictions
        
    
    def get_feature_importance(self, importance_type='gain', top_k=20):
        """Return pseudo-importance for compatibility with model API."""
        if not self.is_fitted:
            raise ValueError(f"{self.name} has not been trained yet!")

        import pandas as pd

        self.feature_importance_type = importance_type

        if not self.feature_names:
            return pd.DataFrame(columns=['feature', 'importance'])

        importance = np.zeros(len(self.feature_names), dtype=float)
        if self.seasonal_lag_col in self.feature_names:
            importance[self.feature_names.index(self.seasonal_lag_col)] = 1.0

        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        if top_k:
            feature_df = feature_df.head(top_k)

        return feature_df
    
    def get_model(self):
        """Return the underlying Naive model."""
        return self.model
