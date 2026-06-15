"""
LightGBM-based demand forecasting model.
"""
import lightgbm as lgb
import numpy as np
from .base_model import BaseModel


class LightGBMModel(BaseModel):
    """LightGBM implementation for demand forecasting."""
    
    def __init__(self, name='lightgbm_forecast', **lgb_params):
        super().__init__(name)
        self.lgb_params = lgb_params or self._default_params()
        self.feature_importance_type = 'gain'
    
    def _default_params(self):
        """Default LightGBM hyperparameters."""
        return {
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None, num_rounds=1000, early_stopping=50):
        """
        Train LightGBM model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
            num_rounds: Number of boosting rounds
            early_stopping: Early stopping rounds
            
        Returns:
            Training history
        """
        print(f"Training {self.name} with {X_train.shape[0]} samples and {X_train.shape[1]} features...")
        
        train_data = lgb.Dataset(X_train, label=y_train)
        
        eval_sets = [train_data]
        eval_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            eval_sets.append(val_data)
            eval_names.append('valid')
        
        self.model = lgb.train(
            params=self.lgb_params,
            train_set=train_data,
            num_boost_round=num_rounds,
            valid_sets=eval_sets,
            valid_names=eval_names,
            callbacks=[
                lgb.early_stopping(early_stopping),
                lgb.log_evaluation(period=100),
            ]
        )
        
        self.is_fitted = True
        self.feature_names = X_train.columns.tolist()
        
        print(f"[+] Model training complete! (num_boost_round: {self.model.num_trees()})")
        return self
    
    def predict(self, X):
        """
        Make predictions on new data.
        
        Args:
            X: Features to predict on
            
        Returns:
            Predicted values
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} has not been trained yet!")
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_feature_importance(self, importance_type='gain', top_k=20):
        """
        Get feature importance scores.
        
        Args:
            importance_type: 'gain' or 'split'
            top_k: Return top K features
            
        Returns:
            DataFrame with feature importance
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} has not been trained yet!")
        
        import pandas as pd
        
        importance = self.model.feature_importance(importance_type=importance_type)
        feature_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        if top_k:
            feature_df = feature_df.head(top_k)
        
        return feature_df
    
    def get_model(self):
        """Return the underlying LightGBM model."""
        return self.model
