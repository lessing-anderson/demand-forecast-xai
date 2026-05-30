"""
Abstract base model class for forecasting.
"""
from abc import ABC, abstractmethod


class BaseModel(ABC):
    """Abstract base class for demand forecasting models."""
    
    def __init__(self, name):
        self.name = name
        self.model = None
        self.feature_names = None
        self.is_fitted = False
    
    @abstractmethod
    def train(self, X_train, y_train, **kwargs):
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X):
        """Make predictions."""
        pass
    
    @abstractmethod
    def get_feature_importance(self):
        """Get feature importance scores."""
        pass
    
    def set_feature_names(self, feature_names):
        """Set feature names for the model."""
        self.feature_names = feature_names
    
    def get_model(self):
        """Return the underlying model object."""
        return self.model
