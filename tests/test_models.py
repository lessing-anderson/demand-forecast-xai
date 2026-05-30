"""
Unit tests for model layer.
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.base_model import BaseModel
from src.models.lightgbm_model import LightGBMModel


class TestLightGBMModel(unittest.TestCase):
    """Test LightGBM model class."""
    
    def setUp(self):
        """Create sample data and model."""
        # Create sample data
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = np.random.randn(n_samples)
        
        self.X_test = pd.DataFrame(
            np.random.randn(20, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        
        self.model = LightGBMModel(name='test_lgb')
    
    def test_model_initialization(self):
        """Test model can be initialized."""
        self.assertEqual(self.model.name, 'test_lgb')
        self.assertFalse(self.model.is_fitted)
    
    def test_model_training(self):
        """Test model can be trained."""
        self.model.train(
            self.X_train, self.y_train,
            num_rounds=10,
            early_stopping=5
        )
        
        self.assertTrue(self.model.is_fitted)
        self.assertIsNotNone(self.model.model)
    
    def test_model_prediction(self):
        """Test model can make predictions."""
        self.model.train(
            self.X_train, self.y_train,
            num_rounds=10,
            early_stopping=5
        )
        
        predictions = self.model.predict(self.X_test)
        
        self.assertEqual(len(predictions), len(self.X_test))
        self.assertTrue(np.all(np.isfinite(predictions)))
    
    def test_feature_importance(self):
        """Test feature importance retrieval."""
        self.model.train(
            self.X_train, self.y_train,
            num_rounds=10,
            early_stopping=5
        )
        
        importance = self.model.get_feature_importance(top_k=3)
        
        self.assertEqual(len(importance), 3)
        self.assertIn('feature', importance.columns)
        self.assertIn('importance', importance.columns)
    
    def test_predict_before_training_raises_error(self):
        """Test that prediction before training raises error."""
        with self.assertRaises(ValueError):
            self.model.predict(self.X_test)


if __name__ == '__main__':
    unittest.main()
