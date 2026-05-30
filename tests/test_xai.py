"""
Unit tests for xAI modules.
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.lightgbm_model import LightGBMModel
from src.xai.feature_importance import FeatureImportanceAnalyzer


class TestFeatureImportanceAnalyzer(unittest.TestCase):
    """Test feature importance analysis."""
    
    def setUp(self):
        """Create sample model and data."""
        np.random.seed(42)
        n_samples = 100
        n_features = 5
        
        self.X_train = pd.DataFrame(
            np.random.randn(n_samples, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_train = np.random.randn(n_samples)
        
        self.X_test = pd.DataFrame(
            np.random.randn(30, n_features),
            columns=[f'feature_{i}' for i in range(n_features)]
        )
        self.y_test = np.random.randn(30)
        
        # Train model
        self.model = LightGBMModel(name='test_lgb')
        self.model.train(self.X_train, self.y_train, num_rounds=10)
        
        self.analyzer = FeatureImportanceAnalyzer(self.model)
    
    def test_get_model_importance(self):
        """Test getting model-based feature importance."""
        importance = self.analyzer.get_model_importance(top_k=3)
        
        self.assertEqual(len(importance), 3)
        self.assertIn('feature', importance.columns)
        self.assertTrue(importance['importance'].is_monotonic_decreasing)
    
    def test_get_permutation_importance(self):
        """Test getting permutation importance."""
        importance = self.analyzer.get_permutation_importance(
            self.X_test, self.y_test,
            n_repeats=2,
            top_k=3
        )
        
        self.assertEqual(len(importance), 3)
        self.assertIn('importance_mean', importance.columns)
        self.assertIn('importance_std', importance.columns)
    
    def test_compare_importance_methods(self):
        """Test comparing different importance methods."""
        comparison = self.analyzer.compare_importance_methods(
            self.X_test, self.y_test,
            n_repeats=2,
            top_k=3
        )
        
        self.assertEqual(len(comparison), 3)
        self.assertIn('gain_normalized', comparison.columns)
        self.assertIn('perm_normalized', comparison.columns)


if __name__ == '__main__':
    unittest.main()
