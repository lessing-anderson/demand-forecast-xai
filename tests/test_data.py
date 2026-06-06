"""
Unit tests for data layer modules.
"""
import unittest
import pandas as pd
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.data_processing import create_calendar_features, encode_event_features, create_lag_features
from src.data.splitting import get_features_and_target, prepare_train_data


class TestDataPreprocessing(unittest.TestCase):
    """Test data preprocessing functions."""
    
    def setUp(self):
        """Create sample data for testing."""
        # Create sample dataframe
        self.df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'id': ['item_1'] * 100,
            'sales': np.random.randint(0, 100, 100),
            'event_name_1': ['New Year'] * 10 + [None] * 90,
            'event_type_1': ['Holiday'] * 10 + [None] * 90,
            'event_name_2': [None] * 100,
            'event_type_2': [None] * 100,
        })
    
    def test_calendar_features(self):
        """Test calendar feature creation."""
        df = create_calendar_features(self.df.copy())
        
        self.assertIn('day_of_month', df.columns)
        self.assertIn('day_of_week', df.columns)
        self.assertIn('week_of_year', df.columns)
        self.assertTrue((df['day_of_month'] >= 1).all())
        self.assertTrue((df['day_of_month'] <= 31).all())
    
    def test_lag_features(self):
        """Test lag feature creation."""
        df = self.df.copy()
        df = df.sort_values('date')
        df = create_lag_features(df, lags=[7, 14])
        
        self.assertIn('sales_lag_7', df.columns)
        self.assertIn('sales_lag_14', df.columns)
    
    def test_event_encoding(self):
        """Test event feature encoding."""
        df = encode_event_features(self.df.copy())
        
        # Should be numeric
        self.assertTrue(pd.api.types.is_numeric_dtype(df['event_name_1']))
        self.assertTrue(pd.api.types.is_numeric_dtype(df['event_type_1']))


class TestDataSplitting(unittest.TestCase):
    """Test data splitting functions."""
    
    def setUp(self):
        """Create sample data."""
        self.df = pd.DataFrame({
            'date': pd.date_range('2020-01-01', periods=100),
            'id': ['item_1'] * 100,
            'sales': np.random.randint(0, 100, 100),
            'feature_1': np.random.randn(100),
            'feature_2': np.random.randn(100),
        })
    
    def test_get_features_and_target(self):
        """Test features and target separation."""
        X, y = get_features_and_target(self.df, target_col='sales')
        
        self.assertNotIn('sales', X.columns)
        self.assertIn('feature_1', X.columns)
        self.assertEqual(len(X), len(y))
    
    def test_prepare_train_data_removes_nulls(self):
        """Test that training data removes null targets."""
        df = self.df.copy()
        df.loc[10:15, 'sales'] = np.nan
        
        split_date = '2020-03-15'
        X_train, y_train, test_df = prepare_train_data(df, split_date)
        
        # Should not have NaN in y_train
        self.assertTrue(~y_train.isna().any())


if __name__ == '__main__':
    unittest.main()
