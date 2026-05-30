"""
Main data processing and model training pipeline.
Orchestrates workflow from raw data to predictions.
"""
import pandas as pd
import numpy as np
from pathlib import Path

from src.data.loader import load_and_merge
from src.data.preprocessing import create_features, reduce_mem_usage
from src.data.splitting import isolate_ground_truth, prepare_train_data, prepare_test_data
from src.models.lightgbm_model import LightGBMModel
from src.utils.metrics import evaluate_forecast
from src.utils.logger import setup_logger


logger = setup_logger('pipeline')


class DemandForecastPipeline:
    """End-to-end demand forecasting pipeline."""
    
    def __init__(self, raw_data_path, split_date='2016-04-24', store_filter=None):
        """
        Initialize pipeline.
        
        Args:
            raw_data_path: Path to raw M5 data
            split_date: Train/test split date (YYYY-MM-DD)
            store_filter: Optional store_id to filter
        """
        self.raw_data_path = raw_data_path
        self.split_date = split_date
        self.store_filter = store_filter
        
        self.df = None
        self.df_ground_truth = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.test_df = None
        self.model = None
        self.predictions = None
        
        logger.info(f"Initialized pipeline for {store_filter or 'all stores'}")
    
    def load_and_prepare_data(self, optimize_memory=False):
        """Load and prepare data up to splitting."""
        logger.info("Loading and merging data...")
        self.df = load_and_merge(self.raw_data_path, self.store_filter)
        
        logger.info("Creating features...")
        self.df = create_features(self.df)
        
        if optimize_memory:
            logger.info("Optimizing memory usage...")
            self.df = reduce_mem_usage(self.df)
        
        logger.info("Isolating ground truth and nullifying test targets...")
        self.df, self.df_ground_truth = isolate_ground_truth(self.df, self.split_date)
        
        logger.info(f"Data shape: {self.df.shape}")
        return self
    
    def prepare_train_test(self):
        """Prepare training and test sets."""
        logger.info("Preparing train/test splits...")
        
        self.X_train, self.y_train, test_df = prepare_train_data(
            self.df, self.split_date
        )
        
        self.X_test, self.test_df = prepare_test_data(test_df)
        
        logger.info(f"X_train shape: {self.X_train.shape}")
        logger.info(f"y_train shape: {self.y_train.shape}")
        logger.info(f"X_test shape: {self.X_test.shape}")
        
        return self
    
    def train_model(self, model_name='lightgbm', **model_kwargs):
        """Train forecasting model."""
        logger.info(f"Training {model_name} model...")
        
        if model_name == 'lightgbm':
            self.model = LightGBMModel(**model_kwargs)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        self.model.train(self.X_train, self.y_train)
        
        return self
    
    def predict(self, X=None):
        """Make predictions on test set or custom data."""
        if X is None:
            X = self.X_test
        
        logger.info(f"Making predictions on {X.shape[0]} samples...")
        self.predictions = self.model.predict(X)
        
        return self.predictions
    
    def evaluate(self):
        """Evaluate model on test set."""
        logger.info("Evaluating model...")
        
        # Get actual values from ground truth
        test_dates = self.test_df[['id', 'date']].copy()
        test_dates['pred'] = self.predictions
        
        # Merge with ground truth
        results = test_dates.merge(
            self.df_ground_truth,
            on=['id', 'date'],
            how='left'
        )
        
        # Evaluate on non-null values
        mask = results['actual_sales'].notna()
        metrics = evaluate_forecast(
            results.loc[mask, 'actual_sales'].values,
            results.loc[mask, 'pred'].values,
            'Test Set'
        )
        
        return metrics, results
    
    def get_feature_importance(self, top_k=20):
        """Get feature importance from trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet!")
        
        return self.model.get_feature_importance(top_k=top_k)
    
    def run_full_pipeline(self, optimize_memory=False, model_name='lightgbm', **model_kwargs):
        """
        Execute full pipeline from raw data to evaluation.
        
        Args:
            optimize_memory: Apply memory optimization
            model_name: Name of model to train
            **model_kwargs: Model-specific parameters
            
        Returns:
            Metrics and predictions
        """
        logger.info("=" * 60)
        logger.info("DEMAND FORECAST PIPELINE - FULL RUN")
        logger.info("=" * 60)
        
        self.load_and_prepare_data(optimize_memory)
        self.prepare_train_test()
        self.train_model(model_name, **model_kwargs)
        self.predict()
        metrics, results = self.evaluate()
        
        logger.info("=" * 60)
        logger.info("PIPELINE COMPLETE")
        logger.info("=" * 60)
        
        return metrics, results
