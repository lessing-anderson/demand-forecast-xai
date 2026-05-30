"""
Feature importance analysis module.
"""
import pandas as pd
import numpy as np
from sklearn.inspection import permutation_importance


class FeatureImportanceAnalyzer:
    """Compute various feature importance metrics."""
    
    def __init__(self, model):
        """
        Initialize analyzer.
        
        Args:
            model: Trained model
        """
        self.model = model
    
    def get_model_importance(self, importance_type='gain', top_k=20):
        """
        Get model's built-in feature importance.
        
        Args:
            importance_type: Type of importance (model-specific)
            top_k: Return top K features
            
        Returns:
            DataFrame with feature importance
        """
        if hasattr(self.model, 'get_feature_importance'):
            return self.model.get_feature_importance(importance_type, top_k)
        else:
            raise ValueError("Model does not support get_feature_importance method")
    
    def get_permutation_importance(self, X, y, n_repeats=10, top_k=20):
        """
        Compute permutation feature importance.
        
        Args:
            X: Features
            y: Target
            n_repeats: Number of repetitions
            top_k: Return top K features
            
        Returns:
            DataFrame with permutation importance
        """
        print(f"Computing permutation importance ({n_repeats} repeats)...")
        
        perm_importance = permutation_importance(
            self.model.get_model(),
            X, y,
            n_repeats=n_repeats,
            random_state=42,
            n_jobs=-1
        )
        
        importance_df = pd.DataFrame({
            'feature': X.columns,
            'importance_mean': perm_importance.importances_mean,
            'importance_std': perm_importance.importances_std
        }).sort_values('importance_mean', ascending=False)
        
        if top_k:
            importance_df = importance_df.head(top_k)
        
        print("✓ Permutation importance computed")
        return importance_df
    
    def compare_importance_methods(self, X, y, n_repeats=10, top_k=15):
        """
        Compare multiple importance methods.
        
        Args:
            X: Features
            y: Target
            n_repeats: Permutation importance repeats
            top_k: Top K features to compare
            
        Returns:
            DataFrame comparing importance methods
        """
        print("Comparing feature importance methods...")
        
        gain_imp = self.get_model_importance(importance_type='gain', top_k=None)
        perm_imp = self.get_permutation_importance(X, y, n_repeats, top_k=None)
        
        # Merge by feature
        comparison = gain_imp.set_index('feature').rename(columns={'importance': 'gain_importance'})
        comparison['perm_importance'] = perm_imp.set_index('feature')['importance_mean']
        
        # Normalize for comparison
        comparison['gain_normalized'] = comparison['gain_importance'] / comparison['gain_importance'].max()
        comparison['perm_normalized'] = comparison['perm_importance'] / comparison['perm_importance'].max()
        
        comparison = comparison.sort_values('gain_normalized', ascending=False)
        
        if top_k:
            comparison = comparison.head(top_k)
        
        return comparison.reset_index()
