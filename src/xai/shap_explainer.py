"""
SHAP-based explainability module.
"""
import shap
import pandas as pd
import numpy as np


class SHAPExplainer:
    """SHAP explainer for model predictions."""
    
    def __init__(self, model, X_background=None):
        """
        Initialize SHAP explainer.
        
        Args:
            model: Trained model with predict method
            X_background: Background data for SHAP
        """
        self.model = model
        self.X_background = X_background
        self.explainer = None
        self.shap_values = None
    
    def fit(self, X_sample=None):
        """
        Create SHAP explainer.
        
        Args:
            X_sample: Data sample to use for SHAP background
        """
        print("Initializing SHAP explainer...")
        
        # Use background data or first 100 samples
        bg_data = self.X_background if self.X_background is not None else X_sample.iloc[:100]
        
        self.explainer = shap.TreeExplainer(self.model)
        print("✓ SHAP explainer ready")
        return self
    
    def explain_instance(self, X_instance):
        """
        Get SHAP values for a single instance.
        
        Args:
            X_instance: Single row (2D array or DataFrame)
            
        Returns:
            SHAP values for the instance
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        shap_vals = self.explainer.shap_values(X_instance)
        return shap_vals
    
    def explain_batch(self, X_batch):
        """
        Get SHAP values for a batch of instances.
        
        Args:
            X_batch: Multiple rows
            
        Returns:
            SHAP values for the batch
        """
        if self.explainer is None:
            raise ValueError("Explainer not fitted. Call fit() first.")
        
        print(f"Computing SHAP values for {len(X_batch)} instances...")
        shap_vals = self.explainer.shap_values(X_batch)
        print("✓ SHAP values computed")
        return shap_vals
    
    def get_feature_importance_from_shap(self, X_data, top_k=20):
        """
        Compute mean absolute SHAP values for feature importance.
        
        Args:
            X_data: Data to compute importance on
            top_k: Return top K features
            
        Returns:
            DataFrame with SHAP-based importance
        """
        shap_vals = self.explain_batch(X_data)
        
        mean_abs_shap = np.abs(shap_vals).mean(axis=0)
        
        importance_df = pd.DataFrame({
            'feature': X_data.columns,
            'mean_abs_shap': mean_abs_shap
        }).sort_values('mean_abs_shap', ascending=False)
        
        if top_k:
            importance_df = importance_df.head(top_k)
        
        return importance_df
