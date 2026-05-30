"""
LIME-based explainability module.
"""
import lime
import lime.lime_tabular
import pandas as pd
import numpy as np


class LIMEExplainer:
    """LIME explainer for model predictions."""
    
    def __init__(self, model, X_train, feature_names=None):
        """
        Initialize LIME explainer.
        
        Args:
            model: Trained model with predict method
            X_train: Training data for LIME
            feature_names: Feature names (optional)
        """
        self.model = model
        self.X_train = X_train if isinstance(X_train, np.ndarray) else X_train.values
        self.feature_names = feature_names or (X_train.columns.tolist() if hasattr(X_train, 'columns') else None)
        
        print("Initializing LIME explainer...")
        self.explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train,
            feature_names=self.feature_names,
            mode='regression',
            verbose=False
        )
        print("✓ LIME explainer ready")
    
    def explain_instance(self, X_instance, num_features=10):
        """
        Get LIME explanation for a single instance.
        
        Args:
            X_instance: Single row (1D or 2D array)
            num_features: Number of features to explain
            
        Returns:
            LIME explanation object
        """
        if isinstance(X_instance, pd.DataFrame):
            X_instance = X_instance.values[0]
        elif isinstance(X_instance, pd.Series):
            X_instance = X_instance.values
        
        explanation = self.explainer.explain_instance(
            data_row=X_instance,
            predict_fn=self.model.predict,
            num_features=num_features
        )
        return explanation
    
    def get_feature_weights_for_instance(self, X_instance, num_features=10):
        """
        Get LIME feature weights for single instance as DataFrame.
        
        Args:
            X_instance: Single row
            num_features: Number of features
            
        Returns:
            DataFrame with feature names and weights
        """
        explanation = self.explain_instance(X_instance, num_features)
        
        weights = pd.DataFrame(
            explanation.as_list(),
            columns=['feature', 'weight']
        )
        
        return weights
    
    def explain_batch(self, X_batch, num_features=10):
        """
        Get LIME explanations for multiple instances.
        
        Args:
            X_batch: Multiple rows
            num_features: Number of features per instance
            
        Returns:
            List of explanations
        """
        if isinstance(X_batch, pd.DataFrame):
            X_batch = X_batch.values
        
        print(f"Computing LIME explanations for {len(X_batch)} instances...")
        explanations = []
        
        for i, instance in enumerate(X_batch):
            if (i + 1) % max(1, len(X_batch) // 10) == 0:
                print(f"  Explained {i + 1}/{len(X_batch)} instances")
            
            exp = self.explainer.explain_instance(
                data_row=instance,
                predict_fn=self.model.predict,
                num_features=num_features
            )
            explanations.append(exp)
        
        print("✓ LIME explanations computed")
        return explanations
