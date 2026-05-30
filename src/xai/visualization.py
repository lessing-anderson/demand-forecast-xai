"""
Visualization utilities for model explanations.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_feature_importance(importance_df, top_k=20, title="Feature Importance", figsize=(10, 8)):
    """
    Plot feature importance as horizontal bar chart.
    
    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        top_k: Number of top features to show
        title: Plot title
        figsize: Figure size
    """
    df = importance_df.head(top_k).copy()
    
    plt.figure(figsize=figsize)
    sns.barplot(data=df, y='feature', x=df.columns[1], palette='viridis')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel('Importance Score')
    plt.ylabel('Feature')
    plt.tight_layout()
    plt.show()


def plot_shap_summary(shap_values, X, plot_type='bar', top_k=15):
    """
    Plot SHAP summary visualization.
    
    Args:
        shap_values: SHAP values array
        X: Feature data
        plot_type: 'bar' or 'beeswarm'
        top_k: Top K features
    """
    try:
        import shap
        
        if plot_type == 'bar':
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, plot_type='bar', max_display=top_k, show=False)
        else:
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, X, max_display=top_k, show=False)
        
        plt.tight_layout()
        plt.show()
    except ImportError:
        print("SHAP not installed. Install with: pip install shap")


def plot_model_vs_permutation_importance(comparison_df, figsize=(12, 8)):
    """
    Compare model importance vs permutation importance.
    
    Args:
        comparison_df: DataFrame with both importance columns
        figsize: Figure size
    """
    df = comparison_df.sort_values('gain_normalized', ascending=False).head(15)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(df))
    width = 0.35
    
    ax.barh(x - width/2, df['gain_normalized'], width, label='Gain Importance', alpha=0.8)
    ax.barh(x + width/2, df['perm_normalized'], width, label='Permutation Importance', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(df['feature'])
    ax.set_xlabel('Normalized Importance')
    ax.set_title('Feature Importance: Model vs Permutation', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_predictions_vs_actual(y_true, y_pred, figsize=(10, 6)):
    """
    Plot predicted vs actual values.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        figsize: Figure size
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Scatter plot
    ax1.scatter(y_true, y_pred, alpha=0.5)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    ax1.set_xlabel('Actual')
    ax1.set_ylabel('Predicted')
    ax1.set_title('Predictions vs Actual')
    ax1.grid(True, alpha=0.3)
    
    # Residuals
    residuals = y_true - y_pred
    ax2.scatter(y_pred, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--', lw=2)
    ax2.set_xlabel('Predicted')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
