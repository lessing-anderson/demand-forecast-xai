"""
Feature selection and importance analysis experiment.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DemandForecastPipeline
from src.xai.feature_importance import FeatureImportanceAnalyzer


def run_feature_selection_experiment(raw_data_path, store_filter=None, save_dir='./models/artifacts'):
    """
    Run feature importance analysis.
    
    Args:
        raw_data_path: Path to raw data
        store_filter: Optional store to filter
        save_dir: Directory to save results
        
    Returns:
        Dictionary with importance analyses
    """
    print(f"\nFeature Selection Experiment: {store_filter or 'all stores'}")
    print("=" * 60)
    
    # Train baseline model
    pipeline = DemandForecastPipeline(raw_data_path, store_filter=store_filter)
    pipeline.load_and_prepare_data(optimize_memory=True)
    pipeline.prepare_train_test()
    pipeline.train_model('lightgbm', num_rounds=500, early_stopping=50)
    pipeline.predict()
    
    # Analyze feature importance
    analyzer = FeatureImportanceAnalyzer(pipeline.model)
    
    # Get gain-based importance
    gain_importance = analyzer.get_model_importance(importance_type='gain', top_k=None)
    
    # Get permutation importance
    perm_importance = analyzer.get_permutation_importance(
        pipeline.X_test,
        pipeline.test_df['sales'],  # Use actual values where available
        n_repeats=10,
        top_k=None
    )
    
    # Compare methods
    comparison = analyzer.compare_importance_methods(
        pipeline.X_test,
        pipeline.test_df['sales'],
        n_repeats=10,
        top_k=20
    )
    
    print("\nTop 10 Features (Gain Importance):")
    print(gain_importance.head(10).to_string(index=False))
    
    print("\nTop 10 Features (Permutation Importance):")
    print(perm_importance.head(10).to_string(index=False))
    
    # Save results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    gain_path = f"{save_dir}/feature_importance_gain_{store_filter or 'all'}.parquet"
    gain_importance.to_parquet(gain_path)
    
    perm_path = f"{save_dir}/feature_importance_perm_{store_filter or 'all'}.parquet"
    perm_importance.to_parquet(perm_path)
    
    comp_path = f"{save_dir}/feature_importance_comparison_{store_filter or 'all'}.parquet"
    comparison.to_parquet(comp_path)
    
    print(f"\n✓ Saved gain importance to: {gain_path}")
    print(f"✓ Saved permutation importance to: {perm_path}")
    print(f"✓ Saved comparison to: {comp_path}")
    
    return {
        'gain_importance': gain_importance,
        'perm_importance': perm_importance,
        'comparison': comparison,
        'pipeline': pipeline,
        'analyzer': analyzer
    }


if __name__ == '__main__':
    raw_path = './data/raw'
    
    print("Running feature selection experiment...")
    results = run_feature_selection_experiment(raw_path, store_filter='CA_1')
