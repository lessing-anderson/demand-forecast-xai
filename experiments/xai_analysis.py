"""
xAI analysis experiment using SHAP and LIME.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DemandForecastPipeline
from src.xai.shap_explainer import SHAPExplainer
from src.xai.lime_explainer import LIMEExplainer
from src.xai.feature_importance import FeatureImportanceAnalyzer


def run_xai_analysis_experiment(raw_data_path, store_filter=None, save_dir='./models/artifacts'):
    """
    Run comprehensive xAI analysis with SHAP and LIME.
    
    Args:
        raw_data_path: Path to raw data
        store_filter: Optional store to filter
        save_dir: Directory to save results
        
    Returns:
        Dictionary with xAI analysis results
    """
    print(f"\nxAI Analysis Experiment: {store_filter or 'all stores'}")
    print("=" * 60)
    
    # Train baseline model
    pipeline = DemandForecastPipeline(raw_data_path, store_filter=store_filter)
    pipeline.load_and_prepare_data(optimize_memory=True)
    pipeline.prepare_train_test()
    pipeline.train_model('lightgbm', num_rounds=500, early_stopping=50)
    pipeline.predict()
    
    # Initialize explainers
    print("\nInitializing explainers...")
    
    # SHAP explainer
    shap_explainer = SHAPExplainer(pipeline.model)
    shap_explainer.fit(pipeline.X_train)
    
    # LIME explainer
    lime_explainer = LIMEExplainer(pipeline.model, pipeline.X_train)
    
    # Feature importance analyzer
    importance_analyzer = FeatureImportanceAnalyzer(pipeline.model)
    
    # Get SHAP-based feature importance
    print("\nComputing SHAP-based feature importance...")
    shap_importance = shap_explainer.get_feature_importance_from_shap(
        pipeline.X_test.head(100),  # Use subset for speed
        top_k=15
    )
    
    print("Top Features (SHAP):")
    print(shap_importance.to_string(index=False))
    
    # Get model gain importance
    gain_importance = importance_analyzer.get_model_importance(top_k=15)
    
    print("\nTop Features (Gain):")
    print(gain_importance.to_string(index=False))
    
    # Save results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    shap_path = f"{save_dir}/xai_shap_importance_{store_filter or 'all'}.parquet"
    shap_importance.to_parquet(shap_path)
    
    print(f"\n✓ Saved SHAP importance to: {shap_path}")
    
    return {
        'shap_explainer': shap_explainer,
        'lime_explainer': lime_explainer,
        'importance_analyzer': importance_analyzer,
        'shap_importance': shap_importance,
        'pipeline': pipeline
    }


if __name__ == '__main__':
    raw_path = './data/raw'
    
    print("Running xAI analysis experiment...")
    results = run_xai_analysis_experiment(raw_path, store_filter='CA_1')
