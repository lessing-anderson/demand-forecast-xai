"""
Baseline LightGBM experiment.
Trains a model and evaluates performance.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DemandForecastPipeline
from src.utils.helpers import save_pickle


def run_baseline_experiment(raw_data_path, store_filter=None, save_dir='./models/artifacts'):
    """
    Run baseline LightGBM training and evaluation.
    
    Args:
        raw_data_path: Path to raw data
        store_filter: Optional store to filter
        save_dir: Directory to save results
        
    Returns:
        Dictionary with metrics and paths
    """
    pipeline = DemandForecastPipeline(raw_data_path, store_filter=store_filter)
    
    metrics, results = pipeline.run_full_pipeline(
        optimize_memory=True,
        model_name='lightgbm',
        num_rounds=1000,
        early_stopping=50
    )
    
    # Save results
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    model_path = f"{save_dir}/lightgbm_model_{store_filter or 'all'}.pkl"
    save_pickle(pipeline.model.get_model(), model_path)
    
    results_path = f"{save_dir}/predictions_{store_filter or 'all'}.parquet"
    results.to_parquet(results_path)
    
    print(f"\n✓ Model saved to: {model_path}")
    print(f"✓ Results saved to: {results_path}")
    
    return {
        'metrics': metrics,
        'results': results,
        'model_path': model_path,
        'results_path': results_path,
        'pipeline': pipeline
    }


if __name__ == '__main__':
    raw_path = './data/raw'
    
    # Run for single store
    print("Running baseline experiment for CA_1 store...")
    baseline_ca1 = run_baseline_experiment(raw_path, store_filter='CA_1')
