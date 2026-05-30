#!/usr/bin/env python
"""
Main CLI entry point for the demand forecasting pipeline.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DemandForecastPipeline
from config.experiment_config import load_config, save_config


def main():
    parser = argparse.ArgumentParser(
        description='Demand Forecasting with xAI - M5 Dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_pipeline.py --store CA_1
  python scripts/run_pipeline.py --config config/config.yaml --optimize-memory
  python scripts/run_pipeline.py --store TX_1 --save-dir ./results/tx1/
        """
    )
    
    parser.add_argument(
        '--raw-path',
        default='./data/raw',
        help='Path to raw M5 data (default: ./data/raw)'
    )
    
    parser.add_argument(
        '--store',
        default=None,
        help='Store ID to filter (e.g., CA_1). If None, use all stores.'
    )
    
    parser.add_argument(
        '--split-date',
        default='2016-04-24',
        help='Train/test split date YYYY-MM-DD (default: 2016-04-24)'
    )
    
    parser.add_argument(
        '--optimize-memory',
        action='store_true',
        help='Apply memory optimization to dataframe'
    )
    
    parser.add_argument(
        '--num-rounds',
        type=int,
        default=1000,
        help='Number of LightGBM boosting rounds (default: 1000)'
    )
    
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=50,
        help='Early stopping rounds (default: 50)'
    )
    
    parser.add_argument(
        '--save-dir',
        default='./models/artifacts',
        help='Directory to save models and results (default: ./models/artifacts)'
    )
    
    parser.add_argument(
        '--config',
        default=None,
        help='Path to config YAML file (overrides other args)'
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    if args.config:
        config = load_config(args.config)
        raw_path = config['data']['raw_path']
        store_filter = config['data']['store_filter']
        split_date = config['data']['train_split_date']
        optimize_memory = config['data']['optimize_memory']
        num_rounds = config['model']['params']['num_rounds']
        early_stopping = config['model']['params']['early_stopping_rounds']
        save_dir = config['experiment']['output_dir']
    else:
        raw_path = args.raw_path
        store_filter = args.store
        split_date = args.split_date
        optimize_memory = args.optimize_memory
        num_rounds = args.num_rounds
        early_stopping = args.early_stopping
        save_dir = args.save_dir
    
    # Run pipeline
    print("\n" + "=" * 70)
    print("DEMAND FORECASTING PIPELINE - M5 DATASET")
    print("=" * 70)
    print(f"Raw data path: {raw_path}")
    print(f"Store filter: {store_filter or 'All stores'}")
    print(f"Split date: {split_date}")
    print(f"Memory optimization: {optimize_memory}")
    print("=" * 70 + "\n")
    
    try:
        pipeline = DemandForecastPipeline(raw_path, split_date, store_filter)
        metrics, results = pipeline.run_full_pipeline(
            optimize_memory=optimize_memory,
            model_name='lightgbm',
            num_rounds=num_rounds,
            early_stopping=early_stopping
        )
        
        # Save results
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        model_path = f"{save_dir}/lightgbm_{store_filter or 'all'}.pkl"
        from src.utils.helpers import save_pickle
        save_pickle(pipeline.model.get_model(), model_path)
        
        results_path = f"{save_dir}/predictions_{store_filter or 'all'}.parquet"
        results.to_parquet(results_path)
        
        print(f"\n✓ Model saved to: {model_path}")
        print(f"✓ Results saved to: {results_path}")
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Pipeline failed with error:")
        print(f"  {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
