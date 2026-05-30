#!/usr/bin/env python
"""
Model evaluation and reporting script.
"""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import DemandForecastPipeline
from src.xai.feature_importance import FeatureImportanceAnalyzer
from src.xai.visualization import plot_feature_importance, plot_predictions_vs_actual


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained model and generate reports',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/evaluate_model.py --store CA_1
  python scripts/evaluate_model.py --store TX_1 --plot
        """
    )
    
    parser.add_argument(
        '--raw-path',
        default='./data/raw',
        help='Path to raw M5 data'
    )
    
    parser.add_argument(
        '--store',
        default=None,
        help='Store ID to evaluate'
    )
    
    parser.add_argument(
        '--split-date',
        default='2016-04-24',
        help='Train/test split date'
    )
    
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        default=20,
        help='Top K features to analyze'
    )
    
    args = parser.parse_args()
    
    print("\n" + "=" * 70)
    print("MODEL EVALUATION & ANALYSIS")
    print("=" * 70 + "\n")
    
    try:
        # Run pipeline
        pipeline = DemandForecastPipeline(args.raw_path, args.split_date, args.store)
        metrics, results = pipeline.run_full_pipeline(optimize_memory=True)
        
        # Feature importance
        analyzer = FeatureImportanceAnalyzer(pipeline.model)
        importance = analyzer.get_model_importance(top_k=args.top_k)
        
        print(f"\nTop {args.top_k} Features:")
        print(importance.to_string(index=False))
        
        # Plot if requested
        if args.plot:
            print("\nGenerating visualizations...")
            plot_feature_importance(importance, top_k=args.top_k)
            plot_predictions_vs_actual(
                results[results['actual_sales'].notna()]['actual_sales'].values,
                results[results['actual_sales'].notna()]['pred'].values
            )
        
        return 0
        
    except Exception as e:
        print(f"\n✗ Evaluation failed: {type(e).__name__}: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
