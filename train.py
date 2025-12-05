"""
Unified training interface for crypto price forecasting.

This script provides a single entry point for all training methods:
- basic: Fast LightGBM with basic features (recommended for quick results)
- advanced: LightGBM with 50+ technical indicators (RSI, MACD, etc.)
- ensemble: Multi-model ensemble (LGB + XGB + CatBoost)

Usage:
  # Quick training with basic features (best for most cases)
  python train.py --method basic --trials 20
  
  # Advanced features with technical indicators
  python train.py --method advanced --trials 20
  
  # Ensemble of multiple models
  python train.py --method ensemble --trials 10 --models lgb xgb
  
  # Search for optimal training start date
  python train.py --method basic --trials 50 --search-date
  
  # Save best submission
  python train.py --method basic --trials 30 --save submissions/my_best.csv

Best known result: Final=0.08042 (method=basic, start_date=2023-01-01)
"""

import argparse
import sys
from pathlib import Path

# Import training modules
try:
    from lgbm_tune import main as train_basic
    from advanced_lgbm_simple import main as train_advanced
    from ensemble_tune import main as train_ensemble
except ImportError as e:
    print(f"Error importing training modules: {e}")
    sys.exit(1)


def print_banner():
    """Print welcome banner"""
    print("=" * 80)
    print("  CRYPTO PRICE FORECASTING - UNIFIED TRAINING INTERFACE")
    print("=" * 80)
    print()


def print_method_info(method: str):
    """Print information about selected method"""
    info = {
        'basic': """
╔══════════════════════════════════════════════════════════════════════════════╗
║ METHOD: BASIC (RECOMMENDED)                                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝
  • Fast training with fundamental features
  • Time features, price returns, lags, rolling statistics
  • Best known score: Final=0.08042
  • Training time: ~5-10 min for 20 trials
  • Best for: Quick iterations and baseline models
""",
        'advanced': """
╔══════════════════════════════════════════════════════════════════════════════╗
║ METHOD: ADVANCED                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
  • 50+ technical indicators (RSI, MACD, Bollinger Bands, volatility, momentum)
  • More features but may not improve over basic method
  • Training time: ~10-20 min for 20 trials
  • Best for: Exploring technical analysis features
""",
        'ensemble': """
╔══════════════════════════════════════════════════════════════════════════════╗
║ METHOD: ENSEMBLE                                                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
  • Combines multiple models (LightGBM, XGBoost, CatBoost)
  • Automatic weight optimization
  • Training time: ~15-30 min for 10 trials
  • Best for: Squeezing out extra performance
  • Requires: xgboost, catboost packages
"""
    }
    print(info.get(method, "Unknown method"))


def validate_args(args):
    """Validate command line arguments"""
    if args.method not in ['basic', 'advanced', 'ensemble']:
        print(f"Error: Invalid method '{args.method}'. Choose from: basic, advanced, ensemble")
        sys.exit(1)
    
    if args.trials < 1:
        print("Error: --trials must be >= 1")
        sys.exit(1)
    
    if args.method == 'ensemble':
        if not args.models:
            print("Warning: No models specified for ensemble, using default: lgb xgb")
            args.models = ['lgb', 'xgb']
        
        valid_models = {'lgb', 'xgb', 'cat'}
        if not set(args.models).issubset(valid_models):
            print(f"Error: Invalid model(s). Choose from: lgb, xgb, cat")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='Unified training interface for crypto forecasting',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training (recommended)
  python train.py --method basic --trials 20

  # Search for best start date
  python train.py --method basic --trials 50 --search-date

  # Advanced features
  python train.py --method advanced --trials 20 --search-date

  # Ensemble with LightGBM + XGBoost
  python train.py --method ensemble --trials 10 --models lgb xgb

  # Save best submission
  python train.py --method basic --trials 30 --save submissions/final.csv

Best known configuration:
  method=basic, start_date=2023-01-01, num_leaves=31, depth=6, lr=0.01
  Score: Final=0.08042 (Public=0.07420, Private=0.08664)
        """
    )
    
    # Core arguments
    parser.add_argument('--method', type=str, required=True,
                       choices=['basic', 'advanced', 'ensemble'],
                       help='Training method: basic (fast), advanced (indicators), ensemble (multi-model)')
    parser.add_argument('--trials', type=int, default=20,
                       help='Number of hyperparameter trials (default: 20)')
    
    # Data arguments
    parser.add_argument('--search-date', action='store_true',
                       help='Search for optimal training start date (2022-06 to 2024-09)')
    parser.add_argument('--start-date', type=str, default='2023-01-01',
                       help='Fixed training start date (if not searching, default: 2023-01-01)')
    parser.add_argument('--val-size', type=float, default=0.2,
                       help='Validation set ratio (default: 0.2)')
    
    # Ensemble-specific
    parser.add_argument('--models', nargs='+', choices=['lgb', 'xgb', 'cat'],
                       help='Models for ensemble (e.g., lgb xgb cat)')
    
    # Output
    parser.add_argument('--save', type=str,
                       help='Save best submission to CSV file')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Validate
    validate_args(args)
    
    # Print banner
    print_banner()
    print_method_info(args.method)
    
    # Prepare arguments for specific method
    if args.method == 'basic':
        # Call lgbm_tune.py main
        sys.argv = [
            'lgbm_tune.py',
            '--trials', str(args.trials),
            '--val-size', str(args.val_size),
            '--seed', str(args.seed),
        ]
        if args.search_date:
            sys.argv.append('--search-date')
        else:
            sys.argv.extend(['--start-date', args.start_date])
        
        if args.save:
            sys.argv.extend(['--save-best', args.save])
        else:
            sys.argv.extend(['--save-best', 'submissions/basic_best.csv'])
        
        train_basic()
    
    elif args.method == 'advanced':
        # Call advanced_lgbm_simple.py main
        sys.argv = [
            'advanced_lgbm_simple.py',
            '--trials', str(args.trials),
        ]
        if args.search_date:
            sys.argv.append('--search-date')
        
        if args.save:
            sys.argv.extend(['--save-best', args.save])
        else:
            sys.argv.extend(['--save-best', 'submissions/advanced_best.csv'])
        
        train_advanced()
    
    elif args.method == 'ensemble':
        # Call ensemble_tune.py main
        sys.argv = [
            'ensemble_tune.py',
            '--trials', str(args.trials),
            '--val-size', str(args.val_size),
            '--start-date', args.start_date,
            '--seed', str(args.seed),
        ]
        if args.models:
            sys.argv.extend(['--models'] + args.models)
        
        if args.save:
            sys.argv.extend(['--save-best', args.save])
        else:
            sys.argv.extend(['--save-best', 'submissions/ensemble_best.csv'])
        
        train_ensemble()


if __name__ == '__main__':
    main()
