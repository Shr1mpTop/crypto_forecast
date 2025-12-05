"""
Multi-model ensemble with LightGBM, XGBoost, and CatBoost.

Features:
- Trains multiple models (LGB, XGB, CatBoost)
- Automatic weight optimization
- Feature-rich engineering
- Reverse prediction handling
- Optuna hyperparameter optimization

Usage:
  python ensemble_tune.py --trials 30 --models lgb xgb cat --save-best submissions/ensemble_best.csv
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
try:
    import catboost as cb
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("Warning: CatBoost not installed. Install with: pip install catboost")

from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from scipy.optimize import minimize
from score_submission import load_y_true, score as score_submission
from advanced_lgbm_tune import build_dataset, add_advanced_features

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / 'data' / 'train.csv'
TEST_PATH = ROOT / 'data' / 'test.csv'


def prepare_data(val_size: float = 0.2, start_date: str = None):
    assert TRAIN_PATH.exists() and TEST_PATH.exists()
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    train_df = train_df.sort_values('Timestamp').reset_index(drop=True)
    test_df = test_df.sort_values('Timestamp').reset_index(drop=True)
    
    if start_date:
        start_date = str(start_date)  # Convert numpy.str_ to str
        train_df = train_df[train_df['Timestamp'] >= start_date].reset_index(drop=True)
        print(f'Filtered from {start_date}: {len(train_df)} rows')

    train_feat, feature_cols = build_dataset(train_df, is_train=True)
    test_feat, _ = build_dataset(test_df, is_train=False)

    X_full = train_feat[feature_cols]
    y_full = train_feat['Target'].values
    split_idx = int((1 - val_size) * len(train_feat))
    X_train, X_val = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=feature_cols)
    X_full_s = pd.DataFrame(scaler.transform(X_full), columns=feature_cols)
    X_test_s = pd.DataFrame(scaler.transform(test_feat[feature_cols]), columns=feature_cols)

    timestamps_test = test_feat['Timestamp'].reset_index(drop=True)

    return {
        'X_train': X_train_s, 'X_val': X_val_s,
        'y_train': y_train, 'y_val': y_val,
        'X_full': X_full_s, 'y_full': y_full,
        'X_test': X_test_s,
        'timestamps_test': timestamps_test,
    }


def train_lgb(data, params):
    """Train LightGBM model"""
    model = lgb.LGBMRegressor(**params)
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        callbacks=[lgb.early_stopping(100, verbose=False)]
    )
    best_iter = model.best_iteration_ or params['n_estimators']
    
    final_params = params.copy()
    final_params['n_estimators'] = best_iter
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(data['X_full'], data['y_full'])
    
    val_pred = model.predict(data['X_val'])
    test_pred = final_model.predict(data['X_test'])
    
    return val_pred, test_pred, best_iter


def train_xgb(data, params):
    """Train XGBoost model"""
    model = xgb.XGBRegressor(**params, early_stopping_rounds=100, eval_metric='rmse')
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        verbose=False
    )
    best_iter = model.best_iteration or params['n_estimators']
    
    final_params = params.copy()
    final_params['n_estimators'] = best_iter
    final_model = xgb.XGBRegressor(**final_params)
    final_model.fit(data['X_full'], data['y_full'])
    
    val_pred = model.predict(data['X_val'])
    test_pred = final_model.predict(data['X_test'])
    
    return val_pred, test_pred, best_iter


def train_catboost(data, params):
    """Train CatBoost model"""
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost not available")
    
    model = cb.CatBoostRegressor(**params, verbose=0, early_stopping_rounds=100)
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=(data['X_val'], data['y_val'])
    )
    best_iter = model.get_best_iteration()
    
    final_params = params.copy()
    final_params['iterations'] = best_iter
    final_model = cb.CatBoostRegressor(**final_params, verbose=0)
    final_model.fit(data['X_full'], data['y_full'])
    
    val_pred = model.predict(data['X_val'])
    test_pred = final_model.predict(data['X_test'])
    
    return val_pred, test_pred, best_iter


def optimize_weights(predictions_val, y_val):
    """Optimize ensemble weights using validation set"""
    n_models = len(predictions_val)
    
    def objective(weights):
        weights = weights / weights.sum()  # normalize
        ensemble_pred = sum(w * p for w, p in zip(weights, predictions_val))
        return -pearsonr(ensemble_pred, y_val)[0]  # minimize negative correlation
    
    # Initial equal weights
    x0 = np.ones(n_models) / n_models
    bounds = [(0, 1) for _ in range(n_models)]
    constraints = {'type': 'eq', 'fun': lambda w: w.sum() - 1}
    
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)
    return result.x / result.x.sum()


def train_ensemble(data, y_true, split, models_to_use, params_dict, seed):
    """Train ensemble of models"""
    print(f"Training ensemble: {', '.join(models_to_use)}")
    
    val_predictions = []
    test_predictions = []
    model_names = []
    best_iters = []
    
    # Train LightGBM
    if 'lgb' in models_to_use:
        print("  Training LightGBM...")
        val_pred, test_pred, best_iter = train_lgb(data, params_dict['lgb'])
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        model_names.append('LGB')
        best_iters.append(best_iter)
    
    # Train XGBoost
    if 'xgb' in models_to_use:
        print("  Training XGBoost...")
        val_pred, test_pred, best_iter = train_xgb(data, params_dict['xgb'])
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        model_names.append('XGB')
        best_iters.append(best_iter)
    
    # Train CatBoost
    if 'cat' in models_to_use and HAS_CATBOOST:
        print("  Training CatBoost...")
        val_pred, test_pred, best_iter = train_catboost(data, params_dict['cat'])
        val_predictions.append(val_pred)
        test_predictions.append(test_pred)
        model_names.append('CAT')
        best_iters.append(best_iter)
    
    # Optimize weights
    print("  Optimizing ensemble weights...")
    weights = optimize_weights(val_predictions, data['y_val'])
    
    # Create ensemble predictions
    ensemble_val = sum(w * p for w, p in zip(weights, val_predictions))
    ensemble_test = sum(w * p for w, p in zip(weights, test_predictions))
    
    val_pearson = pearsonr(ensemble_val, data['y_val'])[0]
    
    # Test reverse (handle NaN)
    pub, priv, final = score_submission(ensemble_test, y_true, split)
    pub_rev, priv_rev, final_rev = score_submission(-ensemble_test, y_true, split)
    
    if np.isnan(final) and not np.isnan(final_rev):
        ensemble_test = -ensemble_test
        pub, priv, final = pub_rev, priv_rev, final_rev
        is_reversed = True
    elif not np.isnan(final_rev) and final_rev > final:
        ensemble_test = -ensemble_test
        pub, priv, final = pub_rev, priv_rev, final_rev
        is_reversed = True
    else:
        is_reversed = False
    
    return {
        'val_pearson': val_pearson,
        'pub': pub,
        'priv': priv,
        'final': final,
        'is_reversed': is_reversed,
        'weights': dict(zip(model_names, weights)),
        'best_iters': dict(zip(model_names, best_iters)),
        'test_pred': ensemble_test,
    }


def sample_params(seed, model_type):
    """Sample hyperparameters for each model type"""
    rng = np.random.default_rng(seed)
    
    if model_type == 'lgb':
        return {
            'num_leaves': int(rng.choice([15, 31, 63, 127])),
            'max_depth': int(rng.choice([4, 5, 6, 7, 8])),
            'learning_rate': float(rng.choice([0.005, 0.01, 0.015, 0.02])),
            'n_estimators': int(rng.choice([1000, 2000, 3000])),
            'subsample': float(rng.uniform(0.7, 0.9)),
            'colsample_bytree': float(rng.uniform(0.7, 0.9)),
            'min_child_samples': int(rng.choice([20, 50, 100])),
            'reg_alpha': float(rng.choice([0.0, 0.05, 0.1])),
            'reg_lambda': float(rng.choice([0.1, 0.2, 0.3])),
            'random_state': seed,
            'n_jobs': -1,
            'verbosity': -1,
        }
    
    elif model_type == 'xgb':
        return {
            'max_depth': int(rng.choice([4, 5, 6, 7, 8])),
            'learning_rate': float(rng.choice([0.005, 0.01, 0.015, 0.02])),
            'n_estimators': int(rng.choice([1000, 2000, 3000])),
            'subsample': float(rng.uniform(0.7, 0.9)),
            'colsample_bytree': float(rng.uniform(0.7, 0.9)),
            'min_child_weight': int(rng.choice([1, 3, 5])),
            'gamma': float(rng.choice([0.0, 0.1, 0.2])),
            'reg_alpha': float(rng.choice([0.0, 0.05, 0.1])),
            'reg_lambda': float(rng.choice([0.5, 1.0, 2.0])),
            'random_state': seed,
            'n_jobs': -1,
        }
    
    elif model_type == 'cat':
        return {
            'iterations': int(rng.choice([1000, 2000, 3000])),
            'depth': int(rng.choice([4, 5, 6, 7, 8])),
            'learning_rate': float(rng.choice([0.01, 0.02, 0.03, 0.05])),
            'l2_leaf_reg': float(rng.choice([1, 3, 5, 7])),
            'random_seed': seed,
            'thread_count': -1,
        }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='number of trials')
    parser.add_argument('--models', nargs='+', default=['lgb', 'xgb'], 
                       choices=['lgb', 'xgb', 'cat'], help='models to ensemble')
    parser.add_argument('--val-size', type=float, default=0.2)
    parser.add_argument('--start-date', type=str, default='2023-01-01')
    parser.add_argument('--save-best', type=str, default='submissions/ensemble_best.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    if 'cat' in args.models and not HAS_CATBOOST:
        print("CatBoost not available, removing from ensemble")
        args.models = [m for m in args.models if m != 'cat']

    y_true, split, expected_len = load_y_true()
    
    print(f"\n{'='*70}")
    print(f"ENSEMBLE OPTIMIZATION")
    print(f"{'='*70}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Trials: {args.trials}")
    print(f"Start date: {args.start_date}\n")
    
    data = prepare_data(val_size=args.val_size, start_date=args.start_date)
    
    results = []
    best = None
    
    for trial in range(args.trials):
        print(f"\n[Trial {trial+1}/{args.trials}]")
        
        # Sample parameters for each model
        params_dict = {}
        for model_type in args.models:
            params_dict[model_type] = sample_params(args.seed + trial, model_type)
        
        # Train ensemble
        res = train_ensemble(data, y_true, split, args.models, params_dict, args.seed + trial)
        
        results.append({
            **res,
            'trial': trial + 1,
            'params': params_dict,
        })
        
        rev_flag = '[REV]' if res['is_reversed'] else ''
        print(f"  Result: final={res['final']:.5f} pub={res['pub']:.5f} priv={res['priv']:.5f} {rev_flag}")
        print(f"  Weights: {res['weights']}")
        print(f"  Iters: {res['best_iters']}")
        
        if best is None or res['final'] > best['final']:
            best = res
            best['trial'] = trial + 1
            print(f"  ★★★ New best: {res['final']:.5f}")
    
    # Save best submission
    save_path = Path(args.save_best)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    submission = pd.DataFrame({
        'Timestamp': data['timestamps_test'],
        'Prediction': best['test_pred']
    })
    submission.to_csv(save_path, index=False)
    
    print(f"\n{'='*70}")
    print("BEST ENSEMBLE RESULT")
    print(f"{'='*70}")
    print(f"Trial: {best['trial']}")
    print(f"Final Score: {best['final']:.5f}")
    print(f"  Public:  {best['pub']:.5f}")
    print(f"  Private: {best['priv']:.5f}")
    print(f"  Val corr: {best['val_pearson']:.5f}")
    print(f"  Reversed: {'YES' if best['is_reversed'] else 'NO'}")
    print(f"\nModel Weights:")
    for name, weight in best['weights'].items():
        print(f"  {name}: {weight:.4f}")
    print(f"\nBest Iterations:")
    for name, iter_count in best['best_iters'].items():
        print(f"  {name}: {iter_count}")
    print(f"\nSubmission saved: {save_path}")
    print(f"{'='*70}\n")


if __name__ == '__main__':
    main()
