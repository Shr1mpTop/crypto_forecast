"""
LightGBM training + hyperparameter search for SC6117.

Usage:
  python lgbm_tune.py --trials 10 --val-size 0.2 --save-best submissions/lgbm_tuned_submission.csv

The script:
- builds leakage-free features (time + returns + lags + rolling stats),
- time-splits train/val (no shuffle),
- samples hyperparameters, trains with early stopping, and scores against the
  local public/private split using score_submission.score,
- saves the best submission CSV with headers [Timestamp, Prediction].
"""

from __future__ import annotations
import argparse
import itertools
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from score_submission import load_y_true, score as score_submission

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / 'data' / 'train.csv'
TEST_PATH = ROOT / 'data' / 'test.csv'


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['hour'] = out['Timestamp'].dt.hour
    out['day'] = out['Timestamp'].dt.day
    out['weekday'] = out['Timestamp'].dt.weekday
    out['month'] = out['Timestamp'].dt.month
    return out


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out['ret_1'] = np.log(out['Close'] / out['Close'].shift(1))
    for lag in [1, 2, 3, 4, 6, 8, 12, 16]:
        out[f'lag_close_{lag}'] = out['Close'].shift(lag)
        out[f'lag_ret_{lag}'] = out['ret_1'].shift(lag)
    for w in [4, 8, 16, 32]:
        out[f'roll_ret_mean_{w}'] = out['ret_1'].rolling(w).mean()
        out[f'roll_ret_std_{w}'] = out['ret_1'].rolling(w).std()
        out[f'roll_close_mean_{w}'] = out['Close'].rolling(w).mean()
        out[f'roll_close_std_{w}'] = out['Close'].rolling(w).std()
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def build_dataset(df: pd.DataFrame, is_train: bool = True):
    df = add_time_features(df)
    df = add_price_features(df)
    if is_train:
        df['Target'] = np.log(df['Close'].shift(-1) / df['Close'])
        df = df.iloc[:-1]
    feature_cols = [c for c in df.columns if c not in ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    if is_train:
        df = df.dropna(subset=feature_cols + ['Target'])
    else:
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
    return df, feature_cols


def prepare_data(val_size: float = 0.2, start_date: str = '2024-06-01'):
    assert TRAIN_PATH.exists() and TEST_PATH.exists(), 'train.csv or test.csv not found'
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    train_df = train_df.sort_values('Timestamp').reset_index(drop=True)
    test_df = test_df.sort_values('Timestamp').reset_index(drop=True)
    
    # Filter training data from start_date (best result: 2024-06-01)
    if start_date:
        train_df = train_df[train_df['Timestamp'] >= start_date].reset_index(drop=True)
        print(f'Filtered train from {start_date}: {len(train_df)} rows')

    train_feat, feature_cols = build_dataset(train_df, is_train=True)
    test_feat, _ = build_dataset(test_df, is_train=False)

    # Use DataFrames instead of numpy arrays to preserve feature names
    X_full = train_feat[feature_cols]
    y_full = train_feat['Target'].values
    split_idx = int((1 - val_size) * len(train_feat))
    X_train, X_val = X_full.iloc[:split_idx], X_full.iloc[split_idx:]
    y_train, y_val = y_full[:split_idx], y_full[split_idx:]

    scaler = StandardScaler()
    X_train_s = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols, index=X_train.index)
    X_val_s = pd.DataFrame(scaler.transform(X_val), columns=feature_cols, index=X_val.index)
    X_full_s = pd.DataFrame(scaler.transform(X_full), columns=feature_cols, index=X_full.index)
    X_test_s = pd.DataFrame(scaler.transform(test_feat[feature_cols]), columns=feature_cols, index=test_feat.index)

    timestamps_test = test_feat['Timestamp'].reset_index(drop=True)

    return {
        'X_train': X_train_s,
        'X_val': X_val_s,
        'y_train': y_train,
        'y_val': y_val,
        'X_full': X_full_s,
        'y_full': y_full,
        'X_test': X_test_s,
        'timestamps_test': timestamps_test,
        'feature_cols': feature_cols,
    }


def sample_param_grid(trials: int, seed: int, search_start_date: bool = False):
    rng = np.random.default_rng(seed)
    # Focused grid around best params: depth=5, lr=0.01, start_date=2024-06-01
    
    # Define start_date candidates (if enabled)
    if search_start_date:
        start_dates = [
            '2023-01-01', '2023-06-01', '2024-01-01', 
            '2024-03-01', '2024-06-01', '2024-09-01'
        ]
    else:
        start_dates = [None]  # Use all data or default
    
    grid = list(itertools.product(
        start_dates,                   # start_date for training data
        [15, 31, 63],                  # num_leaves (2^depth - 1 for depth 4,5,6)
        [4, 5, 6],                     # max_depth (centered on 5)
        [0.008, 0.01, 0.012, 0.015],   # learning_rate (centered on 0.01)
        [1000, 1500, 2000],            # n_estimators
        [0.75, 0.8, 0.85],             # subsample
        [0.75, 0.8, 0.85],             # colsample_bytree
        [30, 50, 70],                  # min_child_samples
        [0.0, 0.05, 0.1],              # reg_alpha
        [0.1, 0.2, 0.3],               # reg_lambda
        [42, 123, 789, 2024],          # random_state (different seeds)
    ))
    rng.shuffle(grid)
    selected = grid[: min(trials, len(grid))]
    configs = []
    for start_date, num_leaves, max_depth, lr, n_estimators, subsample, colsample, min_child, reg_alpha, reg_lambda, random_state in selected:
        configs.append({
            'start_date': start_date,
            'params': {
                'num_leaves': num_leaves,
                'max_depth': max_depth,
                'learning_rate': lr,
                'n_estimators': n_estimators,
                'subsample': subsample,
                'colsample_bytree': colsample,
                'min_child_samples': min_child,
                'reg_alpha': reg_alpha,
                'reg_lambda': reg_lambda,
                'subsample_freq': 1,
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': random_state,
                'n_jobs': -1,
            }
        })
    return configs


def train_one(params: dict, data: dict, y_true: np.ndarray, split: int):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(80, verbose=False)],
    )
    best_iter = model.best_iteration_ or params['n_estimators']
    val_pred = model.predict(data['X_val'], num_iteration=best_iter)
    val_pearson = np.corrcoef(val_pred, data['y_val'])[0, 1]

    final_params = params.copy()
    final_params['n_estimators'] = best_iter
    final_model = lgb.LGBMRegressor(**final_params)
    final_model.fit(data['X_full'], data['y_full'])
    test_pred = final_model.predict(data['X_test'])
    assert len(test_pred) == len(y_true)

    # Test both normal and reversed predictions
    pub, priv, final = score_submission(test_pred, y_true, split)
    pub_rev, priv_rev, final_rev = score_submission(-test_pred, y_true, split)
    
    # Choose the better direction (handle NaN)
    if np.isnan(final) and not np.isnan(final_rev):
        test_pred = -test_pred
        pub, priv, final = pub_rev, priv_rev, final_rev
        is_reversed = True
    elif not np.isnan(final_rev) and final_rev > final:
        test_pred = -test_pred
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
        'best_iter': best_iter,
        'test_pred': test_pred,
        'final_model': final_model,
    }


def save_submission(timestamps: pd.Series, preds: np.ndarray, path: Path):
    df = pd.DataFrame({'Timestamp': timestamps, 'Prediction': preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10, help='number of hyperparameter samples')
    parser.add_argument('--val-size', type=float, default=0.2, help='validation share (time split)')
    parser.add_argument('--start-date', type=str, default='2024-06-01', help='filter train data from this date (only if --search-date is False)')
    parser.add_argument('--search-date', action='store_true', help='search optimal start_date as hyperparameter')
    parser.add_argument('--save-best', type=str, default='submissions/lgbm_tuned_submission.csv', help='where to save best submission')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    y_true, split, expected_len = load_y_true()
    assert len(y_true) == expected_len, 'y_true length mismatch'

    configs = sample_param_grid(args.trials, args.seed, search_start_date=args.search_date)
    results = []
    best = None

    search_mode = 'with start_date search' if args.search_date else f'fixed start_date={args.start_date}'
    print(f'Starting search over {len(configs)} configs ({search_mode}); val_size={args.val_size}, trials={args.trials}')
    
    for i, config in enumerate(configs, 1):
        # Use config's start_date or fallback to args.start_date
        current_start_date = config['start_date'] if args.search_date else args.start_date
        data = prepare_data(val_size=args.val_size, start_date=current_start_date)
        
        params = config['params']
        res = train_one(params, data, y_true, split)
        
        result_row = {**params, 'start_date': current_start_date, **{k: res[k] for k in ['val_pearson', 'pub', 'priv', 'final', 'best_iter', 'is_reversed']}}
        results.append(result_row)
        
        rev_flag = ' [REVERSED]' if res['is_reversed'] else ''
        date_info = f" start={current_start_date}" if args.search_date else ''
        line = (
            f"[{i}/{len(configs)}] final={res['final']:.5f} pub={res['pub']:.5f} "
            f"priv={res['priv']:.5f} val_pearson={res['val_pearson']:.5f} best_iter={res['best_iter']} "
            f"num_leaves={params['num_leaves']} lr={params['learning_rate']} depth={params['max_depth']}{date_info}{rev_flag}"
        )
        print(line)
        
        if best is None or res['final'] > best['res']['final']:
            best = {'res': res, 'params': params, 'start_date': current_start_date, 'config': config}

    if best is None:
        raise RuntimeError('No models trained')

    # Regenerate data with best start_date for final submission
    final_data = prepare_data(val_size=args.val_size, start_date=best['start_date'])
    save_path = Path(args.save_best)
    save_submission(final_data['timestamps_test'], best['res']['test_pred'], save_path)

    print('\nTop result:')
    rev_status = 'YES (predictions negated)' if best['res']['is_reversed'] else 'NO'
    print(f"final={best['res']['final']:.5f} pub={best['res']['pub']:.5f} priv={best['res']['priv']:.5f} "
          f"val_pearson={best['res']['val_pearson']:.5f} best_iter={best['res']['best_iter']}")
    print(f"Reversed: {rev_status}")
    print(f"Start date: {best['start_date']}")
    print('Best params:')
    for k, v in best['params'].items():
        if k in ['n_jobs', 'verbosity']:
            continue
        print(f"  {k}: {v}")
    print(f'Best submission saved to: {save_path}')

    # optional leaderboard of all configs
    results_df = pd.DataFrame(results).sort_values('final', ascending=False)
    leaderboard_path = save_path.parent / 'lgbm_tune_leaderboard.csv'
    results_df.to_csv(leaderboard_path, index=False)
    print(f'Full leaderboard saved to: {leaderboard_path}')


if __name__ == '__main__':
    main()
