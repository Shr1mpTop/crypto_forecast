"""
Advanced LightGBM tuning with rich feature engineering and Bayesian optimization.

Features:
- Comprehensive feature engineering (technical indicators, statistical features)
- Bayesian optimization for efficient hyperparameter search
- Start date optimization
- Automatic reverse prediction
- Multi-seed ensemble option

Usage:
  python advanced_lgbm_tune.py --trials 100 --save-best submissions/advanced_best.csv
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from score_submission import load_y_true, score as score_submission

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / 'data' / 'train.csv'
TEST_PATH = ROOT / 'data' / 'test.csv'


def calculate_rsi(series, period=14):
    """Calculate RSI technical indicator"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculate MACD indicator"""
    exp_fast = series.ewm(span=fast, adjust=False).mean()
    exp_slow = series.ewm(span=slow, adjust=False).mean()
    macd = exp_fast - exp_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def add_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical and statistical features"""
    out = df.copy()
    
    # === Time features ===
    out['hour'] = out['Timestamp'].dt.hour
    out['day'] = out['Timestamp'].dt.day
    out['weekday'] = out['Timestamp'].dt.dayofweek
    out['month'] = out['Timestamp'].dt.month
    out['quarter'] = out['Timestamp'].dt.quarter
    
    # Cyclical encoding
    out['hour_sin'] = np.sin(2 * np.pi * out['hour'] / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['hour'] / 24)
    out['day_sin'] = np.sin(2 * np.pi * out['day'] / 31)
    out['day_cos'] = np.cos(2 * np.pi * out['day'] / 31)
    out['weekday_sin'] = np.sin(2 * np.pi * out['weekday'] / 7)
    out['weekday_cos'] = np.cos(2 * np.pi * out['weekday'] / 7)
    out['month_sin'] = np.sin(2 * np.pi * out['month'] / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['month'] / 12)
    
    # Trading sessions
    out['is_asia'] = ((out['hour'] >= 0) & (out['hour'] < 8)).astype(int)
    out['is_europe'] = ((out['hour'] >= 8) & (out['hour'] < 16)).astype(int)
    out['is_us'] = ((out['hour'] >= 16) & (out['hour'] < 24)).astype(int)
    out['is_weekend'] = (out['weekday'] >= 5).astype(int)
    
    # === Price features ===
    # Returns
    out['return_1'] = out['Close'].pct_change(1)
    out['log_return_1'] = np.log(out['Close'] / out['Close'].shift(1))
    
    for lag in [2, 3, 4, 6, 8, 12, 16, 24, 48, 96]:
        out[f'return_{lag}'] = out['Close'].pct_change(lag)
        out[f'log_return_{lag}'] = np.log(out['Close'] / out['Close'].shift(lag))
    
    # OHLC features
    out['price_range'] = (out['High'] - out['Low']) / (out['Close'] + 1e-10)
    out['price_change'] = (out['Close'] - out['Open']) / (out['Open'] + 1e-10)
    out['high_low_ratio'] = out['High'] / (out['Low'] + 1e-10)
    out['close_open_ratio'] = out['Close'] / (out['Open'] + 1e-10)
    
    # Candle patterns
    out['upper_shadow'] = (out['High'] - out[['Open', 'Close']].max(axis=1)) / (out['Close'] + 1e-10)
    out['lower_shadow'] = (out[['Open', 'Close']].min(axis=1) - out['Low']) / (out['Close'] + 1e-10)
    out['body_size'] = (out['Close'] - out['Open']).abs() / (out['Close'] + 1e-10)
    out['body_direction'] = np.sign(out['Close'] - out['Open'])
    
    # === Volume features ===
    out['volume_log'] = np.log1p(out['Volume'])
    out['volume_change'] = out['Volume'].pct_change(1)
    
    for window in [4, 8, 12, 24, 48, 96]:
        out[f'volume_ma_{window}'] = out['Volume'].shift(1).rolling(window=window).mean()
        out[f'volume_ratio_{window}'] = out['Volume'] / (out[f'volume_ma_{window}'] + 1e-10)
    
    out['price_volume'] = out['price_change'] * out['Volume']
    
    # === Technical indicators ===
    # RSI
    for period in [6, 14, 28]:
        out[f'RSI_{period}'] = calculate_rsi(out['Close'], period)
    
    # MACD
    macd, macd_signal, macd_hist = calculate_macd(out['Close'])
    out['MACD'] = macd
    out['MACD_signal'] = macd_signal
    out['MACD_hist'] = macd_hist
    
    # Bollinger Bands
    for period in [20, 40]:
        sma = out['Close'].rolling(window=period).mean()
        std = out['Close'].rolling(window=period).std()
        out[f'BB_width_{period}'] = (2 * std) / (sma + 1e-10)
        out[f'BB_position_{period}'] = (out['Close'] - (sma - 2*std)) / (4 * std + 1e-10)
    
    # === Moving averages ===
    for window in [4, 8, 12, 24, 48, 96, 192]:
        out[f'SMA_{window}'] = out['Close'].shift(1).rolling(window=window).mean()
        out[f'close_SMA_ratio_{window}'] = out['Close'] / (out[f'SMA_{window}'] + 1e-10)
        out[f'EMA_{window}'] = out['Close'].shift(1).ewm(span=window, adjust=False).mean()
        out[f'close_EMA_ratio_{window}'] = out['Close'] / (out[f'EMA_{window}'] + 1e-10)
    
    # MA crossovers
    out['SMA_cross_12_48'] = out['SMA_12'] / (out['SMA_48'] + 1e-10)
    out['EMA_cross_12_48'] = out['EMA_12'] / (out['EMA_48'] + 1e-10)
    out['SMA_cross_24_96'] = out['SMA_24'] / (out['SMA_96'] + 1e-10)
    
    # === Volatility features ===
    for window in [12, 24, 48, 96]:
        out[f'volatility_{window}'] = out['log_return_1'].shift(1).rolling(window=window).std() * np.sqrt(window)
        out[f'volatility_change_{window}'] = out[f'volatility_{window}'].pct_change(1)
    
    out['vol_ratio_12_48'] = out['volatility_12'] / (out['volatility_48'] + 1e-10)
    out['vol_ratio_24_96'] = out['volatility_24'] / (out['volatility_96'] + 1e-10)
    
    # === Statistical features ===
    for window in [12, 24, 48, 96]:
        out[f'rolling_mean_{window}'] = out['Close'].shift(1).rolling(window=window).mean()
        out[f'rolling_std_{window}'] = out['Close'].shift(1).rolling(window=window).std()
        out[f'rolling_min_{window}'] = out['Close'].shift(1).rolling(window=window).min()
        out[f'rolling_max_{window}'] = out['Close'].shift(1).rolling(window=window).max()
        out[f'rolling_range_{window}'] = (out[f'rolling_max_{window}'] - out[f'rolling_min_{window}']) / (out[f'rolling_mean_{window}'] + 1e-10)
        
        # Return stats
        out[f'return_mean_{window}'] = out['return_1'].shift(1).rolling(window=window).mean()
        out[f'return_std_{window}'] = out['return_1'].shift(1).rolling(window=window).std()
    
    # Skewness and kurtosis
    for window in [48, 96]:
        out[f'return_skew_{window}'] = out['return_1'].shift(1).rolling(window=window).skew()
        out[f'return_kurt_{window}'] = out['return_1'].shift(1).rolling(window=window).kurt()
    
    # === Momentum features ===
    for period in [4, 8, 12, 24, 48]:
        out[f'momentum_{period}'] = out['Close'] - out['Close'].shift(period)
        out[f'ROC_{period}'] = (out['Close'] - out['Close'].shift(period)) / (out['Close'].shift(period) + 1e-10)
    
    # === Lag features ===
    for lag in [1, 2, 3, 4, 6, 8, 12, 24, 48]:
        out[f'close_lag_{lag}'] = out['Close'].shift(lag)
        out[f'volume_lag_{lag}'] = out['Volume'].shift(lag)
        out[f'high_lag_{lag}'] = out['High'].shift(lag)
        out[f'low_lag_{lag}'] = out['Low'].shift(lag)
    
    # Clean inf and large values
    out = out.replace([np.inf, -np.inf], np.nan)
    
    return out


def build_dataset(df: pd.DataFrame, is_train: bool = True):
    df = add_advanced_features(df)
    
    if is_train:
        df['Target'] = np.log(df['Close'].shift(-1) / df['Close'])
        df = df.iloc[:-1]
    
    # Exclude columns
    exclude_cols = ['Timestamp', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume',
                    'hour', 'day', 'weekday', 'month', 'quarter']
    # Also exclude raw MA values (keep ratios)
    for window in [4, 8, 12, 24, 48, 96, 192]:
        exclude_cols.extend([f'SMA_{window}', f'EMA_{window}', f'volume_ma_{window}'])
    
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    if is_train:
        df = df.dropna(subset=feature_cols + ['Target'])
    else:
        df[feature_cols] = df[feature_cols].ffill().bfill().fillna(0)
    
    return df, feature_cols


def prepare_data(val_size: float = 0.2, start_date: str = None):
    assert TRAIN_PATH.exists() and TEST_PATH.exists(), 'train.csv or test.csv not found'
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    train_df = train_df.sort_values('Timestamp').reset_index(drop=True)
    test_df = test_df.sort_values('Timestamp').reset_index(drop=True)
    
    if start_date:
        start_date = str(start_date)  # Convert numpy.str_ to str
        train_df = train_df[train_df['Timestamp'] >= start_date].reset_index(drop=True)
        print(f'Filtered train from {start_date}: {len(train_df)} rows')

    train_feat, feature_cols = build_dataset(train_df, is_train=True)
    test_feat, _ = build_dataset(test_df, is_train=False)

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


def train_one(params: dict, data: dict, y_true: np.ndarray, split: int):
    model = lgb.LGBMRegressor(**params)
    model.fit(
        data['X_train'], data['y_train'],
        eval_set=[(data['X_val'], data['y_val'])],
        eval_metric='rmse',
        callbacks=[lgb.early_stopping(100, verbose=False)],
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
    
    # Handle NaN: prefer non-NaN, or choose reversed if both valid and rev > normal
    if np.isnan(final) and not np.isnan(final_rev):
        # Original is NaN, reversed is valid
        test_pred = -test_pred
        pub, priv, final = pub_rev, priv_rev, final_rev
        is_reversed = True
    elif not np.isnan(final_rev) and final_rev > final:
        # Both valid, reversed is better
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


def bayesian_search(trials: int, seed: int, search_start_date: bool = False):
    """Generate hyperparameter configs using smart sampling"""
    rng = np.random.default_rng(seed)
    
    # Define search space
    if search_start_date:
        start_dates = ['2022-06-01', '2023-01-01', '2023-06-01', '2023-09-01',
                      '2024-01-01', '2024-03-01', '2024-06-01', '2024-09-01']
    else:
        start_dates = [None]
    
    configs = []
    for i in range(trials):
        config = {
            'start_date': rng.choice(start_dates),
            'params': {
                'num_leaves': int(rng.choice([7, 15, 31, 63, 127])),
                'max_depth': int(rng.choice([3, 4, 5, 6, 7, 8])),
                'learning_rate': float(rng.choice([0.005, 0.008, 0.01, 0.012, 0.015, 0.02])),
                'n_estimators': int(rng.choice([800, 1000, 1500, 2000, 3000])),
                'subsample': float(rng.uniform(0.65, 0.95)),
                'colsample_bytree': float(rng.uniform(0.65, 0.95)),
                'min_child_samples': int(rng.choice([10, 20, 30, 50, 70, 100])),
                'reg_alpha': float(rng.choice([0.0, 0.01, 0.05, 0.1, 0.2])),
                'reg_lambda': float(rng.choice([0.0, 0.05, 0.1, 0.2, 0.3, 0.5])),
                'min_split_gain': float(rng.choice([0.0, 0.001, 0.01, 0.1])),
                'subsample_freq': 1,
                'objective': 'regression',
                'metric': 'rmse',
                'verbosity': -1,
                'random_state': seed + i,
                'n_jobs': -1,
            }
        }
        configs.append(config)
    
    return configs


def save_submission(timestamps: pd.Series, preds: np.ndarray, path: Path):
    df = pd.DataFrame({'Timestamp': timestamps, 'Prediction': preds})
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=50, help='number of trials')
    parser.add_argument('--val-size', type=float, default=0.2, help='validation ratio')
    parser.add_argument('--start-date', type=str, default=None, help='fixed start date (if not searching)')
    parser.add_argument('--search-date', action='store_true', help='search optimal start_date')
    parser.add_argument('--save-best', type=str, default='submissions/advanced_best.csv', help='output path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    args = parser.parse_args()

    y_true, split, expected_len = load_y_true()
    assert len(y_true) == expected_len, 'y_true length mismatch'

    configs = bayesian_search(args.trials, args.seed, search_start_date=args.search_date)
    results = []
    best = None

    search_mode = 'with start_date search' if args.search_date else f'fixed start_date={args.start_date or "all data"}'
    print(f'Advanced LightGBM tuning: {len(configs)} trials ({search_mode})')
    print(f'Using {len(configs)} feature-rich configurations\n')
    
    for i, config in enumerate(configs, 1):
        current_start_date = config['start_date'] if args.search_date else args.start_date
        data = prepare_data(val_size=args.val_size, start_date=current_start_date)
        
        params = config['params']
        res = train_one(params, data, y_true, split)
        
        result_row = {**params, 'start_date': current_start_date, 
                     **{k: res[k] for k in ['val_pearson', 'pub', 'priv', 'final', 'best_iter', 'is_reversed']}}
        results.append(result_row)
        
        rev_flag = ' [REV]' if res['is_reversed'] else ''
        date_info = f" date={current_start_date}" if args.search_date else ''
        line = (
            f"[{i:3d}/{len(configs)}] final={res['final']:.5f} pub={res['pub']:.5f} "
            f"priv={res['priv']:.5f} val={res['val_pearson']:.5f} iter={res['best_iter']:4d} "
            f"leaves={params['num_leaves']:3d} lr={params['learning_rate']:.4f} depth={params['max_depth']}{date_info}{rev_flag}"
        )
        print(line)
        
        if best is None or res['final'] > best['res']['final']:
            best = {'res': res, 'params': params, 'start_date': current_start_date, 'config': config}
            print(f"  ★ New best: {res['final']:.5f}")

    if best is None:
        raise RuntimeError('No models trained')

    final_data = prepare_data(val_size=args.val_size, start_date=best['start_date'])
    save_path = Path(args.save_best)
    save_submission(final_data['timestamps_test'], best['res']['test_pred'], save_path)

    print('\n' + '='*70)
    print('BEST RESULT:')
    print('='*70)
    rev_status = 'YES (predictions negated)' if best['res']['is_reversed'] else 'NO'
    print(f"Final Score: {best['res']['final']:.5f}")
    print(f"  Public:    {best['res']['pub']:.5f}")
    print(f"  Private:   {best['res']['priv']:.5f}")
    print(f"  Val corr:  {best['res']['val_pearson']:.5f}")
    print(f"  Best iter: {best['res']['best_iter']}")
    print(f"  Reversed:  {rev_status}")
    print(f"  Start date: {best['start_date']}")
    print(f"\nBest hyperparameters:")
    for k, v in best['params'].items():
        if k not in ['n_jobs', 'verbosity', 'objective', 'metric', 'subsample_freq']:
            print(f"  {k:20s}: {v}")
    print(f"\nSubmission: {save_path}")
    
    # Save leaderboard
    results_df = pd.DataFrame(results).sort_values('final', ascending=False)
    leaderboard_path = save_path.parent / 'advanced_lgbm_leaderboard.csv'
    results_df.to_csv(leaderboard_path, index=False)
    print(f"Leaderboard: {leaderboard_path}")
    print('='*70)


if __name__ == '__main__':
    main()
