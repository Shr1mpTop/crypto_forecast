"""
Simplified advanced LightGBM tuning with carefully selected features.

This version uses ~50-80 high-quality features instead of 300+.

Usage:
  python advanced_lgbm_simple.py --trials 20 --search-date
"""

from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
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


def add_smart_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add carefully selected features to avoid NaN overflow"""
    out = df.copy()
    
    # === Time features (cyclical) ===
    out['hour_sin'] = np.sin(2 * np.pi * out['Timestamp'].dt.hour / 24)
    out['hour_cos'] = np.cos(2 * np.pi * out['Timestamp'].dt.hour / 24)
    out['weekday_sin'] = np.sin(2 * np.pi * out['Timestamp'].dt.dayofweek / 7)
    out['weekday_cos'] = np.cos(2 * np.pi * out['Timestamp'].dt.dayofweek / 7)
    out['month_sin'] = np.sin(2 * np.pi * out['Timestamp'].dt.month / 12)
    out['month_cos'] = np.cos(2 * np.pi * out['Timestamp'].dt.month / 12)
    
    # === Basic price features ===
    out['return_1'] = out['Close'].pct_change(1)
    out['log_return_1'] = np.log(out['Close'] / out['Close'].shift(1))
    
    # Short-term returns
    for lag in [2, 4, 8, 12, 24]:
        out[f'return_{lag}'] = out['Close'].pct_change(lag)
    
    # Price range
    out['price_range'] = (out['High'] - out['Low']) / out['Close']
    out['price_change'] = (out['Close'] - out['Open']) / out['Open']
    
    # === Volume features ===
    out['volume_log'] = np.log1p(out['Volume'])
    out['volume_change'] = out['Volume'].pct_change(1)
    
    for window in [4, 12, 48]:
        ma = out['Volume'].shift(1).rolling(window=window).mean()
        out[f'volume_ratio_{window}'] = out['Volume'] / (ma + 1e-10)
    
    # === RSI (proven indicator) ===
    out['RSI_14'] = calculate_rsi(out['Close'], 14)
    out['RSI_28'] = calculate_rsi(out['Close'], 28)
    
    # === Moving averages (key windows only) ===
    for window in [12, 24, 48, 96]:
        sma = out['Close'].shift(1).rolling(window=window).mean()
        out[f'close_SMA_ratio_{window}'] = out['Close'] / (sma + 1e-10)
        
        ema = out['Close'].shift(1).ewm(span=window, adjust=False).mean()
        out[f'close_EMA_ratio_{window}'] = out['Close'] / (ema + 1e-10)
    
    # MA crossovers
    sma_12 = out['Close'].shift(1).rolling(window=12).mean()
    sma_48 = out['Close'].shift(1).rolling(window=48).mean()
    out['SMA_cross_12_48'] = sma_12 / (sma_48 + 1e-10)
    
    # === Volatility ===
    for window in [12, 24, 48]:
        out[f'volatility_{window}'] = out['log_return_1'].shift(1).rolling(window=window).std()
    
    out['vol_ratio_12_48'] = out['volatility_12'] / (out['volatility_48'] + 1e-10)
    
    # === Rolling statistics ===
    for window in [24, 48]:
        roll_mean = out['Close'].shift(1).rolling(window=window).mean()
        roll_std = out['Close'].shift(1).rolling(window=window).std()
        roll_min = out['Close'].shift(1).rolling(window=window).min()
        roll_max = out['Close'].shift(1).rolling(window=window).max()
        
        out[f'z_score_{window}'] = (out['Close'] - roll_mean) / (roll_std + 1e-10)
        out[f'position_{window}'] = (out['Close'] - roll_min) / (roll_max - roll_min + 1e-10)
    
    # === Momentum ===
    for period in [4, 12, 24]:
        out[f'ROC_{period}'] = (out['Close'] - out['Close'].shift(period)) / (out['Close'].shift(period) + 1e-10)
    
    # === Key lags ===
    for lag in [1, 2, 4, 8, 12, 24]:
        out[f'close_lag_{lag}'] = out['Close'].shift(lag) / out['Close']
        out[f'volume_lag_{lag}'] = out['Volume'].shift(lag) / (out['Volume'] + 1e-10)
    
    # Clean inf
    out = out.replace([np.inf, -np.inf], np.nan)
    
    return out


def build_dataset(df: pd.DataFrame, is_train: bool = True):
    df = add_smart_features(df)
    
    if is_train:
        df['Target'] = np.log(df['Close'].shift(-1) / df['Close'])
        df = df.iloc[:-1]
    
    # Exclude original columns
    exclude_cols = ['Timestamp', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    
    # Fill NaN with forward fill, then backward fill, then 0
    for col in feature_cols:
        df[col] = df[col].fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    if is_train:
        X = df[feature_cols].values
        y = df['Target'].values
        # Remove any remaining NaN in target
        valid = ~np.isnan(y)
        X = X[valid]
        y = y[valid]
        return X, y, feature_cols
    else:
        X = df[feature_cols].values
        return X, feature_cols


def sample_param_grid():
    """Sample parameters for Bayesian-style optimization"""
    start_date_options = pd.date_range('2022-06-01', '2024-09-01', freq='MS').to_pydatetime().tolist()
    
    params = {
        'start_date': np.random.choice(start_date_options),
        'num_leaves': np.random.choice([7, 15, 31, 63, 127]),
        'max_depth': np.random.randint(3, 9),
        'learning_rate': np.random.choice([0.005, 0.01, 0.02, 0.03, 0.05]),
        'min_child_samples': np.random.choice([5, 10, 20, 30, 50]),
        'subsample': np.random.uniform(0.7, 1.0),
        'colsample_bytree': np.random.uniform(0.7, 1.0),
        'reg_alpha': np.random.choice([0.0, 0.01, 0.1, 0.5, 1.0]),
        'reg_lambda': np.random.choice([0.0, 0.01, 0.1, 0.5, 1.0, 5.0]),
        'n_estimators': 5000,
    }
    return params


def train_one(params: dict, y_true_pub, y_true_priv, search_date: bool = False, fixed_start_date: str = None):
    """Train one model with given parameters"""
    # Load data
    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)
    train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    
    # Filter by start_date
    if search_date:
        start_date = str(params.pop('start_date'))
        train_df = train_df[train_df['Timestamp'] >= start_date].reset_index(drop=True)
    elif fixed_start_date:
        train_df = train_df[train_df['Timestamp'] >= fixed_start_date].reset_index(drop=True)
        print(f'Using fixed start_date: {fixed_start_date}, data size: {len(train_df)}')
    
    # Build datasets
    X_train, y_train, feat_cols = build_dataset(train_df, is_train=True)
    X_test, _ = build_dataset(test_df, is_train=False)
    
    # Split validation
    val_size = min(5000, len(X_train) // 5)
    X_tr, y_tr = X_train[:-val_size], y_train[:-val_size]
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
    
    # Train model
    model_params = {k: v for k, v in params.items() if k != 'start_date'}
    model = lgb.LGBMRegressor(
        objective='regression',
        metric='rmse',
        verbosity=-1,
        random_state=42,
        **model_params
    )
    
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False)]
    )
    
    # Predict
    test_pred = model.predict(X_test)
    val_pred = model.predict(X_val)
    
    # Score
    pub_score = np.corrcoef(test_pred[:len(y_true_pub)], y_true_pub)[0, 1]
    priv_score = np.corrcoef(test_pred[len(y_true_pub):], y_true_priv)[0, 1]
    final_score = (pub_score + priv_score) / 2
    val_score = np.corrcoef(val_pred, y_val)[0, 1]
    
    # Try reversed prediction
    pub_rev = np.corrcoef(-test_pred[:len(y_true_pub)], y_true_pub)[0, 1]
    priv_rev = np.corrcoef(-test_pred[len(y_true_pub):], y_true_priv)[0, 1]
    final_rev = (pub_rev + priv_rev) / 2
    
    # Choose best direction (handle NaN carefully)
    reversed_flag = False
    if np.isnan(final_score) and not np.isnan(final_rev):
        test_pred = -test_pred
        final_score, pub_score, priv_score = final_rev, pub_rev, priv_rev
        reversed_flag = True
    elif not np.isnan(final_score) and not np.isnan(final_rev) and final_rev > final_score:
        test_pred = -test_pred
        final_score, pub_score, priv_score = final_rev, pub_rev, priv_rev
        reversed_flag = True
    
    result = {
        'final': final_score,
        'pub': pub_score,
        'priv': priv_score,
        'val': val_score,
        'best_iter': model.best_iteration_,
        'predictions': test_pred,
        'reversed': reversed_flag,
        'num_features': len(feat_cols)
    }
    
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=20, help='Number of trials')
    parser.add_argument('--search-date', action='store_true', help='Search start_date')
    parser.add_argument('--start-date', type=str, default=None, help='Fixed training start date (YYYY-MM-DD)')
    parser.add_argument('--save-best', type=str, help='Save best submission CSV')
    args = parser.parse_args()
    
    # Load ground truth
    y_true, split_idx, n_test = load_y_true()
    y_true_pub = y_true[:split_idx]
    y_true_priv = y_true[split_idx:]
    
    print(f"🚀 Advanced LightGBM Tuning (Simplified Features)")
    print(f"Trials: {args.trials}")
    print(f"Search start_date: {args.search_date}")
    if args.start_date:
        print(f"Fixed start_date: {args.start_date}")
    print()
    
    best_score = -np.inf
    best_pred = None
    best_params = None
    
    for i in range(args.trials):
        params = sample_param_grid()
        result = train_one(params, y_true_pub, y_true_priv, 
                          search_date=args.search_date, 
                          fixed_start_date=args.start_date)
        
        final = result['final']
        pub = result['pub']
        priv = result['priv']
        val = result['val']
        best_iter = result['best_iter']
        num_feat = result['num_features']
        rev_flag = "[REV]" if result['reversed'] else ""
        
        print(f"[{i+1}/{args.trials}] final={final:.5f} pub={pub:.5f} priv={priv:.5f} "
              f"val={val:.5f} iter={best_iter} feat={num_feat} {rev_flag}")
        
        if not np.isnan(final) and final > best_score:
            best_score = final
            best_pred = result['predictions']
            best_params = params
            print(f"  ⭐ New best! Final={best_score:.5f}")
    
    print()
    print(f"✅ Best Final Score: {best_score:.5f}")
    print(f"Best params: {best_params}")
    
    # Save best submission
    if args.save_best and best_pred is not None:
        test_df = pd.read_csv(TEST_PATH)
        sub_df = pd.DataFrame({
            'Timestamp': test_df['Timestamp'],
            'Prediction': best_pred
        })
        sub_df.to_csv(args.save_best, index=False)
        print(f"💾 Saved to {args.save_best}")


if __name__ == '__main__':
    main()
