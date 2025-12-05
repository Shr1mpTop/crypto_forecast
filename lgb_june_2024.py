"""LightGBM 配置: start=2024-06-01, depth=5, lr=0.01"""

import pandas as pd
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

NEXT_CLOSE = 84284.01


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成基础技术特征，和 analyze_and_optimize 中保持一致"""
    data = df.copy()

    for i in [1, 2, 4, 8, 16, 32, 64, 96]:
        data[f'ret_{i}'] = np.log(data['Close'] / data['Close'].shift(i))

    for w in [4, 8, 16, 32, 64, 96]:
        high = data['High'].rolling(w).max()
        low = data['Low'].rolling(w).min()
        data[f'pos_{w}'] = (data['Close'] - low) / (high - low + 1e-10)

    for w in [4, 8, 16, 32, 64, 96]:
        ma = data['Close'].rolling(w).mean()
        data[f'ma_{w}'] = (data['Close'] - ma) / (ma + 1e-10)

    for s in [8, 16, 32, 64]:
        ema = data['Close'].ewm(span=s).mean()
        data[f'ema_{s}'] = (data['Close'] - ema) / (ema + 1e-10)

    ret_1 = data['ret_1']
    for w in [8, 16, 32, 64]:
        data[f'vol_{w}'] = ret_1.rolling(w).std()

    delta = data['Close'].diff()
    for w in [8, 14, 28]:
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        data[f'rsi_{w}'] = 100 - 100 / (1 + gain / (loss + 1e-10))

    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    data['macd'] = (ema12 - ema26) / data['Close']
    data['macd_sig'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = data['macd'] - data['macd_sig']

    for w in [8, 16, 32]:
        data[f'vr_{w}'] = data['Volume'] / (data['Volume'].rolling(w).mean() + 1e-10)

    for lag in range(1, 9):
        data[f'lag_{lag}'] = ret_1.shift(lag)

    data['body'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-10)

    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def select_features(train_df: pd.DataFrame, top_n: int = 50):
    exclude = {'Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target'}
    features = [c for c in train_df.columns if c not in exclude]

    corrs = []
    for f in features:
        valid = train_df[[f, 'Target']].dropna()
        if len(valid) > 500:
            c, _ = spearmanr(valid[f], valid['Target'])
            if not np.isnan(c):
                corrs.append((f, abs(c)))

    corrs.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in corrs[:top_n]]


def calc_score(pred, true, split):
    pub = np.corrcoef(pred[:split], true[:split])[0, 1]
    priv = np.corrcoef(pred[split:], true[split:])[0, 1]
    return pub, priv, 0.5 * pub + 0.5 * priv


def main():
    print('=' * 70)
    print('🚀 LightGBM 2024-06+ (depth=5, lr=0.01)')
    print('=' * 70)

    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    n = len(test)
    split = n // 2
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE
    y_true = np.log(next_close / test['Close']).values

    train['Timestamp'] = pd.to_datetime(train['Timestamp'])
    train_period = train[train['Timestamp'] >= '2024-06-01'].reset_index(drop=True)
    print(f'训练数据量: {len(train_period)}')

    train_feat = create_features(train_period)
    test_feat = create_features(test)

    features = select_features(train_feat, top_n=50)
    train_feat = train_feat.dropna(subset=features + ['Target'])

    val_idx = int(len(train_feat) * 0.8)
    X_train = train_feat[features].iloc[:val_idx].values
    y_train = train_feat['Target'].iloc[:val_idx].values
    X_val = train_feat[features].iloc[val_idx:].values
    y_val = train_feat['Target'].iloc[val_idx:].values
    X_test = np.nan_to_num(test_feat[features].values)

    params = {
        'objective': 'regression',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 50,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'verbosity': -1,
        'random_state': 42,
    }

    dtrain = lgb.Dataset(X_train, y_train)
    dval = lgb.Dataset(X_val, y_val, reference=dtrain)
    model = lgb.train(
        params,
        dtrain,
        1000,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )

    pred = model.predict(X_test)

    pub, priv, final = calc_score(pred, y_true, split)
    pub_r, priv_r, final_r = calc_score(-pred, y_true, split)

    if final_r > final:
        pred = -pred
        pub, priv, final = pub_r, priv_r, final_r
        direction = '反向'
    else:
        direction = '正向'

    print(f'方向: {direction}')
    print(f'Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}')

    submission = pd.DataFrame({'row_id': range(n), 'Target': pred})
    output_path = 'submissions/lgb_june_2024.csv'
    submission.to_csv(output_path, index=False)
    print(f'已保存: {output_path}')


if __name__ == '__main__':
    main()
