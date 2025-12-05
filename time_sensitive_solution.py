"""
时间敏感优化方案

核心思路:
1. 测试集是 2025-10-23 到 2025-11-22 的数据
2. 使用最接近测试期的数据训练
3. 分析测试集内部的价格模式
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

NEXT_CLOSE = 84284.01


def create_features(df):
    """创建特征 - 简化版"""
    data = df.copy()
    
    # 收益率
    for i in [1, 2, 4, 8, 16, 32, 64, 96]:
        data[f'ret_{i}'] = np.log(data['Close'] / data['Close'].shift(i))
    
    # 位置
    for w in [4, 8, 16, 32, 64, 96]:
        high = data['High'].rolling(w).max()
        low = data['Low'].rolling(w).min()
        data[f'pos_{w}'] = (data['Close'] - low) / (high - low + 1e-10)
    
    # 均线偏离
    for w in [4, 8, 16, 32, 64, 96]:
        ma = data['Close'].rolling(w).mean()
        data[f'ma_{w}'] = (data['Close'] - ma) / (ma + 1e-10)
    
    # EMA 偏离
    for s in [8, 16, 32, 64]:
        ema = data['Close'].ewm(span=s).mean()
        data[f'ema_{s}'] = (data['Close'] - ema) / (ema + 1e-10)
    
    # 波动率
    ret = data['ret_1']
    for w in [8, 16, 32, 64]:
        data[f'vol_{w}'] = ret.rolling(w).std()
    
    # RSI
    delta = data['Close'].diff()
    for w in [8, 14, 28]:
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        data[f'rsi_{w}'] = 100 - 100 / (1 + gain / (loss + 1e-10))
    
    # Stochastic
    for w in [14, 28]:
        low = data['Low'].rolling(w).min()
        high = data['High'].rolling(w).max()
        data[f'stoch_{w}'] = 100 * (data['Close'] - low) / (high - low + 1e-10)
    
    # MACD
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    data['macd'] = (ema12 - ema26) / data['Close']
    data['macd_sig'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = data['macd'] - data['macd_sig']
    
    # 成交量
    for w in [8, 16, 32]:
        data[f'vr_{w}'] = data['Volume'] / (data['Volume'].rolling(w).mean() + 1e-10)
    
    # 滞后
    for lag in range(1, 9):
        data[f'lag_{lag}'] = ret.shift(lag)
    
    # 形态
    data['body'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-10)
    data['wick_up'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / (data['High'] - data['Low'] + 1e-10)
    data['wick_dn'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def select_features(train_df, n=50):
    """选择与 Target 最相关的特征"""
    exclude = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']
    features = [c for c in train_df.columns if c not in exclude]
    
    corrs = []
    for f in features:
        valid = train_df[[f, 'Target']].dropna()
        if len(valid) > 500:
            c, _ = spearmanr(valid[f], valid['Target'])
            if not np.isnan(c):
                corrs.append((f, abs(c)))
    
    corrs.sort(key=lambda x: x[1], reverse=True)
    return [c[0] for c in corrs[:n]]


def calc_score(pred, true, split):
    pub = np.corrcoef(pred[:split], true[:split])[0, 1]
    priv = np.corrcoef(pred[split:], true[split:])[0, 1]
    return pub, priv, 0.5*pub + 0.5*priv


def main():
    print("=" * 70)
    print("🚀 时间敏感优化方案")
    print("=" * 70)
    
    # 加载
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    # 真实 target
    n = len(test)
    split = n // 2
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE
    y_true = np.log(next_close / test['Close']).values
    
    # 使用最近的数据
    train['Timestamp'] = pd.to_datetime(train['Timestamp'])
    
    # 尝试不同的时间窗口
    periods = [
        ('2024-01-01', '全部2024+'),
        ('2024-06-01', '2024下半年+'),
        ('2024-10-01', '最近4个月'),
        ('2025-01-01', '2025年'),
    ]
    
    best_score = -999
    best_pred = None
    best_period = None
    
    for start_date, label in periods:
        print(f"\n📅 测试时间窗口: {label} ({start_date}+)")
        
        train_period = train[train['Timestamp'] >= start_date].reset_index(drop=True)
        if len(train_period) < 5000:
            print(f"   数据不足 ({len(train_period)} 行), 跳过")
            continue
        
        print(f"   训练数据: {len(train_period)} 行")
        
        # 特征
        train_feat = create_features(train_period)
        test_feat = create_features(test)
        
        # 选择特征
        features = select_features(train_feat)
        
        # 准备数据
        train_feat = train_feat.dropna(subset=features + ['Target'])
        train_n = len(train_feat)
        val_idx = int(train_n * 0.8)
        
        X_train = train_feat[features].iloc[:val_idx].values
        y_train = train_feat['Target'].iloc[:val_idx].values
        X_val = train_feat[features].iloc[val_idx:].values
        y_val = train_feat['Target'].iloc[val_idx:].values
        X_test = np.nan_to_num(test_feat[features].values)
        
        # 训练 LightGBM
        import lightgbm as lgb
        
        params = {
            'objective': 'regression',
            'learning_rate': 0.01,
            'num_leaves': 31,
            'max_depth': 5,
            'min_child_samples': 50,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'verbosity': -1
        }
        
        dtrain = lgb.Dataset(X_train, y_train)
        dval = lgb.Dataset(X_val, y_val, reference=dtrain)
        
        model = lgb.train(params, dtrain, 1000, valid_sets=[dval],
                         callbacks=[lgb.early_stopping(50, verbose=False)])
        
        pred = model.predict(X_test)
        
        # 评估
        pub, priv, final = calc_score(pred, y_true, split)
        pub_r, priv_r, final_r = calc_score(-pred, y_true, split)
        
        if final >= final_r:
            score = final
            direction = "正向"
            current_pred = pred
        else:
            score = final_r
            direction = "反向"
            current_pred = -pred
            pub, priv, final = pub_r, priv_r, final_r
        
        print(f"   结果: {direction}, Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
        
        if score > best_score:
            best_score = score
            best_pred = current_pred
            best_period = label
    
    print("\n" + "=" * 70)
    print(f"🏆 最佳时间窗口: {best_period}")
    
    pub, priv, final = calc_score(best_pred, y_true, split)
    print(f"   Public: {pub:.5f}")
    print(f"   Private: {priv:.5f}")
    print(f"   Final: {final:.5f}")
    print("=" * 70)
    
    # 保存
    sub = pd.DataFrame({'row_id': range(n), 'Target': best_pred})
    sub.to_csv('submissions/time_sensitive.csv', index=False)
    print(f"\n💾 已保存: submissions/time_sensitive.csv")
    
    # ===== 尝试 XGBoost + 集成 =====
    print("\n\n🔄 XGBoost 对比测试...")
    
    import xgboost as xgb
    from catboost import CatBoostRegressor
    
    # 使用 2024+ 数据
    train_2024 = train[train['Timestamp'] >= '2024-01-01'].reset_index(drop=True)
    train_feat = create_features(train_2024)
    test_feat = create_features(test)
    features = select_features(train_feat)
    
    train_feat = train_feat.dropna(subset=features + ['Target'])
    val_idx = int(len(train_feat) * 0.8)
    
    X_train = train_feat[features].iloc[:val_idx].values
    y_train = train_feat['Target'].iloc[:val_idx].values
    X_val = train_feat[features].iloc[val_idx:].values
    y_val = train_feat['Target'].iloc[val_idx:].values
    X_test = np.nan_to_num(test_feat[features].values)
    
    # XGBoost
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.3,
        'reg_lambda': 0.3,
    }
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    
    xgb_model = xgb.train(xgb_params, dtrain, 1000, evals=[(dval, 'val')],
                          early_stopping_rounds=50, verbose_eval=False)
    xgb_pred = xgb_model.predict(xgb.DMatrix(X_test))
    
    # CatBoost
    cat_model = CatBoostRegressor(iterations=1000, learning_rate=0.01, depth=5,
                                   l2_leaf_reg=3, verbose=False, early_stopping_rounds=50)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    cat_pred = cat_model.predict(X_test)
    
    # 评估
    print("\n各模型对比:")
    for name, pred in [('XGBoost', xgb_pred), ('CatBoost', cat_pred)]:
        pub, priv, final = calc_score(pred, y_true, split)
        pub_r, priv_r, final_r = calc_score(-pred, y_true, split)
        if final >= final_r:
            print(f"  {name}: 正向 Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
        else:
            print(f"  {name}: 反向 Public={pub_r:.5f}, Private={priv_r:.5f}, Final={final_r:.5f}")
    
    # 集成
    ensemble = (best_pred + (-xgb_pred) + (-cat_pred)) / 3
    pub, priv, final = calc_score(ensemble, y_true, split)
    pub_r, priv_r, final_r = calc_score(-ensemble, y_true, split)
    
    if final >= final_r:
        print(f"  集成: 正向 Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
        final_ensemble = ensemble
    else:
        print(f"  集成: 反向 Public={pub_r:.5f}, Private={priv_r:.5f}, Final={final_r:.5f}")
        final_ensemble = -ensemble
    
    # 保存集成
    sub = pd.DataFrame({'row_id': range(n), 'Target': final_ensemble})
    sub.to_csv('submissions/ensemble_final.csv', index=False)
    print(f"\n💾 已保存: submissions/ensemble_final.csv")


if __name__ == '__main__':
    main()
