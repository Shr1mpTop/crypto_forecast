"""
进阶优化方案 - 针对相关系数最大化

关键洞察:
1. 相关系数只关心排序，不关心绝对值
2. 可以尝试预测收益率的 rank 而非绝对值
3. 使用更多滞后特征捕捉时序结构
4. 针对 Public/Private 分别优化
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, rankdata
import warnings
warnings.filterwarnings('ignore')

NEXT_CLOSE = 84284.01


def create_advanced_features(df):
    """创建进阶特征"""
    data = df.copy()
    
    # === 收益率序列 ===
    for i in [1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32, 48, 64, 96]:
        data[f'ret_{i}'] = np.log(data['Close'] / data['Close'].shift(i))
    
    # === 动量排名特征 ===
    for w in [8, 16, 32, 64, 96]:
        ret = np.log(data['Close'] / data['Close'].shift(1))
        data[f'mom_rank_{w}'] = ret.rolling(w).apply(
            lambda x: rankdata(x)[-1] / len(x) if len(x) > 0 else 0.5, raw=True
        )
    
    # === 价格相对位置 ===
    for w in [4, 8, 16, 32, 64, 96]:
        high_roll = data['High'].rolling(w).max()
        low_roll = data['Low'].rolling(w).min()
        data[f'pos_{w}'] = (data['Close'] - low_roll) / (high_roll - low_roll + 1e-10)
    
    # === 均线系统 ===
    for w in [4, 8, 16, 32, 64, 96]:
        ma = data['Close'].rolling(w).mean()
        data[f'ma_dev_{w}'] = (data['Close'] - ma) / (ma + 1e-10)
        data[f'ma_trend_{w}'] = ma.pct_change(4)
    
    # === EMA 系统 ===
    for span in [8, 16, 32, 64]:
        ema = data['Close'].ewm(span=span).mean()
        data[f'ema_dev_{span}'] = (data['Close'] - ema) / (ema + 1e-10)
    
    # === 波动率 ===
    ret = np.log(data['Close'] / data['Close'].shift(1))
    for w in [4, 8, 16, 32, 64, 96]:
        data[f'vol_{w}'] = ret.rolling(w).std()
        data[f'vol_rank_{w}'] = data[f'vol_{w}'].rolling(96).apply(
            lambda x: rankdata(x)[-1] / len(x) if len(x) > 0 else 0.5, raw=True
        )
    
    # === RSI ===
    delta = data['Close'].diff()
    for w in [6, 14, 28]:
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        data[f'rsi_{w}'] = 100 - (100 / (1 + gain / (loss + 1e-10)))
    
    # === Stochastic ===
    for w in [14, 28]:
        low_min = data['Low'].rolling(w).min()
        high_max = data['High'].rolling(w).max()
        data[f'stoch_{w}'] = 100 * (data['Close'] - low_min) / (high_max - low_min + 1e-10)
    
    # === MACD ===
    ema12 = data['Close'].ewm(span=12).mean()
    ema26 = data['Close'].ewm(span=26).mean()
    data['macd'] = (ema12 - ema26) / data['Close']
    data['macd_signal'] = data['macd'].ewm(span=9).mean()
    data['macd_hist'] = data['macd'] - data['macd_signal']
    
    # === 成交量特征 ===
    for w in [4, 8, 16, 32]:
        vol_ma = data['Volume'].rolling(w).mean()
        data[f'vol_ratio_{w}'] = data['Volume'] / (vol_ma + 1e-10)
    
    # OBV
    obv = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    data['obv_trend'] = obv.pct_change(8)
    
    # === 收益率的滞后特征 ===
    ret = np.log(data['Close'] / data['Close'].shift(1))
    for lag in range(1, 13):
        data[f'ret_lag_{lag}'] = ret.shift(lag)
    
    # === 收益率的滚动统计 ===
    for w in [8, 16, 32, 64]:
        data[f'ret_mean_{w}'] = ret.rolling(w).mean()
        data[f'ret_skew_{w}'] = ret.rolling(w).skew()
        data[f'ret_kurt_{w}'] = ret.rolling(w).kurt()
    
    # === 价格形态 ===
    data['body'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-10)
    data['upper_wick'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / (data['High'] - data['Low'] + 1e-10)
    data['lower_wick'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    
    # === 交叉特征 ===
    data['vol_mom'] = data['vol_8'] * data['ma_dev_8']
    data['rsi_mom'] = (data['rsi_14'] - 50) * data['ma_dev_16']
    
    # 清理
    data = data.replace([np.inf, -np.inf], np.nan)
    
    return data


def feature_selection(train_df, n_features=60):
    """基于相关性的特征选择"""
    feature_cols = [c for c in train_df.columns if c not in 
                   ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']]
    
    correlations = []
    for col in feature_cols:
        valid = train_df[[col, 'Target']].dropna()
        if len(valid) > 1000:
            corr, _ = spearmanr(valid[col], valid['Target'])
            if not np.isnan(corr):
                correlations.append((col, abs(corr), corr))
    
    correlations.sort(key=lambda x: x[1], reverse=True)
    
    # 去除高度相关的特征（保留与 target 相关性最高的）
    selected = []
    for name, abs_corr, corr in correlations:
        if len(selected) >= n_features:
            break
        
        # 检查与已选特征的相关性
        is_redundant = False
        for sel_name in selected:
            valid = train_df[[name, sel_name]].dropna()
            if len(valid) > 100:
                feat_corr, _ = spearmanr(valid[name], valid[sel_name])
                if abs(feat_corr) > 0.9:
                    is_redundant = True
                    break
        
        if not is_redundant:
            selected.append(name)
    
    print(f"\n📊 选择了 {len(selected)} 个非冗余特征")
    return selected


def train_models(X_train, y_train, X_val, y_val):
    """训练多个模型"""
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    from sklearn.linear_model import Ridge
    
    models = {}
    
    # LightGBM
    lgb_params = {
        'objective': 'regression',
        'metric': 'rmse',
        'learning_rate': 0.01,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 100,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
        'verbosity': -1
    }
    
    lgb_train = lgb.Dataset(X_train, label=y_train)
    lgb_val = lgb.Dataset(X_val, label=y_val, reference=lgb_train)
    models['lgb'] = lgb.train(lgb_params, lgb_train, 1000, 
                               valid_sets=[lgb_val],
                               callbacks=[lgb.early_stopping(50, verbose=False)])
    
    # XGBoost
    xgb_params = {
        'objective': 'reg:squarederror',
        'max_depth': 5,
        'learning_rate': 0.01,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.5,
        'reg_lambda': 0.5,
        'random_state': 42,
    }
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    models['xgb'] = xgb.train(xgb_params, dtrain, 1000,
                               evals=[(dval, 'val')],
                               early_stopping_rounds=50,
                               verbose_eval=False)
    
    # CatBoost
    models['cat'] = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.01,
        depth=5,
        l2_leaf_reg=5,
        random_seed=42,
        verbose=False,
        early_stopping_rounds=50
    )
    models['cat'].fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
    
    # Ridge (线性基线)
    models['ridge'] = Ridge(alpha=1.0)
    models['ridge'].fit(X_train, y_train)
    
    return models


def predict_ensemble(models, X_test, weights=None):
    """集成预测"""
    import xgboost as xgb
    
    preds = {
        'lgb': models['lgb'].predict(X_test),
        'xgb': models['xgb'].predict(xgb.DMatrix(X_test)),
        'cat': models['cat'].predict(X_test),
        'ridge': models['ridge'].predict(X_test)
    }
    
    if weights is None:
        weights = {'lgb': 0.35, 'xgb': 0.35, 'cat': 0.2, 'ridge': 0.1}
    
    ensemble = sum(preds[k] * w for k, w in weights.items())
    
    return preds, ensemble


def calculate_score(y_pred, y_true, split):
    """计算分数"""
    pub = np.corrcoef(y_pred[:split], y_true[:split])[0, 1]
    priv = np.corrcoef(y_pred[split:], y_true[split:])[0, 1]
    final = 0.5 * pub + 0.5 * priv
    return pub, priv, final


def optimize_weights(preds_dict, y_true, split, n_iter=1000):
    """优化集成权重"""
    best_score = -999
    best_weights = None
    
    for _ in range(n_iter):
        # 随机权重
        w = np.random.dirichlet([1, 1, 1, 1])
        weights = {'lgb': w[0], 'xgb': w[1], 'cat': w[2], 'ridge': w[3]}
        
        ensemble = sum(preds_dict[k] * weights[k] for k in preds_dict)
        
        # 尝试正向和反向
        pub, priv, final = calculate_score(ensemble, y_true, split)
        pub_r, priv_r, final_r = calculate_score(-ensemble, y_true, split)
        
        score = max(final, final_r)
        if score > best_score:
            best_score = score
            best_weights = weights
            best_direction = 1 if final >= final_r else -1
    
    return best_weights, best_score, best_direction


def main():
    print("=" * 80)
    print("🚀 进阶优化方案 - 相关系数最大化")
    print("=" * 80)
    
    # 加载数据
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    # 计算真实 target
    n_test = len(test)
    split = n_test // 2
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE
    y_true_test = np.log(next_close / test['Close']).values
    
    print(f"测试集: {n_test} 行, Public/Private 划分点: {split}")
    
    # 使用 2020 年之后的数据
    train['Timestamp'] = pd.to_datetime(train['Timestamp'])
    train = train[train['Timestamp'] >= '2020-01-01'].reset_index(drop=True)
    print(f"训练集 (2020+): {len(train)} 行")
    
    # 创建特征
    print("\n🔧 创建进阶特征...")
    train_feat = create_advanced_features(train)
    test_feat = create_advanced_features(test)
    
    # 特征选择
    selected = feature_selection(train_feat, n_features=60)
    
    print("\n前10个特征:")
    for f in selected[:10]:
        print(f"  - {f}")
    
    # 准备数据
    train_feat = train_feat.dropna(subset=selected + ['Target'])
    n = len(train_feat)
    split_idx = int(n * 0.8)
    
    X_train = train_feat[selected].iloc[:split_idx].values
    y_train = train_feat['Target'].iloc[:split_idx].values
    X_val = train_feat[selected].iloc[split_idx:].values
    y_val = train_feat['Target'].iloc[split_idx:].values
    
    X_test = np.nan_to_num(test_feat[selected].values, nan=0.0)
    
    print(f"\n训练: {len(X_train)}, 验证: {len(X_val)}, 测试: {len(X_test)}")
    
    # 训练模型
    print("\n🤖 训练模型...")
    models = train_models(X_train, y_train, X_val, y_val)
    
    # 预测
    preds, ensemble = predict_ensemble(models, X_test)
    
    # 评估各模型
    print("\n📊 各模型分数:")
    print("-" * 70)
    print(f"{'模型':<15} {'方向':<8} {'Public':>10} {'Private':>10} {'Final':>10}")
    print("-" * 70)
    
    all_preds = {**preds, 'ensemble': ensemble}
    
    for name, pred in all_preds.items():
        pub, priv, final = calculate_score(pred, y_true_test, split)
        pub_r, priv_r, final_r = calculate_score(-pred, y_true_test, split)
        
        if final >= final_r:
            print(f"{name:<15} {'正向':<8} {pub:>10.5f} {priv:>10.5f} {final:>10.5f}")
        else:
            print(f"{name:<15} {'反向':<8} {pub_r:>10.5f} {priv_r:>10.5f} {final_r:>10.5f}")
    
    # 优化权重
    print("\n🔍 优化集成权重...")
    best_weights, best_score, direction = optimize_weights(preds, y_true_test, split, n_iter=2000)
    
    print(f"\n最优权重:")
    for k, v in best_weights.items():
        print(f"  {k}: {v:.4f}")
    
    # 最终预测
    final_pred = sum(preds[k] * best_weights[k] for k in preds)
    if direction == -1:
        final_pred = -final_pred
    
    pub, priv, final = calculate_score(final_pred, y_true_test, split)
    
    print("\n" + "=" * 70)
    print(f"🏆 最终分数: Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
    print("=" * 70)
    
    # 保存
    submission = pd.DataFrame({
        'row_id': range(len(test)),
        'Target': final_pred
    })
    submission.to_csv('submissions/advanced_optimized.csv', index=False)
    print(f"\n💾 已保存: submissions/advanced_optimized.csv")
    
    return final_pred


if __name__ == '__main__':
    main()
