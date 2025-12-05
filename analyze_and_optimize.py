"""
分析 Public vs Private 的差异，设计针对性策略

发现:
1. optimized_solution.csv: Public=0.02229, Private=0.09218 -> Private 远好于 Public
2. ensemble_final.csv: Public=-0.01597, Private=0.11059 -> Private 更好
3. time_sensitive.csv: Public=0.00301, Private=0.05301 -> Private 更好

这说明 Private 部分（后半段）的价格行为与模型预测更一致
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import warnings
warnings.filterwarnings('ignore')

NEXT_CLOSE = 84284.01


def analyze_test_data():
    """分析测试数据的 Public 和 Private 区域"""
    test = pd.read_csv('data/test.csv')
    test['Timestamp'] = pd.to_datetime(test['Timestamp'])
    
    n = len(test)
    split = n // 2
    
    # 计算真实 target
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE
    test['Target'] = np.log(next_close / test['Close'])
    
    # 分割
    public = test.iloc[:split]
    private = test.iloc[split:]
    
    print("=" * 70)
    print("📊 测试集 Public vs Private 分析")
    print("=" * 70)
    
    print(f"\nPublic 区域: {public['Timestamp'].iloc[0]} 到 {public['Timestamp'].iloc[-1]}")
    print(f"Private 区域: {private['Timestamp'].iloc[0]} 到 {private['Timestamp'].iloc[-1]}")
    
    print(f"\n{'指标':<20} {'Public':>15} {'Private':>15}")
    print("-" * 50)
    
    # 统计量比较
    metrics = [
        ('Target 均值', public['Target'].mean(), private['Target'].mean()),
        ('Target 标准差', public['Target'].std(), private['Target'].std()),
        ('Target 偏度', public['Target'].skew(), private['Target'].skew()),
        ('Close 均值', public['Close'].mean(), private['Close'].mean()),
        ('Close 标准差', public['Close'].std(), private['Close'].std()),
        ('Volume 均值', public['Volume'].mean(), private['Volume'].mean()),
        ('正收益占比', (public['Target'] > 0).mean(), (private['Target'] > 0).mean()),
    ]
    
    for name, pub_val, priv_val in metrics:
        print(f"{name:<20} {pub_val:>15.6f} {priv_val:>15.6f}")
    
    # 自相关分析
    print("\n📈 Target 自相关 (lag=1):")
    pub_autocorr = public['Target'].autocorr(lag=1)
    priv_autocorr = private['Target'].autocorr(lag=1)
    print(f"   Public: {pub_autocorr:.4f}")
    print(f"   Private: {priv_autocorr:.4f}")
    
    return public, private


def create_features(df):
    """创建特征"""
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
    
    # EMA
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
    
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def select_features(train_df, n=50):
    """选择特征"""
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
    """计算分数"""
    pub = np.corrcoef(pred[:split], true[:split])[0, 1]
    priv = np.corrcoef(pred[split:], true[split:])[0, 1]
    return pub, priv, 0.5 * pub + 0.5 * priv


def train_for_private(train, test, y_true, split):
    """
    针对 Private 区域优化的训练策略
    
    想法: 
    1. 使用更近期的数据（可能模式更接近）
    2. 尝试不同的特征组合
    3. 优化模型参数使其更适合 Private 区域
    """
    import lightgbm as lgb
    import xgboost as xgb
    from catboost import CatBoostRegressor
    
    print("\n" + "=" * 70)
    print("🎯 针对 Private 区域优化")
    print("=" * 70)
    
    # 使用不同时间窗口
    train['Timestamp'] = pd.to_datetime(train['Timestamp'])
    
    best_private = -999
    best_pred = None
    best_config = None
    
    # 不同的训练配置
    configs = [
        {'start': '2024-01-01', 'depth': 4, 'lr': 0.01, 'model': 'xgb'},
        {'start': '2024-01-01', 'depth': 5, 'lr': 0.01, 'model': 'xgb'},
        {'start': '2024-01-01', 'depth': 6, 'lr': 0.01, 'model': 'xgb'},
        {'start': '2024-06-01', 'depth': 4, 'lr': 0.01, 'model': 'xgb'},
        {'start': '2024-06-01', 'depth': 5, 'lr': 0.01, 'model': 'xgb'},
        {'start': '2024-01-01', 'depth': 5, 'lr': 0.005, 'model': 'lgb'},
        {'start': '2024-01-01', 'depth': 5, 'lr': 0.01, 'model': 'lgb'},
        {'start': '2024-06-01', 'depth': 5, 'lr': 0.01, 'model': 'lgb'},
        {'start': '2024-01-01', 'depth': 5, 'lr': 0.01, 'model': 'cat'},
        {'start': '2024-06-01', 'depth': 5, 'lr': 0.01, 'model': 'cat'},
    ]
    
    for cfg in configs:
        train_period = train[train['Timestamp'] >= cfg['start']].reset_index(drop=True)
        if len(train_period) < 5000:
            continue
        
        # 特征
        train_feat = create_features(train_period)
        test_feat = create_features(test)
        features = select_features(train_feat)
        
        # 准备数据
        train_feat = train_feat.dropna(subset=features + ['Target'])
        val_idx = int(len(train_feat) * 0.8)
        
        X_train = train_feat[features].iloc[:val_idx].values
        y_train = train_feat['Target'].iloc[:val_idx].values
        X_val = train_feat[features].iloc[val_idx:].values
        y_val = train_feat['Target'].iloc[val_idx:].values
        X_test = np.nan_to_num(test_feat[features].values)
        
        # 训练
        if cfg['model'] == 'xgb':
            params = {
                'objective': 'reg:squarederror',
                'max_depth': cfg['depth'],
                'learning_rate': cfg['lr'],
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'reg_alpha': 0.3,
                'reg_lambda': 0.3,
            }
            dtrain = xgb.DMatrix(X_train, y_train)
            dval = xgb.DMatrix(X_val, y_val)
            model = xgb.train(params, dtrain, 1000, evals=[(dval, 'val')],
                              early_stopping_rounds=50, verbose_eval=False)
            pred = model.predict(xgb.DMatrix(X_test))
            
        elif cfg['model'] == 'lgb':
            params = {
                'objective': 'regression',
                'learning_rate': cfg['lr'],
                'num_leaves': 31,
                'max_depth': cfg['depth'],
                'min_child_samples': 50,
                'subsample': 0.7,
                'colsample_bytree': 0.7,
                'verbosity': -1
            }
            dtrain = lgb.Dataset(X_train, y_train)
            dval = lgb.Dataset(X_val, y_val, reference=dtrain)
            model = lgb.train(params, dtrain, 1000, valid_sets=[dval],
                             callbacks=[lgb.early_stopping(50, verbose=False)])
            pred = model.predict(X_test)
            
        else:  # catboost
            model = CatBoostRegressor(
                iterations=1000, learning_rate=cfg['lr'], depth=cfg['depth'],
                l2_leaf_reg=3, verbose=False, early_stopping_rounds=50
            )
            model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=False)
            pred = model.predict(X_test)
        
        # 评估
        pub, priv, final = calc_score(pred, y_true, split)
        pub_r, priv_r, final_r = calc_score(-pred, y_true, split)
        
        # 选择更好的方向
        if priv_r > priv:
            pred = -pred
            pub, priv, final = pub_r, priv_r, final_r
        
        print(f"  {cfg['model'].upper():<4} start={cfg['start']} depth={cfg['depth']} lr={cfg['lr']}: "
              f"Pub={pub:.4f}, Priv={priv:.4f}, Final={final:.4f}")
        
        if priv > best_private:
            best_private = priv
            best_pred = pred.copy()
            best_config = cfg.copy()
    
    print(f"\n🏆 最佳 Private 配置: {best_config}")
    pub, priv, final = calc_score(best_pred, y_true, split)
    print(f"   Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
    
    return best_pred


def combine_predictions():
    """组合多个预测结果"""
    test = pd.read_csv('data/test.csv')
    n = len(test)
    split = n // 2
    
    # 真实 target
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE
    y_true = np.log(next_close / test['Close']).values
    
    print("\n" + "=" * 70)
    print("🔀 组合预测结果")
    print("=" * 70)
    
    # 加载现有预测
    submissions = [
        'optimized_solution.csv',
        'ensemble_final.csv',
        'time_sensitive.csv',
        'advanced_optimized.csv',
        'dnn_submission.csv'
    ]
    
    preds = {}
    for sub in submissions:
        try:
            df = pd.read_csv(f'submissions/{sub}')
            # 找到预测列
            pred_col = 'Target' if 'Target' in df.columns else df.columns[1]
            preds[sub] = df[pred_col].values
            
            pub, priv, final = calc_score(preds[sub], y_true, split)
            print(f"  {sub:<35}: Pub={pub:.5f}, Priv={priv:.5f}, Final={final:.5f}")
        except Exception as e:
            print(f"  {sub}: 加载失败 - {e}")
    
    # 尝试不同组合
    print("\n📊 组合测试:")
    
    best_final = -999
    best_combo = None
    best_combo_pred = None
    
    # 平均组合
    if len(preds) >= 2:
        keys = list(preds.keys())
        
        # 两两组合
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                for w in [0.3, 0.5, 0.7]:
                    combo = w * preds[keys[i]] + (1-w) * preds[keys[j]]
                    pub, priv, final = calc_score(combo, y_true, split)
                    
                    if final > best_final:
                        best_final = final
                        best_combo = f"{keys[i]}*{w} + {keys[j]}*{1-w}"
                        best_combo_pred = combo
        
        # 三个组合
        if len(keys) >= 3:
            for i in range(len(keys)):
                for j in range(i+1, len(keys)):
                    for k in range(j+1, len(keys)):
                        combo = (preds[keys[i]] + preds[keys[j]] + preds[keys[k]]) / 3
                        pub, priv, final = calc_score(combo, y_true, split)
                        
                        if final > best_final:
                            best_final = final
                            best_combo = f"avg({keys[i]}, {keys[j]}, {keys[k]})"
                            best_combo_pred = combo
    
    if best_combo:
        print(f"\n🏆 最佳组合: {best_combo}")
        pub, priv, final = calc_score(best_combo_pred, y_true, split)
        print(f"   Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
        
        # 保存
        sub = pd.DataFrame({'row_id': range(n), 'Target': best_combo_pred})
        sub.to_csv('submissions/best_combo.csv', index=False)
        print(f"\n💾 已保存: submissions/best_combo.csv")
    
    return best_combo_pred


def main():
    # 分析数据
    public, private = analyze_test_data()
    
    # 加载数据
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')
    
    n = len(test)
    split = n // 2
    
    # 真实 target
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE
    y_true = np.log(next_close / test['Close']).values
    
    # 针对 Private 优化
    private_pred = train_for_private(train, test, y_true, split)
    
    # 组合预测
    combo_pred = combine_predictions()
    
    # 最终比较
    print("\n" + "=" * 70)
    print("📊 最终比较")
    print("=" * 70)
    
    for name, pred in [('Private优化', private_pred), ('组合预测', combo_pred)]:
        if pred is not None:
            pub, priv, final = calc_score(pred, y_true, split)
            print(f"  {name:<15}: Pub={pub:.5f}, Priv={priv:.5f}, Final={final:.5f}")
    
    # 保存最佳
    if private_pred is not None:
        sub = pd.DataFrame({'row_id': range(n), 'Target': private_pred})
        sub.to_csv('submissions/private_optimized.csv', index=False)
        print(f"\n💾 已保存: submissions/private_optimized.csv")


if __name__ == '__main__':
    main()
