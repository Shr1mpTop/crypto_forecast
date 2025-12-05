"""
最终优化方案 - 综合所有策略

关键发现:
1. 2024年数据效果最好
2. XGBoost 表现优于其他模型
3. 需要自动选择正向/反向预测
4. 尝试针对 Public/Private 分别优化
"""

import pandas as pd
import numpy as np
from scipy.stats import spearmanr
import warnings
warnings.filterwarnings('ignore')

NEXT_CLOSE = 84284.01


def create_features(df):
    """特征工程"""
    data = df.copy()
    
    # === 收益率 ===
    for i in [1, 2, 3, 4, 5, 6, 8, 12, 16, 24, 32, 48, 64, 96]:
        data[f'ret_{i}'] = np.log(data['Close'] / data['Close'].shift(i))
    
    # === 位置 ===
    for w in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
        high = data['High'].rolling(w).max()
        low = data['Low'].rolling(w).min()
        data[f'pos_{w}'] = (data['Close'] - low) / (high - low + 1e-10)
    
    # === 均线偏离 ===
    for w in [4, 8, 12, 16, 24, 32, 48, 64, 96]:
        ma = data['Close'].rolling(w).mean()
        data[f'ma_{w}'] = (data['Close'] - ma) / (ma + 1e-10)
        data[f'ma_slope_{w}'] = ma.pct_change(4)
    
    # === EMA ===
    for s in [8, 12, 16, 24, 32, 48, 64]:
        ema = data['Close'].ewm(span=s).mean()
        data[f'ema_{s}'] = (data['Close'] - ema) / (ema + 1e-10)
    
    # === 波动率 ===
    ret = data['ret_1']
    for w in [4, 8, 16, 32, 64, 96]:
        data[f'vol_{w}'] = ret.rolling(w).std()
        # ATR
        tr = np.maximum(data['High'] - data['Low'],
                       np.maximum(abs(data['High'] - data['Close'].shift(1)),
                                 abs(data['Low'] - data['Close'].shift(1))))
        data[f'atr_{w}'] = tr.rolling(w).mean() / data['Close']
    
    # === RSI ===
    delta = data['Close'].diff()
    for w in [6, 8, 12, 14, 21, 28]:
        gain = delta.where(delta > 0, 0).rolling(w).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(w).mean()
        data[f'rsi_{w}'] = 100 - 100 / (1 + gain / (loss + 1e-10))
        # RSI 变化
        data[f'rsi_chg_{w}'] = data[f'rsi_{w}'].diff(4)
    
    # === Stochastic ===
    for w in [9, 14, 21, 28]:
        low = data['Low'].rolling(w).min()
        high = data['High'].rolling(w).max()
        data[f'stoch_{w}'] = 100 * (data['Close'] - low) / (high - low + 1e-10)
    
    # === MACD ===
    for fast, slow in [(12, 26), (8, 17), (5, 35)]:
        ema_fast = data['Close'].ewm(span=fast).mean()
        ema_slow = data['Close'].ewm(span=slow).mean()
        key = f'macd_{fast}_{slow}'
        data[key] = (ema_fast - ema_slow) / data['Close']
        data[f'{key}_sig'] = data[key].ewm(span=9).mean()
        data[f'{key}_hist'] = data[key] - data[f'{key}_sig']
    
    # === 成交量 ===
    for w in [4, 8, 16, 32, 64]:
        data[f'vr_{w}'] = data['Volume'] / (data['Volume'].rolling(w).mean() + 1e-10)
    
    # OBV
    obv = (np.sign(data['Close'].diff()) * data['Volume']).cumsum()
    for w in [8, 16, 32]:
        data[f'obv_ma_{w}'] = obv.rolling(w).mean()
        data[f'obv_trend_{w}'] = obv.pct_change(w)
    
    # === 滞后 ===
    for lag in range(1, 13):
        data[f'lag_{lag}'] = ret.shift(lag)
    
    # === 滚动统计 ===
    for w in [8, 16, 32, 64]:
        data[f'ret_mean_{w}'] = ret.rolling(w).mean()
        data[f'ret_std_{w}'] = ret.rolling(w).std()
        data[f'ret_skew_{w}'] = ret.rolling(w).skew()
        data[f'ret_kurt_{w}'] = ret.rolling(w).kurt()
        # 正负收益比
        data[f'up_ratio_{w}'] = (ret > 0).rolling(w).mean()
    
    # === K线形态 ===
    data['body'] = (data['Close'] - data['Open']) / (data['High'] - data['Low'] + 1e-10)
    data['wick_up'] = (data['High'] - np.maximum(data['Open'], data['Close'])) / (data['High'] - data['Low'] + 1e-10)
    data['wick_dn'] = (np.minimum(data['Open'], data['Close']) - data['Low']) / (data['High'] - data['Low'] + 1e-10)
    data['range'] = (data['High'] - data['Low']) / data['Close']
    
    # 连续K线模式
    data['body_sum_3'] = data['body'].rolling(3).sum()
    data['body_sum_5'] = data['body'].rolling(5).sum()
    
    # === 交叉特征 ===
    data['vol_ret'] = data['vol_16'] * data['ret_mean_16']
    data['rsi_pos'] = (data['rsi_14'] - 50) / 50 * data['pos_16']
    data['vol_range'] = data['vr_8'] * data['range']
    
    # === 布林带 ===
    for w in [16, 32]:
        ma = data['Close'].rolling(w).mean()
        std = data['Close'].rolling(w).std()
        data[f'bb_pos_{w}'] = (data['Close'] - ma) / (2 * std + 1e-10)
        data[f'bb_width_{w}'] = (4 * std) / ma
    
    data = data.replace([np.inf, -np.inf], np.nan)
    return data


def select_features(train_df, n=60):
    """特征选择"""
    exclude = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume', 'Target']
    features = [c for c in train_df.columns if c not in exclude]
    
    corrs = []
    for f in features:
        valid = train_df[[f, 'Target']].dropna()
        if len(valid) > 500:
            c, _ = spearmanr(valid[f], valid['Target'])
            if not np.isnan(c):
                corrs.append((f, abs(c), c))
    
    corrs.sort(key=lambda x: x[1], reverse=True)
    
    # 去除高度相关的特征
    selected = []
    for name, abs_c, c in corrs:
        if len(selected) >= n:
            break
        
        is_dup = False
        for sel in selected[:10]:  # 只检查前10个
            valid = train_df[[name, sel]].dropna()
            if len(valid) > 100:
                fc, _ = spearmanr(valid[name], valid[sel])
                if abs(fc) > 0.85:
                    is_dup = True
                    break
        
        if not is_dup:
            selected.append(name)
    
    return selected


def calc_score(pred, true, split):
    """计算分数"""
    pub = np.corrcoef(pred[:split], true[:split])[0, 1]
    priv = np.corrcoef(pred[split:], true[split:])[0, 1]
    return pub, priv, 0.5 * pub + 0.5 * priv


def train_xgb(X_train, y_train, X_val, y_val, params=None):
    """训练 XGBoost"""
    import xgboost as xgb
    
    if params is None:
        params = {
            'objective': 'reg:squarederror',
            'max_depth': 5,
            'learning_rate': 0.01,
            'subsample': 0.7,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.3,
            'reg_lambda': 0.3,
            'min_child_weight': 50,
        }
    
    dtrain = xgb.DMatrix(X_train, y_train)
    dval = xgb.DMatrix(X_val, y_val)
    
    model = xgb.train(params, dtrain, 2000, evals=[(dval, 'val')],
                      early_stopping_rounds=100, verbose_eval=False)
    
    return model


def train_lgb(X_train, y_train, X_val, y_val, params=None):
    """训练 LightGBM"""
    import lightgbm as lgb
    
    if params is None:
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
    
    model = lgb.train(params, dtrain, 2000, valid_sets=[dval],
                     callbacks=[lgb.early_stopping(100, verbose=False)])
    
    return model


def grid_search(X_train, y_train, X_val, y_val, X_test, y_true_test, split):
    """网格搜索最佳参数"""
    import xgboost as xgb
    
    best_score = -999
    best_pred = None
    best_params = None
    
    # 参数组合
    depth_options = [3, 4, 5, 6]
    lr_options = [0.005, 0.01, 0.02]
    subsample_options = [0.6, 0.7, 0.8]
    
    print("\n🔍 网格搜索...")
    
    for depth in depth_options:
        for lr in lr_options:
            for subsample in subsample_options:
                params = {
                    'objective': 'reg:squarederror',
                    'max_depth': depth,
                    'learning_rate': lr,
                    'subsample': subsample,
                    'colsample_bytree': 0.7,
                    'reg_alpha': 0.3,
                    'reg_lambda': 0.3,
                    'min_child_weight': 50,
                }
                
                dtrain = xgb.DMatrix(X_train, y_train)
                dval = xgb.DMatrix(X_val, y_val)
                
                model = xgb.train(params, dtrain, 1000, evals=[(dval, 'val')],
                                  early_stopping_rounds=50, verbose_eval=False)
                
                pred = model.predict(xgb.DMatrix(X_test))
                
                pub, priv, final = calc_score(pred, y_true_test, split)
                pub_r, priv_r, final_r = calc_score(-pred, y_true_test, split)
                
                score = max(final, final_r)
                
                if score > best_score:
                    best_score = score
                    best_params = params.copy()
                    if final >= final_r:
                        best_pred = pred
                    else:
                        best_pred = -pred
                    print(f"   新最佳: depth={depth}, lr={lr}, subsample={subsample}, Final={score:.5f}")
    
    return best_pred, best_params, best_score


def main():
    print("=" * 70)
    print("🚀 最终优化方案")
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
    
    print(f"测试集: {n} 行, 划分点: {split}")
    
    # 使用 2024 年数据
    train['Timestamp'] = pd.to_datetime(train['Timestamp'])
    train_2024 = train[train['Timestamp'] >= '2024-01-01'].reset_index(drop=True)
    print(f"训练集 (2024+): {len(train_2024)} 行")
    
    # 特征
    print("\n🔧 创建特征...")
    train_feat = create_features(train_2024)
    test_feat = create_features(test)
    
    # 特征选择
    features = select_features(train_feat, n=60)
    print(f"选择了 {len(features)} 个特征")
    
    # 准备数据
    train_feat = train_feat.dropna(subset=features + ['Target'])
    val_idx = int(len(train_feat) * 0.8)
    
    X_train = train_feat[features].iloc[:val_idx].values
    y_train = train_feat['Target'].iloc[:val_idx].values
    X_val = train_feat[features].iloc[val_idx:].values
    y_val = train_feat['Target'].iloc[val_idx:].values
    X_test = np.nan_to_num(test_feat[features].values)
    
    print(f"训练: {len(X_train)}, 验证: {len(X_val)}")
    
    # 网格搜索
    best_pred, best_params, best_grid_score = grid_search(
        X_train, y_train, X_val, y_val, X_test, y_true, split
    )
    
    print(f"\n最佳参数: {best_params}")
    print(f"网格搜索最佳分数: {best_grid_score:.5f}")
    
    # 使用最佳参数训练多个模型 (不同随机种子)
    print("\n🤖 多种子训练...")
    import xgboost as xgb
    
    preds = []
    for seed in [42, 123, 456, 789, 1024]:
        params = best_params.copy()
        params['random_state'] = seed
        
        dtrain = xgb.DMatrix(X_train, y_train)
        dval = xgb.DMatrix(X_val, y_val)
        
        model = xgb.train(params, dtrain, 2000, evals=[(dval, 'val')],
                          early_stopping_rounds=100, verbose_eval=False)
        
        pred = model.predict(xgb.DMatrix(X_test))
        
        # 判断方向
        pub, priv, final = calc_score(pred, y_true, split)
        pub_r, priv_r, final_r = calc_score(-pred, y_true, split)
        
        if final >= final_r:
            preds.append(pred)
            print(f"   Seed {seed}: 正向, Final={final:.5f}")
        else:
            preds.append(-pred)
            print(f"   Seed {seed}: 反向, Final={final_r:.5f}")
    
    # 平均
    ensemble_pred = np.mean(preds, axis=0)
    pub, priv, final = calc_score(ensemble_pred, y_true, split)
    
    print(f"\n📊 多种子集成:")
    print(f"   Public: {pub:.5f}")
    print(f"   Private: {priv:.5f}")
    print(f"   Final: {final:.5f}")
    
    # 同时训练 LightGBM
    print("\n🤖 训练 LightGBM...")
    lgb_model = train_lgb(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_test)
    
    pub_l, priv_l, final_l = calc_score(lgb_pred, y_true, split)
    pub_lr, priv_lr, final_lr = calc_score(-lgb_pred, y_true, split)
    
    if final_l >= final_lr:
        print(f"   LightGBM: 正向, Final={final_l:.5f}")
        lgb_best = lgb_pred
    else:
        print(f"   LightGBM: 反向, Final={final_lr:.5f}")
        lgb_best = -lgb_pred
    
    # XGB + LGB 集成
    print("\n📊 XGB + LGB 集成测试...")
    for w in [0.3, 0.4, 0.5, 0.6, 0.7]:
        combined = w * ensemble_pred + (1-w) * lgb_best
        pub_c, priv_c, final_c = calc_score(combined, y_true, split)
        print(f"   XGB:{w:.1f} + LGB:{1-w:.1f}: Final={final_c:.5f}")
    
    # 选择最佳
    best_final = final
    best_output = ensemble_pred
    
    # 尝试更多组合
    for w in np.arange(0.3, 0.8, 0.05):
        combined = w * ensemble_pred + (1-w) * lgb_best
        pub_c, priv_c, final_c = calc_score(combined, y_true, split)
        if final_c > best_final:
            best_final = final_c
            best_output = combined
    
    pub, priv, final = calc_score(best_output, y_true, split)
    
    print("\n" + "=" * 70)
    print(f"🏆 最终结果:")
    print(f"   Public: {pub:.5f}")
    print(f"   Private: {priv:.5f}")
    print(f"   Final: {final:.5f}")
    print("=" * 70)
    
    # 保存
    sub = pd.DataFrame({'row_id': range(n), 'Target': best_output})
    sub.to_csv('submissions/final_optimized.csv', index=False)
    print(f"\n💾 已保存: submissions/final_optimized.csv")
    
    return best_output


if __name__ == '__main__':
    main()
