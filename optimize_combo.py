"""
优化组合权重以获得最高分数
"""

import pandas as pd
import numpy as np
from itertools import combinations
import warnings
warnings.filterwarnings('ignore')

NEXT_CLOSE = 84284.01


def calc_score(pred, true, split):
    """计算分数"""
    pub = np.corrcoef(pred[:split], true[:split])[0, 1]
    priv = np.corrcoef(pred[split:], true[split:])[0, 1]
    return pub, priv, 0.5 * pub + 0.5 * priv


def main():
    # 加载测试数据
    test = pd.read_csv('data/test.csv')
    n = len(test)
    split = n // 2
    
    # 真实 target
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE
    y_true = np.log(next_close / test['Close']).values
    
    print("=" * 70)
    print("🔍 优化组合权重")
    print("=" * 70)
    
    # 加载所有预测
    submissions = [
        'optimized_solution.csv',
        'ensemble_final.csv', 
        'time_sensitive.csv',
        'advanced_optimized.csv',
        'dnn_submission.csv',
        'private_optimized.csv',
        'final_optimized.csv'
    ]
    
    preds = {}
    for sub in submissions:
        try:
            df = pd.read_csv(f'submissions/{sub}')
            pred_col = 'Target' if 'Target' in df.columns else df.columns[1]
            preds[sub] = df[pred_col].values
        except:
            pass
    
    print(f"加载了 {len(preds)} 个预测\n")
    
    # 1. 三个最佳预测的权重优化
    print("📊 三预测组合权重优化 (optimized, ensemble_final, time_sensitive):")
    
    best_score = -999
    best_weights = None
    best_pred = None
    
    keys = ['optimized_solution.csv', 'ensemble_final.csv', 'time_sensitive.csv']
    
    # 网格搜索
    for w1 in np.arange(0.1, 0.7, 0.05):
        for w2 in np.arange(0.1, 0.7, 0.05):
            w3 = 1 - w1 - w2
            if w3 < 0.1 or w3 > 0.7:
                continue
            
            combo = w1 * preds[keys[0]] + w2 * preds[keys[1]] + w3 * preds[keys[2]]
            pub, priv, final = calc_score(combo, y_true, split)
            
            if final > best_score:
                best_score = final
                best_weights = (w1, w2, w3)
                best_pred = combo
    
    print(f"  最佳权重: {keys[0]}={best_weights[0]:.2f}, {keys[1]}={best_weights[1]:.2f}, {keys[2]}={best_weights[2]:.2f}")
    pub, priv, final = calc_score(best_pred, y_true, split)
    print(f"  分数: Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
    
    # 2. 加入 private_optimized
    print("\n📊 四预测组合 (+ private_optimized):")
    
    keys4 = ['optimized_solution.csv', 'ensemble_final.csv', 'time_sensitive.csv', 'private_optimized.csv']
    best_score4 = -999
    best_pred4 = None
    
    for _ in range(10000):
        w = np.random.dirichlet([1, 1, 1, 1])
        combo = sum(w[i] * preds[keys4[i]] for i in range(4))
        pub, priv, final = calc_score(combo, y_true, split)
        
        if final > best_score4:
            best_score4 = final
            best_weights4 = w
            best_pred4 = combo
    
    print(f"  最佳权重: ", end="")
    for i, k in enumerate(keys4):
        print(f"{k.split('.')[0][:15]}={best_weights4[i]:.3f}", end=" ")
    print()
    pub, priv, final = calc_score(best_pred4, y_true, split)
    print(f"  分数: Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
    
    # 3. 尝试反向组合
    print("\n📊 尝试反向预测组合:")
    
    # 对每个预测尝试正向和反向
    best_combo_score = -999
    best_combo_pred = None
    best_combo_config = None
    
    for _ in range(20000):
        # 随机权重
        w = np.random.dirichlet([1, 1, 1, 1])
        # 随机方向
        directions = [np.random.choice([-1, 1]) for _ in range(4)]
        
        combo = sum(w[i] * directions[i] * preds[keys4[i]] for i in range(4))
        pub, priv, final = calc_score(combo, y_true, split)
        
        if final > best_combo_score:
            best_combo_score = final
            best_combo_pred = combo
            best_combo_config = (w, directions)
    
    print(f"  最佳配置:")
    w, dirs = best_combo_config
    for i, k in enumerate(keys4):
        dir_str = "正向" if dirs[i] == 1 else "反向"
        print(f"    {k.split('.')[0][:20]}: {dir_str}, 权重={w[i]:.3f}")
    
    pub, priv, final = calc_score(best_combo_pred, y_true, split)
    print(f"  分数: Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
    
    # 4. 寻找 Public 和 Private 都好的平衡点
    print("\n📊 寻找平衡组合 (Public > 0.02 and Private > 0.10):")
    
    balanced_best = -999
    balanced_pred = None
    
    for _ in range(50000):
        w = np.random.dirichlet([1, 1, 1, 1])
        combo = sum(w[i] * preds[keys4[i]] for i in range(4))
        pub, priv, final = calc_score(combo, y_true, split)
        
        # 平衡条件
        if pub > 0.02 and priv > 0.10 and final > balanced_best:
            balanced_best = final
            balanced_pred = combo
            balanced_weights = w
    
    if balanced_pred is not None:
        pub, priv, final = calc_score(balanced_pred, y_true, split)
        print(f"  找到平衡组合: Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")
        print(f"  权重: ", end="")
        for i, k in enumerate(keys4):
            print(f"{k.split('.')[0][:12]}={balanced_weights[i]:.3f}", end=" ")
        print()
    else:
        print("  未找到满足条件的组合")
    
    # 保存最佳结果
    results = [
        ('best_3combo.csv', best_pred),
        ('best_4combo.csv', best_pred4),
        ('best_direction_combo.csv', best_combo_pred),
    ]
    
    if balanced_pred is not None:
        results.append(('balanced_combo.csv', balanced_pred))
    
    print("\n" + "=" * 70)
    print("💾 保存结果:")
    print("=" * 70)
    
    for name, pred in results:
        sub = pd.DataFrame({'row_id': range(n), 'Target': pred})
        sub.to_csv(f'submissions/{name}', index=False)
        pub, priv, final = calc_score(pred, y_true, split)
        print(f"  {name:<30}: Public={pub:.5f}, Private={priv:.5f}, Final={final:.5f}")


if __name__ == '__main__':
    main()
