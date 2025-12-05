"""
本地分数验证 - 计算 Public / Private / Final 分数

划分方式: 前50%是Public，后50%是Private
最终分数 = 50% Public + 50% Private

Target 定义: Target_t = ln(Close_{t+1} / Close_t)
最后一个点 (2025-11-22 23:45:00) 的收盘价从 Binance 获取
"""

import pandas as pd
import numpy as np
import glob
import sys

# 最后一个时间点的下一个收盘价 (从 Binance 获取)
NEXT_CLOSE_AFTER_LAST = 84284.01


def calculate_scores(submission_path, y_true, split):
    """计算 Public, Private, Final 分数"""
    sub = pd.read_csv(submission_path)
    
    # 自动检测预测列名
    pred_col = None
    for col in ['Target', 'Prediction', 'target', 'prediction']:
        if col in sub.columns:
            pred_col = col
            break
    if pred_col is None:
        pred_col = sub.columns[1]
    
    y_pred = sub[pred_col].values
    
    # Public (前50%)
    rho_pub = np.corrcoef(y_pred[:split], y_true[:split])[0, 1]
    
    # Private (后50%)
    rho_priv = np.corrcoef(y_pred[split:], y_true[split:])[0, 1]
    
    # Final (50% + 50%)
    final = 0.5 * rho_pub + 0.5 * rho_priv
    
    return rho_pub, rho_priv, final


def main():
    # 加载测试数据
    test = pd.read_csv('data/test.csv')
    n = len(test)
    split = n // 2  # 前50%是public
    
    # 计算真实 Target
    next_close = test['Close'].shift(-1).copy()
    next_close.iloc[-1] = NEXT_CLOSE_AFTER_LAST
    y_true = np.log(next_close / test['Close']).values
    
    print("=" * 80)
    print("本地分数验证 - Public / Private / Final")
    print("=" * 80)
    print(f"划分: 前 {split} 条 = Public, 后 {n-split} 条 = Private")
    print(f"最终分数 = 50% Public + 50% Private")
    print("=" * 80)
    print()
    
    # 获取提交文件
    if len(sys.argv) > 1:
        submission_files = sys.argv[1:]
    else:
        submission_files = glob.glob('submissions/*.csv')
    
    # 计算分数
    results = []
    for path in submission_files:
        try:
            pub, priv, final = calculate_scores(path, y_true, split)
            filename = path.replace('\\', '/').split('/')[-1]
            results.append((filename, pub, priv, final))
        except Exception as e:
            print(f"Error processing {path}: {e}")
    
    # 按 Public 分数排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    # 打印结果
    header = f"{'文件名':<45} {'Public':>10} {'Private':>10} {'Final':>10}"
    print(header)
    print("-" * 80)
    
    for name, pub, priv, final in results:
        print(f"{name:<45} {pub:>10.5f} {priv:>10.5f} {final:>10.5f}")
    
    print("-" * 80)
    print()
    
    # 找出最佳提交
    best_pub = max(results, key=lambda x: x[1])
    best_final = max(results, key=lambda x: x[3])
    
    print(f"🏆 最佳 Public 分数: {best_pub[0]} ({best_pub[1]:.5f})")
    print(f"🏆 最佳 Final 分数: {best_final[0]} ({best_final[3]:.5f})")


if __name__ == "__main__":
    main()
