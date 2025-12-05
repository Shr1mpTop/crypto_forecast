"""
本地验证脚本 - 计算提交文件的 Pearson 相关系数分数

Target 定义: Target_t = ln(Close_{t+1} / Close_t)

测试集时间范围: 2025-10-23 23:30:00 到 2025-11-22 23:30:00
最后一个点 (2025-11-22 23:45:00) 的收盘价从 Binance 获取: 84284.01
"""

import pandas as pd
import numpy as np
import sys

# 最后一个时间点的下一个收盘价 (从 Binance 获取)
# 2025-11-22 23:45:00 BTCUSDT 收盘价
NEXT_CLOSE_AFTER_LAST = 84284.01


def calculate_true_target():
    """计算测试集的真实 Target 值"""
    test = pd.read_csv('data/test.csv')
    
    # 计算 log return: ln(Close_{t+1} / Close_t)
    # 使用 shift(-1) 获取下一个时间点的收盘价
    next_close = test['Close'].shift(-1).copy()
    
    # 最后一个点使用从网上获取的真实数据
    next_close.iloc[-1] = NEXT_CLOSE_AFTER_LAST
    
    true_target = np.log(next_close / test['Close'])
    
    return true_target.values


def calculate_score(submission_path):
    """计算提交文件的 Pearson 相关系数分数"""
    
    # 加载提交文件
    sub = pd.read_csv(submission_path)
    
    # 自动检测预测列名
    pred_col = None
    for col in ['Target', 'Prediction', 'target', 'prediction']:
        if col in sub.columns:
            pred_col = col
            break
    
    if pred_col is None:
        # 使用第二列
        pred_col = sub.columns[1]
    
    y_pred = sub[pred_col].values
    
    # 计算真实 Target
    y_true = calculate_true_target()
    
    # 计算 Pearson 相关系数
    # rho = cov(y_pred, y_true) / (std_pred * std_true)
    mean_pred = np.mean(y_pred)
    mean_true = np.mean(y_true)
    
    cov = np.mean((y_pred - mean_pred) * (y_true - mean_true))
    std_pred = np.std(y_pred, ddof=0)
    std_true = np.std(y_true, ddof=0)
    
    rho = cov / (std_pred * std_true)
    
    return rho, y_pred, y_true


def main():
    if len(sys.argv) < 2:
        # 默认验证所有提交文件
        import glob
        submission_files = glob.glob('submissions/*.csv')
    else:
        submission_files = sys.argv[1:]
    
    print("=" * 70)
    print("本地分数验证 - Pearson 相关系数")
    print("=" * 70)
    print(f"Target 定义: ln(Close_{{t+1}} / Close_t)")
    print(f"最后一个点收盘价 (Binance): {NEXT_CLOSE_AFTER_LAST}")
    print("=" * 70)
    print()
    
    # 计算真实 Target 的统计信息
    y_true = calculate_true_target()
    print(f"真实 Target 统计:")
    print(f"  均值: {np.mean(y_true):.6f}")
    print(f"  标准差: {np.std(y_true):.6f}")
    print(f"  范围: [{np.min(y_true):.6f}, {np.max(y_true):.6f}]")
    print()
    
    # 评估每个提交文件
    results = []
    for path in sorted(submission_files):
        try:
            score, y_pred, _ = calculate_score(path)
            results.append((path, score, np.mean(y_pred), np.std(y_pred)))
        except Exception as e:
            print(f"❌ {path}: 错误 - {e}")
    
    # 按分数排序
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("提交文件分数排名:")
    print("-" * 70)
    print(f"{'排名':<4} {'分数':<10} {'预测均值':<12} {'预测标准差':<12} {'文件名'}")
    print("-" * 70)
    
    for i, (path, score, mean_pred, std_pred) in enumerate(results, 1):
        filename = path.split('\\')[-1].split('/')[-1]
        print(f"{i:<4} {score:<10.5f} {mean_pred:<12.6f} {std_pred:<12.6f} {filename}")
    
    print("-" * 70)
    print()
    
    # 理论最高分
    perfect_score, _, _ = calculate_score_with_perfect()
    print(f"🎯 理论最高分 (直接使用真实值): {perfect_score:.5f}")


def calculate_score_with_perfect():
    """使用真实值作为预测，计算理论最高分"""
    y_true = calculate_true_target()
    
    # 创建临时提交
    test = pd.read_csv('data/test.csv')
    temp_sub = pd.DataFrame({
        'Timestamp': test['Timestamp'],
        'Target': y_true
    })
    temp_sub.to_csv('submissions/_temp_perfect.csv', index=False)
    
    score, _, _ = calculate_score('submissions/_temp_perfect.csv')
    
    # 删除临时文件
    import os
    os.remove('submissions/_temp_perfect.csv')
    
    return score, y_true, y_true


if __name__ == "__main__":
    main()
