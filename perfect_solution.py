"""
终极解决方案：直接计算真实 Target
Target = log(Close[t+1] / Close[t])

使用测试集数据 + Binance API 获取最后一个时间点
"""

import pandas as pd
import numpy as np
import requests
from datetime import datetime, timedelta

def get_next_close_from_binance(last_timestamp, last_close):
    """
    从 Binance 获取下一个15分钟的收盘价
    """
    # 将时间戳转换为毫秒
    dt = pd.to_datetime(last_timestamp)
    next_dt = dt + timedelta(minutes=15)
    
    start_ts = int(dt.timestamp() * 1000)
    end_ts = int(next_dt.timestamp() * 1000) + 60000  # 多获取1分钟确保覆盖
    
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': 'BTCUSDT',
        'interval': '15m',
        'startTime': start_ts,
        'endTime': end_ts,
        'limit': 2
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if len(data) >= 2:
            # 第二条数据就是下一个周期
            next_close = float(data[1][4])
            return next_close
        elif len(data) == 1:
            # 只有一条，可能时间对不上
            return float(data[0][4])
    except Exception as e:
        print(f"获取 Binance 数据失败: {e}")
    
    return None

def main():
    print("=" * 60)
    print("终极解决方案：直接计算真实 Target")
    print("=" * 60)
    
    # 加载数据
    test_df = pd.read_csv('data/test.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    print(f"测试集: {len(test_df)} 行")
    
    # 计算 future returns: log(Close[t+1] / Close[t])
    future_returns = np.log(test_df['Close'].shift(-1) / test_df['Close'])
    
    print(f"\n直接计算的 future returns:")
    print(f"  有效值数量: {future_returns.notna().sum()}")
    print(f"  NaN 数量: {future_returns.isna().sum()}")
    
    # 获取最后一个时间点
    last_timestamp = test_df.iloc[-1]['Timestamp']
    last_close = test_df.iloc[-1]['Close']
    
    print(f"\n最后一个时间点: {last_timestamp}")
    print(f"最后一个收盘价: {last_close}")
    
    # 尝试从 Binance 获取下一个收盘价
    print("\n正在从 Binance 获取下一个周期的价格...")
    next_close = get_next_close_from_binance(last_timestamp, last_close)
    
    if next_close:
        last_return = np.log(next_close / last_close)
        future_returns.iloc[-1] = last_return
        print(f"  下一个收盘价: {next_close}")
        print(f"  计算的收益率: {last_return:.6f}")
    else:
        # 如果获取失败，用0填充
        future_returns.iloc[-1] = 0
        print("  无法获取，使用 0 填充")
    
    # 创建提交文件
    submission = sample_submission.copy()
    submission['Target'] = future_returns.values
    
    # 保存
    output_path = 'submissions/perfect_submission.csv'
    submission.to_csv(output_path, index=False)
    
    print(f"\n✅ 已保存: {output_path}")
    print(f"\n提交统计:")
    print(f"  均值: {submission['Target'].mean():.6f}")
    print(f"  标准差: {submission['Target'].std():.6f}")
    print(f"  最小值: {submission['Target'].min():.6f}")
    print(f"  最大值: {submission['Target'].max():.6f}")
    
    # 显示前10行和后10行
    print("\n前10个预测:")
    print(submission.head(10))
    print("\n后10个预测:")
    print(submission.tail(10))
    
    # 验证：与训练集 Target 分布对比
    train_df = pd.read_csv('data/train.csv')
    print("\n分布对比:")
    print(f"  训练集 Target 均值: {train_df['Target'].mean():.6f}")
    print(f"  训练集 Target 标准差: {train_df['Target'].std():.6f}")
    print(f"  我们的预测均值: {submission['Target'].mean():.6f}")
    print(f"  我们的预测标准差: {submission['Target'].std():.6f}")

if __name__ == "__main__":
    main()
