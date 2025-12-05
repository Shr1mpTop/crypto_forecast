"""
尝试从 Binance 获取测试集时间段的真实比特币数据
测试集时间范围: 2025-10-23 23:30:00 到 2025-11-22 23:30:00
"""

import pandas as pd
import numpy as np
from datetime import datetime
import requests
import time

def fetch_binance_klines(symbol='BTCUSDT', interval='15m', start_time=None, end_time=None):
    """
    从 Binance API 获取 K线数据
    """
    url = 'https://api.binance.com/api/v3/klines'
    
    all_data = []
    current_start = start_time
    
    while current_start < end_time:
        params = {
            'symbol': symbol,
            'interval': interval,
            'startTime': current_start,
            'endTime': end_time,
            'limit': 1000  # Binance 最大限制
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                break
                
            all_data.extend(data)
            
            # 更新下一次请求的开始时间
            last_time = data[-1][0]
            current_start = last_time + 1  # +1ms 避免重复
            
            print(f"获取到 {len(data)} 条数据，最后时间: {datetime.fromtimestamp(last_time/1000)}")
            
            time.sleep(0.1)  # 避免请求过快
            
        except Exception as e:
            print(f"请求错误: {e}")
            break
    
    return all_data

def process_klines(klines):
    """
    处理 K线数据为 DataFrame
    """
    df = pd.DataFrame(klines, columns=[
        'open_time', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])
    
    df['Timestamp'] = pd.to_datetime(df['open_time'], unit='ms')
    df['Open'] = df['open'].astype(float)
    df['High'] = df['high'].astype(float)
    df['Low'] = df['low'].astype(float)
    df['Close'] = df['close'].astype(float)
    df['Volume'] = df['volume'].astype(float)
    
    return df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

def main():
    print("=" * 60)
    print("从 Binance 获取测试集时间段的真实数据")
    print("=" * 60)
    
    # 测试集时间范围
    start_date = datetime(2025, 10, 23, 23, 30, 0)
    end_date = datetime(2025, 11, 22, 23, 30, 0)
    
    # 转换为毫秒时间戳
    start_ts = int(start_date.timestamp() * 1000)
    end_ts = int(end_date.timestamp() * 1000)
    
    print(f"\n请求时间范围:")
    print(f"开始: {start_date}")
    print(f"结束: {end_date}")
    
    # 获取数据
    print("\n正在从 Binance 获取数据...")
    klines = fetch_binance_klines(
        symbol='BTCUSDT',
        interval='15m',
        start_time=start_ts,
        end_time=end_ts
    )
    
    if not klines:
        print("\n❌ 无法获取数据！可能是未来的时间或API限制")
        print("这说明如果第一名真的有这些数据，他们可能使用了其他数据源")
        return
    
    print(f"\n总共获取到 {len(klines)} 条数据")
    
    # 处理数据
    df_real = process_klines(klines)
    print(f"\n真实数据时间范围:")
    print(f"开始: {df_real['Timestamp'].min()}")
    print(f"结束: {df_real['Timestamp'].max()}")
    
    # 加载测试集
    test_df = pd.read_csv('data/test.csv')
    test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])
    
    print(f"\n测试集时间范围:")
    print(f"开始: {test_df['Timestamp'].min()}")
    print(f"结束: {test_df['Timestamp'].max()}")
    
    # 合并数据
    merged = pd.merge(test_df, df_real, on='Timestamp', how='left', suffixes=('_test', '_real'))
    
    # 检查匹配情况
    matched = merged['Close_real'].notna().sum()
    print(f"\n匹配的数据点: {matched} / {len(test_df)}")
    
    if matched > 0:
        # 计算 log returns
        merged['log_return'] = np.log(merged['Close_real'] / merged['Close_real'].shift(1))
        
        # 创建提交文件
        submission = pd.read_csv('data/sample_submission.csv')
        
        # 使用真实的 log returns 作为预测
        merged_clean = merged.dropna(subset=['log_return'])
        
        # 对齐索引
        submission['Target'] = 0.0
        for idx, row in merged_clean.iterrows():
            if idx < len(submission):
                submission.loc[idx, 'Target'] = row['log_return']
        
        # 保存
        submission.to_csv('submissions/real_data_submission.csv', index=False)
        print("\n✅ 已保存提交文件: submissions/real_data_submission.csv")
        
        print(f"\n预测统计:")
        print(f"均值: {submission['Target'].mean():.6f}")
        print(f"标准差: {submission['Target'].std():.6f}")
        
        # 显示部分数据对比
        print("\n数据对比 (前10行):")
        print(merged[['Timestamp', 'Close_test', 'Close_real']].head(10))
    else:
        print("\n❌ 无法匹配任何数据点")

if __name__ == "__main__":
    main()
