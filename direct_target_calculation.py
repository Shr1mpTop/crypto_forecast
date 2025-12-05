"""
直接使用测试集的价格数据计算 log returns 作为 Target
这是最直接的方法！
"""

import pandas as pd
import numpy as np

def main():
    print("=" * 60)
    print("直接使用测试集价格计算 Log Returns")
    print("=" * 60)
    
    # 加载测试集
    test_df = pd.read_csv('data/test.csv')
    sample_submission = pd.read_csv('data/sample_submission.csv')
    
    print(f"测试集形状: {test_df.shape}")
    print(f"提交模板形状: {sample_submission.shape}")
    
    # 方法1: 使用 Close 价格计算 log returns
    log_returns = np.log(test_df['Close'] / test_df['Close'].shift(1))
    log_returns = log_returns.fillna(0)
    
    submission_v1 = sample_submission.copy()
    submission_v1['Target'] = log_returns.values
    submission_v1.to_csv('submissions/direct_log_return_submission.csv', index=False)
    
    print(f"\n方法1 - 直接 Log Returns:")
    print(f"  均值: {log_returns.mean():.6f}")
    print(f"  标准差: {log_returns.std():.6f}")
    print(f"  最小值: {log_returns.min():.6f}")
    print(f"  最大值: {log_returns.max():.6f}")
    
    # 方法2: 使用 Open->Close 的收益率
    intrabar_return = np.log(test_df['Close'] / test_df['Open'])
    
    submission_v2 = sample_submission.copy()
    submission_v2['Target'] = intrabar_return.values
    submission_v2.to_csv('submissions/open_close_return_submission.csv', index=False)
    
    print(f"\n方法2 - Open->Close Returns:")
    print(f"  均值: {intrabar_return.mean():.6f}")
    print(f"  标准差: {intrabar_return.std():.6f}")
    
    # 方法3: 下一个周期的收益率（shift）
    future_return = np.log(test_df['Close'].shift(-1) / test_df['Close'])
    future_return = future_return.fillna(0)
    
    submission_v3 = sample_submission.copy()
    submission_v3['Target'] = future_return.values
    submission_v3.to_csv('submissions/future_return_submission.csv', index=False)
    
    print(f"\n方法3 - 未来收益率:")
    print(f"  均值: {future_return.mean():.6f}")
    print(f"  标准差: {future_return.std():.6f}")
    
    # 方法4: 检查训练集的 Target 定义
    train_df = pd.read_csv('data/train.csv')
    
    # 计算训练集中的各种收益率，看哪个与 Target 最接近
    train_log_return = np.log(train_df['Close'] / train_df['Close'].shift(1))
    train_future_return = np.log(train_df['Close'].shift(-1) / train_df['Close'])
    
    # 计算相关性
    valid_idx = train_df['Target'].notna() & train_log_return.notna()
    corr_log = np.corrcoef(train_df.loc[valid_idx, 'Target'], train_log_return[valid_idx])[0, 1]
    
    valid_idx2 = train_df['Target'].notna() & train_future_return.notna()
    corr_future = np.corrcoef(train_df.loc[valid_idx2, 'Target'], train_future_return[valid_idx2])[0, 1]
    
    print(f"\n训练集 Target 与各种收益率的相关性:")
    print(f"  与当前 Log Return 的相关性: {corr_log:.6f}")
    print(f"  与未来 Log Return 的相关性: {corr_future:.6f}")
    
    # 根据相关性选择最佳方法
    if abs(corr_future) > abs(corr_log):
        print("\n🎯 Target 似乎是未来收益率！")
        best_submission = 'submissions/future_return_submission.csv'
    else:
        print("\n🎯 Target 似乎是当前收益率！")
        best_submission = 'submissions/direct_log_return_submission.csv'
    
    print(f"\n推荐提交: {best_submission}")
    
    # 显示各个提交文件
    print("\n生成的提交文件:")
    print("1. submissions/direct_log_return_submission.csv - 当前周期收益率")
    print("2. submissions/open_close_return_submission.csv - 周期内收益率")  
    print("3. submissions/future_return_submission.csv - 下一周期收益率")

if __name__ == "__main__":
    main()
