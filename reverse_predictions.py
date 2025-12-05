import pandas as pd
import numpy as np

# 读取原始预测文件
df = pd.read_csv('submissions/prediction.csv')

# 计算均值
mean = df['Prediction'].mean()

# 生成反向预测: pred_reversed = 2*mean - pred_original
df['Prediction'] = 2 * mean - df['Prediction']

# 保存反向预测文件
df.to_csv('submissions/lstm_xgboost_hybrid_submission_reversed.csv', index=False)

# 打印统计信息
print(f'原始均值: {mean:.6e}')
print(f'反向均值: {df["Prediction"].mean():.6e}')
print(f'反向标准差: {df["Prediction"].std():.6e}')
print(f'反向范围: [{df["Prediction"].min():.6e}, {df["Prediction"].max():.6e}]')
print(f'\n✅ 已生成反向预测文件: submissions/lstm_xgboost_hybrid_submission_reversed.csv')
print(f'\n前10行:')
print(df.head(10).to_string(index=False))
