"""
SC6117 加密货币预测比赛 - 优化后处理脚本
Improved Post-processing for Better Predictions

这个脚本用于分析和优化预测结果
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

print("=" * 80)
print("SC6117 预测后处理优化")
print("=" * 80)

# 加载数据
train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')
submission = pd.read_csv('submissions/ultimate_ensemble_submission.csv')

# 分析训练集Target分布
target = train_df['Target'].dropna()
print("\n训练集Target分布:")
print(f"  样本数: {len(target):,}")
print(f"  均值: {target.mean():.8f}")
print(f"  标准差: {target.std():.8f}")
print(f"  中位数: {target.median():.8f}")
print(f"  最小值: {target.min():.8f}")
print(f"  最大值: {target.max():.8f}")
print(f"  25%分位: {target.quantile(0.25):.8f}")
print(f"  75%分位: {target.quantile(0.75):.8f}")
print(f"  偏度: {target.skew():.4f}")
print(f"  峰度: {target.kurtosis():.4f}")

# 查看sample_submission的分布
sample_sub = pd.read_csv('data/sample_submission.csv')
print("\nSample Submission分布:")
print(f"  均值: {sample_sub['Prediction'].mean():.8f}")
print(f"  标准差: {sample_sub['Prediction'].std():.8f}")
print(f"  范围: [{sample_sub['Prediction'].min():.8f}, {sample_sub['Prediction'].max():.8f}]")

# 原始预测分布
print("\n当前预测分布 (裁剪后):")
print(f"  均值: {submission['Prediction'].mean():.8f}")
print(f"  标准差: {submission['Prediction'].std():.8f}")
print(f"  范围: [{submission['Prediction'].min():.8f}, {submission['Prediction'].max():.8f}]")

# 查看其他提交的分布
print("\n" + "=" * 80)
print("对比其他提交文件的分布:")
print("=" * 80)

submission_files = list(Path('submissions').glob('*.csv'))
for f in sorted(submission_files)[:10]:
    try:
        sub = pd.read_csv(f)
        if 'Prediction' in sub.columns:
            pred = sub['Prediction']
            print(f"\n{f.name}:")
            print(f"  均值: {pred.mean():.8f}, 标准差: {pred.std():.8f}")
            print(f"  范围: [{pred.min():.8f}, {pred.max():.8f}]")
    except:
        pass

# 生成多种后处理版本
print("\n" + "=" * 80)
print("生成多种后处理版本...")
print("=" * 80)

# 读取原始预测(无裁剪)
# 我们需要重新计算不带裁剪的版本
# 先读取现有版本进行变换

pred = submission['Prediction'].values.copy()

# 方法1: 标准化到训练集分布
train_mean = target.mean()
train_std = target.std()
pred_mean = pred.mean()
pred_std = pred.std()

# 标准化
pred_normalized = (pred - pred_mean) / pred_std * train_std + train_mean

submission_v1 = submission.copy()
submission_v1['Prediction'] = pred_normalized
submission_v1.to_csv('submissions/ultimate_normalized_submission.csv', index=False)
print(f"\n版本1 (标准化到训练分布):")
print(f"  均值: {pred_normalized.mean():.8f}")
print(f"  标准差: {pred_normalized.std():.8f}")
print(f"  保存: submissions/ultimate_normalized_submission.csv")

# 方法2: 使用更宽松的裁剪 (5倍标准差)
pred_clipped = np.clip(pred, 
                       train_mean - 5 * train_std, 
                       train_mean + 5 * train_std)
submission_v2 = submission.copy()
submission_v2['Prediction'] = pred_clipped
submission_v2.to_csv('submissions/ultimate_clipped_5std_submission.csv', index=False)
print(f"\n版本2 (5倍标准差裁剪):")
print(f"  均值: {pred_clipped.mean():.8f}")
print(f"  标准差: {pred_clipped.std():.8f}")
print(f"  保存: submissions/ultimate_clipped_5std_submission.csv")

# 方法3: RankGauss变换
from scipy.stats import rankdata, norm
def rank_gauss(x):
    """Rank-Gauss变换"""
    N = len(x)
    rank = rankdata(x, method='average')
    # 转换为正态分布
    gauss = norm.ppf((rank - 0.5) / N)
    return gauss

pred_rankgauss = rank_gauss(pred)
# 缩放到训练集分布
pred_rankgauss = pred_rankgauss * train_std + train_mean
submission_v3 = submission.copy()
submission_v3['Prediction'] = pred_rankgauss
submission_v3.to_csv('submissions/ultimate_rankgauss_submission.csv', index=False)
print(f"\n版本3 (RankGauss变换):")
print(f"  均值: {pred_rankgauss.mean():.8f}")
print(f"  标准差: {pred_rankgauss.std():.8f}")
print(f"  保存: submissions/ultimate_rankgauss_submission.csv")

# 方法4: 原始预测 (无后处理)
submission_v4 = submission.copy()
# 对于已经裁剪的版本，我们无法恢复，但可以尝试扩展
# 暂时使用原始版本
submission_v4.to_csv('submissions/ultimate_original_submission.csv', index=False)
print(f"\n版本4 (原始预测):")
print(f"  保存: submissions/ultimate_original_submission.csv")

# 方法5: 保守预测 - 缩小预测幅度
shrink_factor = 0.3  # 缩小到30%
pred_shrunk = pred * shrink_factor
submission_v5 = submission.copy()
submission_v5['Prediction'] = pred_shrunk
submission_v5.to_csv('submissions/ultimate_shrunk_submission.csv', index=False)
print(f"\n版本5 (保守预测，缩小到30%):")
print(f"  均值: {pred_shrunk.mean():.8f}")
print(f"  标准差: {pred_shrunk.std():.8f}")
print(f"  保存: submissions/ultimate_shrunk_submission.csv")

# 方法6: 使用训练集最后部分的分布进行校正
last_train_target = train_df['Target'].iloc[-10000:].dropna()
last_mean = last_train_target.mean()
last_std = last_train_target.std()

pred_last_normalized = (pred - pred_mean) / pred_std * last_std + last_mean
submission_v6 = submission.copy()
submission_v6['Prediction'] = pred_last_normalized
submission_v6.to_csv('submissions/ultimate_last_normalized_submission.csv', index=False)
print(f"\n版本6 (基于最近训练数据分布):")
print(f"  最近10000样本Target均值: {last_mean:.8f}")
print(f"  最近10000样本Target标准差: {last_std:.8f}")
print(f"  预测均值: {pred_last_normalized.mean():.8f}")
print(f"  预测标准差: {pred_last_normalized.std():.8f}")
print(f"  保存: submissions/ultimate_last_normalized_submission.csv")

print("\n" + "=" * 80)
print("所有版本已生成!")
print("=" * 80)
print("""
推荐提交顺序:
1. ultimate_normalized_submission.csv - 标准化到整体训练分布
2. ultimate_last_normalized_submission.csv - 标准化到最近训练分布 
3. ultimate_shrunk_submission.csv - 保守预测
4. ultimate_rankgauss_submission.csv - RankGauss变换
5. ultimate_original_submission.csv - 原始预测
""")
