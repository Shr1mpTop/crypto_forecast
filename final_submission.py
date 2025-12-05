"""
SC6117 最终提交优化 - 重新缩放预测到正确范围
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
from scipy.stats import rankdata, norm

print("=" * 80)
print("SC6117 最终提交优化")
print("=" * 80)

# 加载数据
train_df = pd.read_csv('data/train.csv')
submission = pd.read_csv('submissions/ultimate_ensemble_submission.csv')

# 获取预测值
pred = submission['Prediction'].values.copy()

# 训练集Target统计
target = train_df['Target'].dropna()
train_mean = target.mean()
train_std = target.std()

print(f"\n训练集Target: 均值={train_mean:.8f}, 标准差={train_std:.8f}")
print(f"当前预测: 均值={pred.mean():.8f}, 标准差={pred.std():.8f}")

# ============================================
# 方法A: 完全标准化 - 保持排序，匹配分布
# ============================================
print("\n" + "=" * 60)
print("生成最终提交版本...")
print("=" * 60)

# 1. Rank-Gauss标准化 + 训练分布缩放
def rank_gauss(x):
    N = len(x)
    rank = rankdata(x, method='average')
    gauss = norm.ppf((rank - 0.5) / N)
    return gauss

pred_rg = rank_gauss(pred)
pred_final = pred_rg * train_std + train_mean

# 轻微裁剪极端值
lower = np.percentile(target, 0.1)
upper = np.percentile(target, 99.9)
pred_final = np.clip(pred_final, lower, upper)

submission_final = submission.copy()
submission_final['Prediction'] = pred_final
submission_final.to_csv('submissions/final_submission_v1.csv', index=False)

print(f"\n版本1 (Rank-Gauss标准化):")
print(f"  均值: {pred_final.mean():.8f}")
print(f"  标准差: {pred_final.std():.8f}")
print(f"  范围: [{pred_final.min():.8f}, {pred_final.max():.8f}]")

# 2. 简单标准化
pred_std_norm = (pred - pred.mean()) / pred.std() * train_std + train_mean
pred_std_norm = np.clip(pred_std_norm, lower, upper)

submission_v2 = submission.copy()
submission_v2['Prediction'] = pred_std_norm
submission_v2.to_csv('submissions/final_submission_v2.csv', index=False)

print(f"\n版本2 (简单标准化):")
print(f"  均值: {pred_std_norm.mean():.8f}")
print(f"  标准差: {pred_std_norm.std():.8f}")
print(f"  范围: [{pred_std_norm.min():.8f}, {pred_std_norm.max():.8f}]")

# 3. 基于最近数据的标准化
last_target = train_df['Target'].iloc[-50000:].dropna()
last_mean = last_target.mean()
last_std = last_target.std()

pred_last = (pred - pred.mean()) / pred.std() * last_std + last_mean
pred_last = np.clip(pred_last, np.percentile(last_target, 0.5), np.percentile(last_target, 99.5))

submission_v3 = submission.copy()
submission_v3['Prediction'] = pred_last
submission_v3.to_csv('submissions/final_submission_v3.csv', index=False)

print(f"\n版本3 (基于最近50000样本):")
print(f"  参考均值: {last_mean:.8f}, 参考标准差: {last_std:.8f}")
print(f"  预测均值: {pred_last.mean():.8f}")
print(f"  预测标准差: {pred_last.std():.8f}")
print(f"  范围: [{pred_last.min():.8f}, {pred_last.max():.8f}]")

# 4. 类似sample_submission的缩放
sample_sub = pd.read_csv('data/sample_submission.csv')
sample_std = sample_sub['Prediction'].std()
sample_mean = sample_sub['Prediction'].mean()

pred_sample = (pred - pred.mean()) / pred.std() * sample_std + sample_mean
submission_v4 = submission.copy()
submission_v4['Prediction'] = pred_sample
submission_v4.to_csv('submissions/final_submission_v4.csv', index=False)

print(f"\n版本4 (匹配sample_submission分布):")
print(f"  参考均值: {sample_mean:.8f}, 参考标准差: {sample_std:.8f}")
print(f"  预测均值: {pred_sample.mean():.8f}")
print(f"  预测标准差: {pred_sample.std():.8f}")
print(f"  范围: [{pred_sample.min():.8f}, {pred_sample.max():.8f}]")

# 5. 缩放到0附近的小范围
scale_factor = 0.001 / pred.std()  # 目标标准差0.001
pred_scaled = (pred - pred.mean()) * scale_factor
submission_v5 = submission.copy()
submission_v5['Prediction'] = pred_scaled
submission_v5.to_csv('submissions/final_submission_v5.csv', index=False)

print(f"\n版本5 (缩放到小范围):")
print(f"  预测均值: {pred_scaled.mean():.8f}")
print(f"  预测标准差: {pred_scaled.std():.8f}")
print(f"  范围: [{pred_scaled.min():.8f}, {pred_scaled.max():.8f}]")

# 6. 组合集成 - 融合多个现有提交
print("\n" + "=" * 60)
print("组合集成 - 融合多个解决方案...")
print("=" * 60)

# 读取表现较好的提交
submissions_to_blend = [
    'submissions/sc6117_top_tier_submission.csv',
    'submissions/2nd_place_solution_submission.csv',
    'submissions/final_submission_v1.csv'
]

blend_preds = []
for f in submissions_to_blend:
    try:
        sub = pd.read_csv(f)
        if 'Prediction' in sub.columns and len(sub) == len(submission):
            # 标准化到相同尺度
            p = sub['Prediction'].values
            p_norm = (p - p.mean()) / (p.std() + 1e-10)
            blend_preds.append(p_norm)
            print(f"  加载: {f}")
    except Exception as e:
        print(f"  跳过: {f} ({e})")

if len(blend_preds) >= 2:
    # 等权重融合
    blended = np.mean(blend_preds, axis=0)
    # 缩放到训练分布
    blended_final = blended * train_std + train_mean
    blended_final = np.clip(blended_final, lower, upper)
    
    submission_blend = submission.copy()
    submission_blend['Prediction'] = blended_final
    submission_blend.to_csv('submissions/final_blended_submission.csv', index=False)
    
    print(f"\n融合版本 ({len(blend_preds)}个模型):")
    print(f"  预测均值: {blended_final.mean():.8f}")
    print(f"  预测标准差: {blended_final.std():.8f}")
    print(f"  范围: [{blended_final.min():.8f}, {blended_final.max():.8f}]")

print("\n" + "=" * 80)
print("所有版本已生成!")
print("=" * 80)
print("""
推荐提交顺序:
1. final_submission_v1.csv - Rank-Gauss标准化 (推荐首选)
2. final_blended_submission.csv - 多模型融合
3. final_submission_v3.csv - 基于最近数据
4. final_submission_v2.csv - 简单标准化
5. final_submission_v5.csv - 小范围缩放
""")

# 检查预测质量
print("\n" + "=" * 80)
print("预测质量检查")
print("=" * 80)

# 检查与训练集最后部分的相关性
last_train = train_df.iloc[-2881:].copy()
if 'Target' in last_train.columns:
    last_target = last_train['Target'].values
    corr, _ = stats.pearsonr(last_target[~np.isnan(last_target)], 
                              pred_final[:len(last_target)][~np.isnan(last_target)])
    print(f"与训练集最后部分的相关性: {corr:.6f}")
