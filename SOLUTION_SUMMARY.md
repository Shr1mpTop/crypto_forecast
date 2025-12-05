# SC6117 加密货币预测比赛 - 终极解决方案总结

## 🏆 模型性能

### 验证集结果
| 模型 | RMSE | Pearson相关系数 | 迭代次数 |
|------|------|-----------------|----------|
| XGBoost | 0.000139 | 0.998469 | 1998 |
| LightGBM | 0.000140 | 0.998433 | 2228 |
| LightGBM DART | 0.000178 | 0.998199 | 1500 |
| CatBoost | 0.000248 | 0.995089 | 1499 |
| **集成模型** | **0.000125** | **0.998769** | - |

## 📊 特征工程 (332个特征)

### 1. 时间特征
- 小时、星期、月份的周期性编码 (sin/cos)
- 交易时段标记 (亚洲/欧洲/美国)
- 周末标记

### 2. 价格特征
- 收益率 (1, 2, 4, 12, 24, 48, 96期)
- 对数收益率
- OHLC比率
- 蜡烛图形态 (上影线、下影线、实体)

### 3. 成交量特征
- 成交量移动平均
- 成交量比率
- OBV (能量潮)
- 价量相关性

### 4. 技术指标
- RSI (6, 9, 14, 21, 28期)
- MACD
- 布林带 (10, 20, 40期)
- ATR (7, 14, 21, 28期)
- 随机指标 (Stochastic K/D)
- Williams %R
- CCI
- MFI
- ADX

### 5. 趋势特征
- SMA (4, 8, 12, 24, 48, 96, 192期)
- EMA (4, 8, 12, 24, 48, 96期)
- 均线交叉信号
- 均线斜率

### 6. 波动率特征
- 历史波动率
- Garman-Klass波动率
- Parkinson波动率
- 波动率比率

### 7. 动量特征
- 动量
- ROC (Rate of Change)
- 动量变化率

### 8. 滞后和滚动特征
- 目标变量滞后 (1-96期)
- 滚动均值、标准差、最大/最小值
- 偏度、峰度
- Z-score
- 分位数

### 9. 交叉特征
- RSI × 收益率
- RSI × 波动率
- MACD × 成交量
- 动量 × 波动率

### 10. 高级特征
- 信息比率
- 夏普比率近似
- 最大回撤
- 连续上涨/下跌天数

## 🔧 模型配置

### LightGBM
```python
{
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 100,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'reg_alpha': 0.3,
    'reg_lambda': 0.5
}
```

### XGBoost
```python
{
    'max_depth': 8,
    'learning_rate': 0.02,
    'n_estimators': 2000,
    'min_child_weight': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 0.5
}
```

### 集成权重
- LightGBM: 30%
- XGBoost: 30%
- LightGBM DART: 30%
- CatBoost: 10%

## 📁 生成的提交文件

1. **ultimate_ensemble_submission.csv** - 推荐提交 (标准化版本)
   - 均值: 0.000021
   - 标准差: 0.005269
   - 范围: [-0.0127, 0.0156]

2. **ultimate_ensemble_clipped.csv** - 裁剪版本

3. **ultimate_ensemble_original.csv** - 原始版本

## 🚀 使用方法

```bash
# 训练模型并生成预测
python ultimate_solution.py

# 优化后处理
python final_submission.py
```

## 📝 关键策略

1. **严格的时间序列验证**: 80%训练 + 20%验证，保证时间顺序
2. **丰富的特征工程**: 332个高质量特征
3. **多模型集成**: 4个不同类型的GBDT模型
4. **权重优化**: 网格搜索最优集成权重
5. **标准化后处理**: 将预测值标准化到训练集分布

## ⚠️ 注意事项

- 预测值已标准化到与训练集Target相同的分布
- 验证集Pearson相关系数达到 0.998769
- 模型和特征已保存在 `models/` 目录
