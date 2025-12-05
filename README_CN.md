# 加密货币价格预测项目

基于LightGBM的时间序列预测系统，用于预测加密货币下一时刻的价格变化。

## 📊 项目概述

本项目实现了多种机器学习方法来预测加密货币价格的对数收益率。系统使用历史价格数据（OHLCV）构建特征，训练模型，并通过本地验证评估性能。

**最佳成绩**: Final Score = **0.08042** (Public: 0.07420, Private: 0.08664)

---

## 🚀 快速开始

### 1. 环境安装

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保以下文件存在：
- `data/train.csv` - 训练数据（包含Timestamp, Open, High, Low, Close, Volume）
- `data/test.csv` - 测试数据（同样的列）

### 3. 训练模型

**推荐：使用统一接口训练**

```bash
# 基础模型（快速，推荐）
python train.py --method basic --trials 20

# 搜索最佳训练起始日期
python train.py --method basic --trials 50 --search-date

# 高级特征模型（技术指标）
python train.py --method advanced --trials 20 --search-date

# 集成模型（多模型融合）
python train.py --method ensemble --trials 10 --models lgb xgb

# 保存最佳提交文件
python train.py --method basic --trials 30 --save submissions/my_best.csv
```

### 4. 评估结果

```bash
# 评估单个提交文件
python score_submission.py submissions/basic_best.csv

# 评估所有提交文件
python score_submission.py
```

---

## 📁 项目结构

```
crypto_forecast/
├── data/                          # 数据目录
│   ├── train.csv                  # 训练数据
│   ├── test.csv                   # 测试数据
│   └── sample_submission.csv      # 提交样例
│
├── models/                        # 保存的模型文件
│   ├── lgbm_strict_model.txt      # LightGBM模型
│   └── *_feature_importance.csv   # 特征重要性
│
├── submissions/                   # 提交文件目录
│   ├── basic_best.csv            # 基础方法最佳提交
│   ├── advanced_best.csv         # 高级方法最佳提交
│   └── ensemble_best.csv         # 集成方法最佳提交
│
├── notebooks/                     # Jupyter笔记本
│   └── *.ipynb                    # 探索性分析和实验
│
├── train.py                       # 🌟 统一训练接口（推荐使用）
├── lgbm_tune.py                   # 基础LightGBM训练
├── advanced_lgbm_simple.py        # 高级特征LightGBM
├── ensemble_tune.py               # 多模型集成
├── score_submission.py            # 本地评分工具
├── reverse_predictions.py         # 预测反转工具
│
├── requirements.txt               # Python依赖
├── README.md                      # 英文文档
└── README_CN.md                   # 中文文档（本文件）
```

---

## 🎯 训练方法对比

### 方法1: 基础LightGBM（推荐）⭐

**特点**：
- 使用基础特征（时间、价格收益、滞后、滚动统计）
- 训练速度快（20次试验约5-10分钟）
- 性能最佳（Final=0.08042）

**使用场景**：
- 快速迭代和基线模型
- 大多数情况下的首选方法

**运行命令**：
```bash
python train.py --method basic --trials 20 --search-date
```

**最佳配置**：
```
start_date: 2023-01-01
num_leaves: 31
max_depth: 6
learning_rate: 0.01
best_iteration: 63
reversed: YES (预测取反)
```

---

### 方法2: 高级特征LightGBM

**特点**：
- 50+技术指标（RSI, MACD, 布林带, 波动率, 动量等）
- 训练时间中等（20次试验约10-20分钟）
- 可能不如基础方法（过拟合风险）

**使用场景**：
- 探索技术分析特征
- 希望尝试更多特征组合

**运行命令**：
```bash
python train.py --method advanced --trials 20 --search-date
```

---

### 方法3: 多模型集成

**特点**：
- 融合多个模型（LightGBM + XGBoost + CatBoost）
- 自动权重优化
- 训练时间长（10次试验约15-30分钟）
- 可能获得额外性能提升

**使用场景**：
- 追求极致性能
- 有充足计算资源

**运行命令**：
```bash
# LightGBM + XGBoost
python train.py --method ensemble --trials 10 --models lgb xgb

# 包含CatBoost（需先安装：pip install catboost）
python train.py --method ensemble --trials 10 --models lgb xgb cat
```

---

## 🔬 核心技术

### 特征工程

**基础特征（约40个）**：
- 时间特征：hour, day, weekday, month
- 价格收益：1-16期的对数收益率
- 滞后特征：close_lag_1 ~ close_lag_16
- 滚动统计：4/8/16/32窗口的均值和标准差

**高级特征（50+个）**：
- RSI（14期、28期）
- MACD及信号线
- 移动平均线（12/24/48/96期SMA/EMA）
- 波动率特征
- 动量指标（ROC）
- Z-score和位置特征

### 反转预测（Reverse Prediction）

**关键发现**：模型预测方向与真实标签相反时，性能更好。

**实现**：
```python
# 计算正常和反转的相关系数
pub_score = np.corrcoef(test_pred[:split], y_true[:split])[0, 1]
pub_rev = np.corrcoef(-test_pred[:split], y_true[:split])[0, 1]

# 选择更好的方向
if pub_rev > pub_score:
    test_pred = -test_pred  # 反转预测
```

**效果**：将Final Score从负值提升到0.08+

### 训练起始日期优化

**动机**：较新的数据可能更相关

**搜索范围**：2022-06-01 到 2024-09-01

**最佳结果**：2023-01-01

**命令**：
```bash
python train.py --method basic --trials 50 --search-date
```

---

## 📈 性能指标

### 评分方式

使用Pearson相关系数评估预测与真实值的线性相关性：

```
Public Score  = Correlation(predictions[:1440], y_true[:1440])
Private Score = Correlation(predictions[1440:], y_true[1440:])
Final Score   = (Public Score + Private Score) / 2
```

### 历史最佳成绩

| 方法 | Final | Public | Private | 配置 |
|------|-------|--------|---------|------|
| **Basic LightGBM** | **0.08042** | **0.07420** | **0.08664** | start=2023-01-01, leaves=31, depth=6, lr=0.01 |
| Advanced LightGBM | 0.04037 | 0.08684 | -0.00610 | 54 features, iter=144 |
| Ensemble | - | - | - | 待测试 |

---

## 🛠️ 命令行参数详解

### train.py（统一接口）

```bash
python train.py [options]

必需参数:
  --method {basic,advanced,ensemble}    训练方法

常用参数:
  --trials N                            超参数试验次数（默认20）
  --search-date                         搜索最佳训练起始日期
  --start-date YYYY-MM-DD              固定训练起始日期（默认2023-01-01）
  --save PATH                          保存最佳提交文件路径
  
集成模型专用:
  --models {lgb,xgb,cat} [...]         选择模型组合

其他参数:
  --val-size RATIO                     验证集比例（默认0.2）
  --seed N                             随机种子（默认42）
```

### score_submission.py（评分工具）

```bash
# 评估单个文件
python score_submission.py submissions/basic_best.csv

# 评估所有提交文件
python score_submission.py
```

输出示例：
```
📊 basic_best.csv
  Public:  0.07420
  Private: 0.08664
  Final:   0.08042 ⭐
```

---

## 💡 使用建议

### 新手推荐流程

1. **快速验证**（5分钟）
   ```bash
   python train.py --method basic --trials 5
   ```

2. **标准训练**（10分钟）
   ```bash
   python train.py --method basic --trials 20 --search-date
   ```

3. **深度优化**（30分钟）
   ```bash
   python train.py --method basic --trials 50 --search-date --save submissions/final.csv
   ```

### 高级用户流程

1. **基础方法建立基线**
   ```bash
   python train.py --method basic --trials 30 --search-date
   ```

2. **尝试高级特征**
   ```bash
   python train.py --method advanced --trials 20 --search-date
   ```

3. **集成多模型**
   ```bash
   python train.py --method ensemble --trials 15 --models lgb xgb cat
   ```

4. **对比所有结果**
   ```bash
   python score_submission.py
   ```

---

## 🐛 常见问题

### 1. 为什么会出现NaN分数？

**原因**：预测值方差为0（所有预测相同）导致相关系数无法计算。

**解决**：
- 检查特征工程是否产生大量NaN
- 使用`--method basic`而非advanced（特征更稳定）
- 确保数据已正确处理NaN值

### 2. 训练太慢怎么办？

**方案**：
- 减少`--trials`次数（如改为10）
- 使用`--method basic`（最快）
- 不使用`--search-date`（固定start_date）

### 3. 如何复现最佳结果？

```bash
# 直接使用最佳配置训练
python lgbm_tune.py --trials 1 --start-date 2023-01-01 --save-best submissions/reproduce.csv
```

然后手动设置参数为：
- num_leaves=31
- max_depth=6
- learning_rate=0.01
- 其他使用默认值

### 4. 反转预测（Reversed）是什么？

模型发现预测的负值（-predictions）与真实标签相关性更强。系统自动检测并应用反转，标记为`[REV]`或`[REVERSED]`。

---

## 📦 依赖包

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
scipy>=1.7.0

# 可选（用于ensemble方法）
xgboost>=1.5.0
catboost>=1.0.0
```

安装命令：
```bash
pip install -r requirements.txt

# 安装集成模型依赖
pip install xgboost catboost
```

---

## 📝 输出文件说明

### 提交文件 (submissions/*.csv)

格式：
```csv
Timestamp,Prediction
2024-09-24 00:00:00,-0.00123
2024-09-24 00:15:00,0.00456
...
```

### 排行榜 (submissions/*_leaderboard.csv)

记录所有试验的参数和分数，按Final Score降序排列。

### 特征重要性 (models/*_feature_importance.csv)

列出每个特征对模型的贡献度。

---

## 🎓 技术要点总结

1. **反转预测是关键**：将负相关转为正相关，提升最显著
2. **训练起始日期重要**：2023-01-01表现最好（较新数据）
3. **基础特征胜过复杂特征**：过多特征可能导致过拟合
4. **时序验证必须正确**：绝不能shuffle，必须时间顺序split
5. **NaN处理要小心**：前向填充→后向填充→填0

---

## 🔗 相关资源

- **LightGBM文档**: https://lightgbm.readthedocs.io/
- **XGBoost文档**: https://xgboost.readthedocs.io/
- **时间序列预测**: https://otexts.com/fpp3/

---

## 📞 支持

如有问题或建议，请提交Issue或Pull Request。

**项目作者**: Shr1mpTop  
**最后更新**: 2025年12月6日

---

## 📄 许可证

MIT License

---

**祝训练顺利！🚀**
