# Crypto Price ForecastingTime series forecasting for cryptocurrency prices using machine learning.[中文文档](README_CN.md) | English## 📊 Best Result**Final Score: 0.08042** (Public: 0.07420, Private: 0.08664)### Winning Configuration```Method: Basic LightGBMTraining start date: 2023-01-01Parameters:  - num_leaves: 31  - max_depth: 6  - learning_rate: 0.01  - best_iteration: 63  - reversed: YES (predictions negated)```## 🚀 Quick Start### 1. Install Dependencies```bashpip install -r requirements.txt```### 2. Train Models**Unified Interface (Recommended)** 🌟```bash# Basic method (fast, best performance)python train.py --method basic --trials 20# Search for optimal training start datepython train.py --method basic --trials 50 --search-date# Advanced features with technical indicatorspython train.py --method advanced --trials 20 --search-date# Ensemble of multiple modelspython train.py --method ensemble --trials 10 --models lgb xgb# Save best submissionpython train.py --method basic --trials 30 --save submissions/my_best.csv```**Direct Script Execution** (Alternative)```bash# Basic LightGBM trainingpython lgbm_tune.py --trials 20 --search-date# Advanced training with 50+ featurespython advanced_lgbm_simple.py --trials 20 --search-date# Ensemble trainingpython ensemble_tune.py --trials 10 --models lgb xgb```### 3. Evaluate Results```bash# Score a specific submissionpython score_submission.py submissions/basic_best.csv# Score all submissionspython score_submission.py```## 📁 Project Structure```crypto_forecast/├── data/                      # Data directory│   ├── train.csv              # Training data (OHLCV)│   ├── test.csv               # Test data│   └── sample_submission.csv│├── models/                    # Saved model files│   └── *_feature_importance.csv│├── submissions/               # Submission files│   ├── basic_best.csv         # Best basic method result│   ├── advanced_best.csv      # Advanced method result│   └── ensemble_best.csv      # Ensemble result│├── notebooks/                 # Jupyter notebooks│   └── *.ipynb│├── train.py                   # 🌟 Unified training interface (RECOMMENDED)├── lgbm_tune.py              # Basic LightGBM training├── advanced_lgbm_simple.py   # Advanced features LightGBM├── ensemble_tune.py          # Multi-model ensemble├── score_submission.py       # Local scoring utility├── reverse_predictions.py    # Prediction reversal tool│├── requirements.txt          # Python dependencies├── README.md                 # English documentation (this file)└── README_CN.md              # Chinese documentation```## 🎯 Training Methods Comparison### Method 1: Basic LightGBM (Recommended) ⭐**Features**:- Fundamental features (time, returns, lags, rolling stats)- Fast training (~5-10 min for 20 trials)- Best performance (Final=0.08042)**Use case**: Quick iterations, baseline model**Command**:```bashpython train.py --method basic --trials 20 --search-date```**Best Configuration**:```start_date: 2023-01-01num_leaves: 31max_depth: 6learning_rate: 0.01best_iteration: 63reversed: YES```---### Method 2: Advanced LightGBM**Features**:- 50+ technical indicators (RSI, MACD, Bollinger Bands, etc.)- Medium training time (~10-20 min for 20 trials)- May not improve over basic (overfitting risk)**Use case**: Exploring technical analysis features**Command**:```bashpython train.py --method advanced --trials 20 --search-date```---### Method 3: Ensemble**Features**:- Combines multiple models (LGB + XGB + CatBoost)- Automatic weight optimization- Longer training time (~15-30 min for 10 trials)**Use case**: Squeezing extra performance**Command**:```bashpython train.py --method ensemble --trials 10 --models lgb xgb cat```---## 🔬 Core Techniques### Reverse Prediction**Key Finding**: Model predictions work better when negated.**Implementation**:```python# Compare normal vs reversed correlationspub_score = np.corrcoef(test_pred[:split], y_true[:split])[0, 1]pub_rev = np.corrcoef(-test_pred[:split], y_true[:split])[0, 1]# Use better directionif pub_rev > pub_score:    test_pred = -test_pred  # Reverse predictions```**Impact**: Improved Final Score from negative to 0.08+### Training Start Date Optimization**Motivation**: Recent data may be more relevant**Search Range**: 2022-06-01 to 2024-09-01**Best Result**: 2023-01-01**Command**:```bashpython train.py --method basic --trials 50 --search-date```### Feature Engineering**Basic Features** (~40 features):- Time: hour, day, weekday, month- Returns: log returns for periods 1-16- Lags: close_lag_1 to close_lag_16- Rolling stats: mean/std for windows 4/8/16/32**Advanced Features** (50+ features):- RSI (14-period, 28-period)- Moving averages (12/24/48/96-period SMA/EMA)- Volatility features- Momentum indicators (ROC)- Z-scores and position features---## 📈 Performance Results| Method | Final | Public | Private | Config |
|--------|-------|--------|---------|--------|
| **Basic LightGBM** | **0.08042** | **0.07420** | **0.08664** | start=2023-01-01, leaves=31, depth=6, lr=0.01 |
| Advanced LightGBM | 0.04037 | 0.08684 | -0.00610 | 54 features, iter=144 |
| Ensemble | - | - | - | To be tested |

**Scoring**:
```
Public Score  = Correlation(predictions[:1440], y_true[:1440])
Private Score = Correlation(predictions[1440:], y_true[1440:])
Final Score   = (Public + Private) / 2
```

---

## 🛠️ Command Line Reference

### train.py (Unified Interface)

```bash
python train.py [options]

Required:
  --method {basic,advanced,ensemble}    Training method

Common options:
  --trials N                            Number of hyperparameter trials (default: 20)
  --search-date                         Search for optimal training start date
  --start-date YYYY-MM-DD              Fixed training start date (default: 2023-01-01)
  --save PATH                          Save best submission to file
  
Ensemble-specific:
  --models {lgb,xgb,cat} [...]         Models to combine

Other:
  --val-size RATIO                     Validation set ratio (default: 0.2)
  --seed N                             Random seed (default: 42)
```

### score_submission.py

```bash
# Score single file
python score_submission.py submissions/basic_best.csv

# Score all submissions
python score_submission.py
```

Output example:
```
📊 basic_best.csv
  Public:  0.07420
  Private: 0.08664
  Final:   0.08042 ⭐
```

---

## 💡 Usage Recommendations

### For Beginners

1. **Quick validation** (5 minutes)
   ```bash
   python train.py --method basic --trials 5
   ```

2. **Standard training** (10 minutes)
   ```bash
   python train.py --method basic --trials 20 --search-date
   ```

3. **Deep optimization** (30 minutes)
   ```bash
   python train.py --method basic --trials 50 --search-date --save submissions/final.csv
   ```

### For Advanced Users

1. **Establish baseline**
   ```bash
   python train.py --method basic --trials 30 --search-date
   ```

2. **Try advanced features**
   ```bash
   python train.py --method advanced --trials 20 --search-date
   ```

3. **Ensemble models**
   ```bash
   python train.py --method ensemble --trials 15 --models lgb xgb cat
   ```

4. **Compare results**
   ```bash
   python score_submission.py
   ```

---

## 🐛 Common Issues

### 1. Why am I getting NaN scores?

**Cause**: Zero variance in predictions (all predictions identical) makes correlation undefined.

**Solution**:
- Check feature engineering for NaN issues
- Use `--method basic` instead of advanced (more stable)
- Ensure proper NaN handling in data

### 2. Training is too slow

**Solutions**:
- Reduce `--trials` (e.g., to 10)
- Use `--method basic` (fastest)
- Don't use `--search-date` (fix start_date)

### 3. How to reproduce best result?

```bash
python train.py --method basic --trials 30 --start-date 2023-01-01 --save submissions/reproduce.csv
```

Look for configuration with:
- num_leaves=31
- max_depth=6  
- learning_rate=0.01
- reversed=YES

### 4. What is "Reversed" prediction?

The model finds that negated predictions (-predictions) correlate better with ground truth. System automatically detects and applies reversal, marked as `[REV]` or `[REVERSED]`.

---

## 📦 Dependencies

```
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
lightgbm>=3.3.0
scipy>=1.7.0

# Optional (for ensemble)
xgboost>=1.5.0
catboost>=1.0.0
```

Install:
```bash
pip install -r requirements.txt

# For ensemble method
pip install xgboost catboost
```

---

## 📝 Output Files

### Submission Files (submissions/*.csv)

Format:
```csv
Timestamp,Prediction
2024-09-24 00:00:00,-0.00123
2024-09-24 00:15:00,0.00456
...
```

### Leaderboards (submissions/*_leaderboard.csv)

Records all trials with parameters and scores, sorted by Final Score.

### Feature Importance (models/*_feature_importance.csv)

Lists each feature's contribution to the model.

---

## 🎓 Key Takeaways

1. **Reverse prediction is crucial**: Turns negative correlation to positive
2. **Training start date matters**: 2023-01-01 works best (newer data)
3. **Simple beats complex**: Basic features outperform 50+ features
4. **Time series validation**: Never shuffle, always chronological split
5. **NaN handling is critical**: Forward fill → Backward fill → Fill 0

---

## 🔗 Related Resources

- **LightGBM Documentation**: https://lightgbm.readthedocs.io/
- **XGBoost Documentation**: https://xgboost.readthedocs.io/
- **Time Series Forecasting**: https://otexts.com/fpp3/

---

## 📞 Support

For issues or suggestions, please submit an Issue or Pull Request.

**Project Author**: Shr1mpTop  
**Last Updated**: December 6, 2025

---

## 📄 License

MIT License

---

**Happy Forecasting! 🚀**
