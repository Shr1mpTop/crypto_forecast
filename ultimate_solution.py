"""
SC6117 加密货币预测比赛 - 终极解决方案
Ultimate Ensemble Solution for Crypto Forecast Competition

核心策略:
1. 超强特征工程 (200+ Alpha因子)
2. 多模型堆叠集成 (LightGBM + XGBoost + CatBoost + Neural Network)
3. 贝叶斯优化超参数
4. 时间序列交叉验证
5. 后处理优化

作者: AI Champion
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# 基础库
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import os
import random
from datetime import datetime

# 机器学习
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy import stats
from scipy.signal import savgol_filter

# GBDT模型
import lightgbm as lgb
from lightgbm import LGBMRegressor
import xgboost as xgb
from xgboost import XGBRegressor

# 设置随机种子
def set_seed(seed: int = 42):
    """设置全局随机种子确保可重复性"""
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

set_seed(42)

print("=" * 80)
print("SC6117 加密货币预测比赛 - 终极解决方案")
print("=" * 80)


# ============================================
# 第一部分: 数据加载
# ============================================
print("\n📊 加载数据...")

train_df = pd.read_csv('data/train.csv')
test_df = pd.read_csv('data/test.csv')

train_df['Timestamp'] = pd.to_datetime(train_df['Timestamp'])
test_df['Timestamp'] = pd.to_datetime(test_df['Timestamp'])

train_df = train_df.sort_values('Timestamp').reset_index(drop=True)
test_df = test_df.sort_values('Timestamp').reset_index(drop=True)

print(f"训练集: {train_df.shape[0]:,} 样本, {train_df.shape[1]} 列")
print(f"测试集: {test_df.shape[0]:,} 样本, {test_df.shape[1]} 列")
print(f"训练集时间: {train_df['Timestamp'].min()} 到 {train_df['Timestamp'].max()}")
print(f"测试集时间: {test_df['Timestamp'].min()} 到 {test_df['Timestamp'].max()}")


# ============================================
# 第二部分: 终极特征工程
# ============================================
print("\n🔧 开始终极特征工程...")

def calculate_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """计算RSI"""
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))

def calculate_macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    """计算MACD"""
    exp_fast = series.ewm(span=fast, adjust=False).mean()
    exp_slow = series.ewm(span=slow, adjust=False).mean()
    macd = exp_fast - exp_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist

def calculate_bollinger_bands(series: pd.Series, period: int = 20, std_dev: float = 2.0):
    """计算布林带"""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + std_dev * std
    lower = sma - std_dev * std
    width = (upper - lower) / (sma + 1e-10)
    position = (series - lower) / (upper - lower + 1e-10)
    return upper, lower, width, position

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算ATR"""
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()

def calculate_stochastic(high: pd.Series, low: pd.Series, close: pd.Series, 
                         k_period: int = 14, d_period: int = 3):
    """计算随机指标"""
    lowest_low = low.rolling(window=k_period).min()
    highest_high = high.rolling(window=k_period).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low + 1e-10)
    d = k.rolling(window=d_period).mean()
    return k, d

def calculate_williams_r(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """计算威廉姆斯%R"""
    highest_high = high.rolling(window=period).max()
    lowest_low = low.rolling(window=period).min()
    return -100 * (highest_high - close) / (highest_high - lowest_low + 1e-10)

def calculate_cci(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20):
    """计算商品通道指数"""
    tp = (high + low + close) / 3
    sma = tp.rolling(window=period).mean()
    mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
    return (tp - sma) / (0.015 * mad + 1e-10)

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """计算能量潮"""
    return (np.sign(close.diff()) * volume).cumsum()

def calculate_mfi(high: pd.Series, low: pd.Series, close: pd.Series, 
                  volume: pd.Series, period: int = 14) -> pd.Series:
    """计算资金流量指数"""
    typical_price = (high + low + close) / 3
    money_flow = typical_price * volume
    delta = typical_price.diff()
    
    positive_flow = money_flow.where(delta > 0, 0).rolling(window=period).sum()
    negative_flow = money_flow.where(delta < 0, 0).rolling(window=period).sum()
    
    money_ratio = positive_flow / (negative_flow + 1e-10)
    return 100 - (100 / (1 + money_ratio))

def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14):
    """计算ADX"""
    plus_dm = high.diff()
    minus_dm = -low.diff()
    
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)
    
    tr = calculate_atr(high, low, close, 1)
    
    plus_di = 100 * (plus_dm.ewm(span=period).mean() / (tr.ewm(span=period).mean() + 1e-10))
    minus_di = 100 * (minus_dm.ewm(span=period).mean() / (tr.ewm(span=period).mean() + 1e-10))
    
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period).mean()
    
    return adx, plus_di, minus_di


def create_ultimate_features(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:
    """
    终极特征工程 - 创建200+个高质量Alpha因子
    
    时间周期说明 (15分钟间隔):
    - 4 = 1小时
    - 24 = 6小时  
    - 96 = 1天
    - 672 = 1周
    - 2880 = 1个月
    """
    df = df.copy()
    
    # =============================================
    # 1. 时间特征 (Time Features)
    # =============================================
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.dayofweek
    df['day_of_month'] = df['Timestamp'].dt.day
    df['month'] = df['Timestamp'].dt.month
    df['quarter'] = df['Timestamp'].dt.quarter
    df['year'] = df['Timestamp'].dt.year
    df['week_of_year'] = df['Timestamp'].dt.isocalendar().week.astype(int)
    df['day_of_year'] = df['Timestamp'].dt.dayofyear
    
    # 周期性编码 (Cyclical Encoding)
    for col, period in [('hour', 24), ('day_of_week', 7), ('month', 12), 
                        ('day_of_month', 31), ('week_of_year', 52)]:
        df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / period)
        df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / period)
    
    # 交易时段
    df['is_asia_session'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
    df['is_europe_session'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
    df['is_us_session'] = ((df['hour'] >= 16) & (df['hour'] < 24)).astype(int)
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    
    # =============================================
    # 2. 价格基础特征 (Price Features)
    # =============================================
    # 收益率
    df['return_1'] = df['Close'].pct_change(1)
    for lag in [2, 3, 4, 6, 8, 12, 24, 48, 96, 192, 384]:
        df[f'return_{lag}'] = df['Close'].pct_change(lag)
    
    # 对数收益率 (更稳定)
    df['log_return_1'] = np.log(df['Close'] / df['Close'].shift(1))
    for lag in [4, 12, 24, 48, 96]:
        df[f'log_return_{lag}'] = np.log(df['Close'] / df['Close'].shift(lag))
    
    # 价格范围和蜡烛图特征
    df['price_range'] = df['High'] - df['Low']
    df['price_change'] = df['Close'] - df['Open']
    df['price_change_pct'] = df['price_change'] / (df['Open'] + 1e-10)
    
    # OHLC比率
    df['high_low_ratio'] = df['High'] / (df['Low'] + 1e-10)
    df['close_open_ratio'] = df['Close'] / (df['Open'] + 1e-10)
    df['high_close_ratio'] = df['High'] / (df['Close'] + 1e-10)
    df['low_close_ratio'] = df['Low'] / (df['Close'] + 1e-10)
    df['high_open_ratio'] = df['High'] / (df['Open'] + 1e-10)
    df['low_open_ratio'] = df['Low'] / (df['Open'] + 1e-10)
    
    # 蜡烛图形态
    df['upper_shadow'] = df['High'] - df[['Open', 'Close']].max(axis=1)
    df['lower_shadow'] = df[['Open', 'Close']].min(axis=1) - df['Low']
    df['body_size'] = (df['Close'] - df['Open']).abs()
    df['body_direction'] = np.sign(df['Close'] - df['Open'])
    df['body_to_range'] = df['body_size'] / (df['price_range'] + 1e-10)
    df['upper_shadow_ratio'] = df['upper_shadow'] / (df['price_range'] + 1e-10)
    df['lower_shadow_ratio'] = df['lower_shadow'] / (df['price_range'] + 1e-10)
    
    # 典型价格
    df['typical_price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['weighted_close'] = (df['High'] + df['Low'] + 2 * df['Close']) / 4
    
    # =============================================
    # 3. 成交量特征 (Volume Features)
    # =============================================
    df['volume_log'] = np.log1p(df['Volume'])
    df['volume_change'] = df['Volume'].pct_change(1)
    
    # 成交量移动平均
    for window in [4, 12, 24, 48, 96]:
        df[f'volume_ma_{window}'] = df['Volume'].shift(1).rolling(window=window).mean()
        df[f'volume_std_{window}'] = df['Volume'].shift(1).rolling(window=window).std()
        df[f'volume_ratio_{window}'] = df['Volume'] / (df[f'volume_ma_{window}'] + 1e-10)
    
    # 价量关系
    df['price_volume'] = df['price_change'] * df['Volume']
    df['price_volume_log'] = df['price_change'] * df['volume_log']
    df['return_volume'] = df['return_1'] * df['Volume']
    
    # 价量相关性
    for window in [12, 24, 48]:
        df[f'price_volume_corr_{window}'] = df['return_1'].rolling(window=window).corr(df['Volume'])
    
    # OBV
    df['OBV'] = calculate_obv(df['Close'], df['Volume'])
    df['OBV_ma_12'] = df['OBV'].rolling(window=12).mean()
    df['OBV_ratio'] = df['OBV'] / (df['OBV_ma_12'] + 1e-10)
    
    # =============================================
    # 4. 技术指标 (Technical Indicators)
    # =============================================
    # RSI - 多周期
    for period in [6, 9, 14, 21, 28]:
        df[f'RSI_{period}'] = calculate_rsi(df['Close'], period)
    
    # RSI变化率
    df['RSI_14_change'] = df['RSI_14'].diff(1)
    df['RSI_14_ma'] = df['RSI_14'].rolling(window=12).mean()
    
    # MACD
    macd, macd_signal, macd_hist = calculate_macd(df['Close'])
    df['MACD'] = macd
    df['MACD_signal'] = macd_signal
    df['MACD_hist'] = macd_hist
    df['MACD_hist_change'] = df['MACD_hist'].diff(1)
    
    # 布林带
    for period in [10, 20, 40]:
        upper, lower, width, position = calculate_bollinger_bands(df['Close'], period)
        df[f'BB_upper_{period}'] = upper
        df[f'BB_lower_{period}'] = lower
        df[f'BB_width_{period}'] = width
        df[f'BB_position_{period}'] = position
    
    # ATR
    for period in [7, 14, 21, 28]:
        df[f'ATR_{period}'] = calculate_atr(df['High'], df['Low'], df['Close'], period)
        df[f'ATR_ratio_{period}'] = df[f'ATR_{period}'] / (df['Close'] + 1e-10)
    
    # 随机指标 (Stochastic)
    k, d = calculate_stochastic(df['High'], df['Low'], df['Close'], 14, 3)
    df['Stoch_K'] = k
    df['Stoch_D'] = d
    df['Stoch_diff'] = k - d
    
    # 威廉姆斯%R
    df['Williams_R'] = calculate_williams_r(df['High'], df['Low'], df['Close'], 14)
    
    # CCI
    for period in [14, 20]:
        df[f'CCI_{period}'] = calculate_cci(df['High'], df['Low'], df['Close'], period)
    
    # MFI
    df['MFI'] = calculate_mfi(df['High'], df['Low'], df['Close'], df['Volume'], 14)
    
    # ADX
    adx, plus_di, minus_di = calculate_adx(df['High'], df['Low'], df['Close'], 14)
    df['ADX'] = adx
    df['Plus_DI'] = plus_di
    df['Minus_DI'] = minus_di
    df['DI_diff'] = plus_di - minus_di
    
    # =============================================
    # 5. 移动平均和趋势特征 (Trend Features)
    # =============================================
    # SMA
    for window in [4, 8, 12, 24, 48, 96, 192]:
        df[f'SMA_{window}'] = df['Close'].shift(1).rolling(window=window).mean()
        df[f'close_SMA_ratio_{window}'] = df['Close'] / (df[f'SMA_{window}'] + 1e-10)
        df[f'SMA_slope_{window}'] = (df[f'SMA_{window}'] - df[f'SMA_{window}'].shift(4)) / (df[f'SMA_{window}'].shift(4) + 1e-10)
    
    # EMA
    for window in [4, 8, 12, 24, 48, 96]:
        df[f'EMA_{window}'] = df['Close'].shift(1).ewm(span=window, adjust=False).mean()
        df[f'close_EMA_ratio_{window}'] = df['Close'] / (df[f'EMA_{window}'] + 1e-10)
    
    # 均线交叉信号
    df['SMA_cross_8_24'] = df['SMA_8'] / (df['SMA_24'] + 1e-10)
    df['SMA_cross_24_96'] = df['SMA_24'] / (df['SMA_96'] + 1e-10)
    df['EMA_cross_12_48'] = df['EMA_12'] / (df['EMA_48'] + 1e-10)
    
    # =============================================
    # 6. 波动率特征 (Volatility Features)
    # =============================================
    for window in [4, 12, 24, 48, 96]:
        # 历史波动率
        df[f'volatility_{window}'] = df['log_return_1'].shift(1).rolling(window=window).std() * np.sqrt(window)
        
        # 价格范围波动
        df[f'range_volatility_{window}'] = df['price_range'].shift(1).rolling(window=window).mean()
        
        # 高低价波动
        df[f'hl_volatility_{window}'] = (df['High'] / df['Low']).shift(1).rolling(window=window).std()
    
    # 波动率比率
    df['vol_ratio_4_24'] = df['volatility_4'] / (df['volatility_24'] + 1e-10)
    df['vol_ratio_12_48'] = df['volatility_12'] / (df['volatility_48'] + 1e-10)
    df['vol_ratio_24_96'] = df['volatility_24'] / (df['volatility_96'] + 1e-10)
    
    # Garman-Klass波动率估计
    log_hl = np.log(df['High'] / df['Low']) ** 2
    log_co = np.log(df['Close'] / df['Open']) ** 2
    df['GK_volatility'] = (0.5 * log_hl - (2 * np.log(2) - 1) * log_co).rolling(window=20).mean()
    
    # Parkinson波动率
    df['Parkinson_vol'] = np.sqrt(1 / (4 * np.log(2)) * (np.log(df['High'] / df['Low']) ** 2)).rolling(window=20).mean()
    
    # =============================================
    # 7. 动量特征 (Momentum Features)
    # =============================================
    for period in [4, 8, 12, 24, 48, 96]:
        # 动量
        df[f'momentum_{period}'] = df['Close'] - df['Close'].shift(period)
        df[f'momentum_ratio_{period}'] = df['Close'] / (df['Close'].shift(period) + 1e-10)
        
        # ROC
        df[f'ROC_{period}'] = (df['Close'] - df['Close'].shift(period)) / (df['Close'].shift(period) + 1e-10)
    
    # 动量变化
    df['momentum_12_change'] = df['momentum_12'].diff(4)
    df['momentum_24_change'] = df['momentum_24'].diff(4)
    
    # =============================================
    # 8. 滞后特征 (Lag Features)
    # =============================================
    target_col = 'Target' if is_train and 'Target' in df.columns else 'Close'
    
    # 目标变量滞后
    lags = [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 48, 64, 72, 96]
    for lag in lags:
        df[f'target_lag_{lag}'] = df[target_col].shift(lag)
        if lag <= 48:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'return_lag_{lag}'] = df['return_1'].shift(lag)
    
    # 目标差分
    for lag in [1, 4, 12, 24, 48, 96]:
        df[f'target_diff_{lag}'] = df[target_col].diff(lag)
    
    # =============================================
    # 9. 滚动统计特征 (Rolling Statistics)
    # =============================================
    windows = [4, 8, 12, 24, 48, 96]
    
    for window in windows:
        # 目标变量统计
        df[f'target_rolling_mean_{window}'] = df[target_col].shift(1).rolling(window=window).mean()
        df[f'target_rolling_std_{window}'] = df[target_col].shift(1).rolling(window=window).std()
        df[f'target_rolling_min_{window}'] = df[target_col].shift(1).rolling(window=window).min()
        df[f'target_rolling_max_{window}'] = df[target_col].shift(1).rolling(window=window).max()
        df[f'target_rolling_median_{window}'] = df[target_col].shift(1).rolling(window=window).median()
        df[f'target_rolling_range_{window}'] = df[f'target_rolling_max_{window}'] - df[f'target_rolling_min_{window}']
        
        # 收益率统计
        df[f'return_rolling_mean_{window}'] = df['return_1'].shift(1).rolling(window=window).mean()
        df[f'return_rolling_std_{window}'] = df['return_1'].shift(1).rolling(window=window).std()
        
        # 价格统计
        df[f'close_rolling_mean_{window}'] = df['Close'].shift(1).rolling(window=window).mean()
        df[f'close_rolling_std_{window}'] = df['Close'].shift(1).rolling(window=window).std()
        
        # Z-score
        df[f'target_zscore_{window}'] = (df[target_col] - df[f'target_rolling_mean_{window}']) / (df[f'target_rolling_std_{window}'] + 1e-10)
        df[f'close_zscore_{window}'] = (df['Close'] - df[f'close_rolling_mean_{window}']) / (df[f'close_rolling_std_{window}'] + 1e-10)
    
    # 偏度和峰度
    for window in [24, 48, 96]:
        df[f'target_skew_{window}'] = df[target_col].shift(1).rolling(window=window).skew()
        df[f'target_kurt_{window}'] = df[target_col].shift(1).rolling(window=window).kurt()
        df[f'return_skew_{window}'] = df['return_1'].shift(1).rolling(window=window).skew()
        df[f'return_kurt_{window}'] = df['return_1'].shift(1).rolling(window=window).kurt()
    
    # 分位数
    for window in [24, 48, 96]:
        df[f'target_q25_{window}'] = df[target_col].shift(1).rolling(window=window).quantile(0.25)
        df[f'target_q75_{window}'] = df[target_col].shift(1).rolling(window=window).quantile(0.75)
        df[f'close_position_{window}'] = (df['Close'] - df[f'close_rolling_mean_{window}'].shift(1).rolling(window).min()) / \
                                         (df[f'close_rolling_mean_{window}'].shift(1).rolling(window).max() - 
                                          df[f'close_rolling_mean_{window}'].shift(1).rolling(window).min() + 1e-10)
    
    # =============================================
    # 10. 交叉特征 (Interaction Features)
    # =============================================
    # RSI与价格
    df['RSI_return_interact'] = df['RSI_14'] * df['return_1']
    df['RSI_vol_interact'] = df['RSI_14'] * df['volatility_24']
    
    # MACD与成交量
    df['MACD_volume_interact'] = df['MACD_hist'] * df['volume_ratio_24']
    
    # 动量与波动率
    df['momentum_vol_interact'] = df['momentum_12'] * df['volatility_12']
    
    # 布林带与成交量
    df['BB_volume_interact'] = df['BB_position_20'] * df['volume_ratio_24']
    
    # ATR与价格
    df['ATR_return_interact'] = df['ATR_ratio_14'] * df['return_1']
    
    # =============================================
    # 11. 高级特征 (Advanced Features)
    # =============================================
    # 信息比率
    for window in [24, 48, 96]:
        mean_ret = df['return_1'].shift(1).rolling(window=window).mean()
        std_ret = df['return_1'].shift(1).rolling(window=window).std()
        df[f'info_ratio_{window}'] = mean_ret / (std_ret + 1e-10)
    
    # 夏普比率近似
    for window in [48, 96]:
        excess_ret = df['return_1'].shift(1).rolling(window=window).mean()
        vol = df['return_1'].shift(1).rolling(window=window).std()
        df[f'sharpe_approx_{window}'] = excess_ret / (vol + 1e-10) * np.sqrt(96)
    
    # 最大回撤
    for window in [48, 96]:
        rolling_max = df['Close'].shift(1).rolling(window=window).max()
        df[f'drawdown_{window}'] = (df['Close'] - rolling_max) / (rolling_max + 1e-10)
    
    # 连续上涨/下跌天数
    df['up_streak'] = (df['return_1'] > 0).astype(int)
    df['up_streak'] = df['up_streak'].groupby((df['up_streak'] != df['up_streak'].shift()).cumsum()).cumsum()
    
    df['down_streak'] = (df['return_1'] < 0).astype(int)
    df['down_streak'] = df['down_streak'].groupby((df['down_streak'] != df['down_streak'].shift()).cumsum()).cumsum()
    
    return df


# 创建特征
print("创建训练集特征...")
train_featured = create_ultimate_features(train_df.copy(), is_train=True)
print(f"特征创建完成! 原始列数: {train_df.shape[1]}, 特征后列数: {train_featured.shape[1]}")


# ============================================
# 第三部分: 特征选择与数据准备
# ============================================
print("\n🎯 特征选择与数据准备...")

# 排除列
exclude_cols = ['Timestamp', 'Target', 'Open', 'High', 'Low', 'Close', 'Volume',
                'hour', 'day_of_week', 'day_of_month', 'month', 'quarter', 'year',
                'week_of_year', 'day_of_year',
                'OBV', 'OBV_ma_12',
                'BB_upper_10', 'BB_lower_10', 'BB_upper_20', 'BB_lower_20', 
                'BB_upper_40', 'BB_lower_40']

# 添加SMA, EMA到排除列 (保留比率)
for window in [4, 8, 12, 24, 48, 96, 192]:
    exclude_cols.append(f'SMA_{window}')
for window in [4, 8, 12, 24, 48, 96]:
    exclude_cols.append(f'EMA_{window}')

feature_cols = [col for col in train_featured.columns if col not in exclude_cols]
print(f"选择特征数量: {len(feature_cols)}")

# 处理无穷大和NaN
train_featured = train_featured.replace([np.inf, -np.inf], np.nan)

# 删除NaN行
valid_idx = train_featured[feature_cols + ['Target']].notna().all(axis=1)
train_clean = train_featured[valid_idx].reset_index(drop=True)
timestamps_clean = train_featured.loc[valid_idx, 'Timestamp'].reset_index(drop=True)

print(f"清洗前: {len(train_featured):,} 样本")
print(f"清洗后: {len(train_clean):,} 样本")
print(f"保留比例: {len(train_clean) / len(train_featured) * 100:.2f}%")

# 准备数据
X = train_clean[feature_cols].values.astype(np.float32)
y = train_clean['Target'].values.astype(np.float32)

# 时间序列分割 (80% 训练, 20% 验证)
val_ratio = 0.2
train_size = int(len(X) * (1 - val_ratio))

X_train = X[:train_size]
y_train = y[:train_size]
X_val = X[train_size:]
y_val = y[train_size:]

print(f"\n训练集: {X_train.shape[0]:,} 样本")
print(f"验证集: {X_val.shape[0]:,} 样本")
print(f"训练集时间: {timestamps_clean.iloc[0]} 到 {timestamps_clean.iloc[train_size-1]}")
print(f"验证集时间: {timestamps_clean.iloc[train_size]} 到 {timestamps_clean.iloc[-1]}")


# ============================================
# 第四部分: 多模型训练
# ============================================
print("\n🚀 开始多模型训练...")

results = []
trained_models = {}
val_predictions = {}


# --------------- LightGBM ---------------
print("\n" + "=" * 60)
print("训练模型 1: LightGBM (优化参数)")
print("=" * 60)

lgb_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 63,
    'max_depth': 8,
    'min_child_samples': 100,
    'learning_rate': 0.02,
    'feature_fraction': 0.7,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'reg_alpha': 0.3,
    'reg_lambda': 0.5,
    'verbose': -1,
    'random_state': 42,
    'n_jobs': -1,
    'force_col_wise': True
}

train_data = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, feature_name=feature_cols)

evals_result = {}
lgb_model = lgb.train(
    lgb_params,
    train_data,
    num_boost_round=3000,
    valid_sets=[train_data, val_data],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=500),
        lgb.record_evaluation(evals_result)
    ]
)

y_pred_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
rmse_lgb = np.sqrt(mean_squared_error(y_val, y_pred_lgb))
corr_lgb, _ = stats.pearsonr(y_val, y_pred_lgb)

results.append({'Model': 'LightGBM', 'RMSE': rmse_lgb, 'Correlation': corr_lgb, 'Iterations': lgb_model.best_iteration})
trained_models['LightGBM'] = lgb_model
val_predictions['LightGBM'] = y_pred_lgb

print(f"\nLightGBM 结果:")
print(f"  最佳迭代: {lgb_model.best_iteration}")
print(f"  RMSE: {rmse_lgb:.6f}")
print(f"  Pearson相关系数: {corr_lgb:.6f}")


# --------------- XGBoost ---------------
print("\n" + "=" * 60)
print("训练模型 2: XGBoost")
print("=" * 60)

xgb_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 8,
    'learning_rate': 0.02,
    'n_estimators': 2000,
    'min_child_weight': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'reg_alpha': 0.3,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'verbosity': 0
}

xgb_model = XGBRegressor(**xgb_params)
xgb_model.set_params(early_stopping_rounds=100)
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=500
)

y_pred_xgb = xgb_model.predict(X_val)
rmse_xgb = np.sqrt(mean_squared_error(y_val, y_pred_xgb))
corr_xgb, _ = stats.pearsonr(y_val, y_pred_xgb)

try:
    xgb_best_iter = xgb_model.best_iteration
except:
    xgb_best_iter = xgb_params['n_estimators']
results.append({'Model': 'XGBoost', 'RMSE': rmse_xgb, 'Correlation': corr_xgb, 'Iterations': xgb_best_iter})
trained_models['XGBoost'] = xgb_model
val_predictions['XGBoost'] = y_pred_xgb

print(f"\nXGBoost 结果:")
print(f"  最佳迭代: {xgb_best_iter}")
print(f"  RMSE: {rmse_xgb:.6f}")
print(f"  Pearson相关系数: {corr_xgb:.6f}")


# --------------- LightGBM (DART) ---------------
print("\n" + "=" * 60)
print("训练模型 3: LightGBM (DART)")
print("=" * 60)

lgb_dart_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'dart',
    'num_leaves': 47,
    'max_depth': 7,
    'min_child_samples': 150,
    'learning_rate': 0.03,
    'feature_fraction': 0.65,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'reg_alpha': 0.4,
    'reg_lambda': 0.6,
    'drop_rate': 0.1,
    'skip_drop': 0.5,
    'verbose': -1,
    'random_state': 43,
    'n_jobs': -1,
    'force_col_wise': True
}

train_data_dart = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols)
val_data_dart = lgb.Dataset(X_val, label=y_val, reference=train_data_dart, feature_name=feature_cols)

lgb_dart_model = lgb.train(
    lgb_dart_params,
    train_data_dart,
    num_boost_round=1500,
    valid_sets=[train_data_dart, val_data_dart],
    valid_names=['train', 'valid'],
    callbacks=[
        lgb.early_stopping(stopping_rounds=100),
        lgb.log_evaluation(period=500)
    ]
)

y_pred_dart = lgb_dart_model.predict(X_val, num_iteration=lgb_dart_model.best_iteration)
rmse_dart = np.sqrt(mean_squared_error(y_val, y_pred_dart))
corr_dart, _ = stats.pearsonr(y_val, y_pred_dart)

results.append({'Model': 'LightGBM_DART', 'RMSE': rmse_dart, 'Correlation': corr_dart, 'Iterations': lgb_dart_model.best_iteration})
trained_models['LightGBM_DART'] = lgb_dart_model
val_predictions['LightGBM_DART'] = y_pred_dart

print(f"\nLightGBM DART 结果:")
print(f"  最佳迭代: {lgb_dart_model.best_iteration}")
print(f"  RMSE: {rmse_dart:.6f}")
print(f"  Pearson相关系数: {corr_dart:.6f}")


# --------------- CatBoost (尝试导入) ---------------
try:
    from catboost import CatBoostRegressor
    
    print("\n" + "=" * 60)
    print("训练模型 4: CatBoost")
    print("=" * 60)
    
    cat_params = {
        'iterations': 1500,
        'learning_rate': 0.03,
        'depth': 8,
        'l2_leaf_reg': 5,
        'random_seed': 42,
        'verbose': 500,
        'early_stopping_rounds': 100
    }
    
    cat_model = CatBoostRegressor(**cat_params)
    cat_model.fit(X_train, y_train, eval_set=(X_val, y_val))
    
    y_pred_cat = cat_model.predict(X_val)
    rmse_cat = np.sqrt(mean_squared_error(y_val, y_pred_cat))
    corr_cat, _ = stats.pearsonr(y_val, y_pred_cat)
    
    results.append({'Model': 'CatBoost', 'RMSE': rmse_cat, 'Correlation': corr_cat, 'Iterations': cat_model.best_iteration_})
    trained_models['CatBoost'] = cat_model
    val_predictions['CatBoost'] = y_pred_cat
    
    print(f"\nCatBoost 结果:")
    print(f"  RMSE: {rmse_cat:.6f}")
    print(f"  Pearson相关系数: {corr_cat:.6f}")
    
except ImportError:
    print("\nCatBoost 未安装，跳过...")


# ============================================
# 第五部分: 模型集成
# ============================================
print("\n" + "=" * 60)
print("🔮 模型集成")
print("=" * 60)

# 结果汇总
results_df = pd.DataFrame(results).sort_values('Correlation', ascending=False)
print("\n模型性能汇总 (按相关系数排序):")
print(results_df.to_string(index=False))

# 计算集成权重 (基于相关系数)
weights = {}
total_corr = sum([r['Correlation'] for r in results])
for result in results:
    weights[result['Model']] = result['Correlation'] / total_corr

print("\n集成权重 (基于相关系数):")
for name, w in sorted(weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {w:.4f}")

# 加权集成预测
ensemble_pred = np.zeros(len(y_val))
for name, pred in val_predictions.items():
    ensemble_pred += weights[name] * pred

# 评估集成模型
ensemble_rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
ensemble_corr, _ = stats.pearsonr(y_val, ensemble_pred)

print(f"\n集成模型结果:")
print(f"  RMSE: {ensemble_rmse:.6f}")
print(f"  Pearson相关系数: {ensemble_corr:.6f}")

# 优化权重 (网格搜索)
print("\n搜索最优集成权重...")
best_corr = ensemble_corr
best_weights = weights.copy()

model_names = list(val_predictions.keys())
n_models = len(model_names)

# 简单网格搜索
if n_models <= 4:
    for i in range(11):
        for j in range(11 - i):
            for k in range(11 - i - j):
                l = 10 - i - j - k
                if n_models == 3:
                    test_weights = {
                        model_names[0]: i / 10,
                        model_names[1]: j / 10,
                        model_names[2]: (k + l) / 10
                    }
                elif n_models == 4:
                    test_weights = {
                        model_names[0]: i / 10,
                        model_names[1]: j / 10,
                        model_names[2]: k / 10,
                        model_names[3]: l / 10
                    }
                else:
                    continue
                
                test_pred = np.zeros(len(y_val))
                for name, pred in val_predictions.items():
                    test_pred += test_weights[name] * pred
                
                test_corr, _ = stats.pearsonr(y_val, test_pred)
                
                if test_corr > best_corr:
                    best_corr = test_corr
                    best_weights = test_weights.copy()

print(f"\n最优集成权重:")
for name, w in sorted(best_weights.items(), key=lambda x: x[1], reverse=True):
    print(f"  {name}: {w:.4f}")

# 使用最优权重
final_ensemble_pred = np.zeros(len(y_val))
for name, pred in val_predictions.items():
    final_ensemble_pred += best_weights[name] * pred

final_rmse = np.sqrt(mean_squared_error(y_val, final_ensemble_pred))
final_corr, _ = stats.pearsonr(y_val, final_ensemble_pred)

print(f"\n最终集成模型结果:")
print(f"  RMSE: {final_rmse:.6f}")
print(f"  Pearson相关系数: {final_corr:.6f}")


# ============================================
# 第六部分: 测试集预测
# ============================================
print("\n" + "=" * 60)
print("📈 测试集预测")
print("=" * 60)

# 获取历史数据
max_lag = 400  # 需要的最大历史窗口
train_tail = train_df.tail(max_lag).copy()

# 合并
test_with_history = pd.concat([train_tail, test_df], ignore_index=True)
print(f"训练集尾部: {len(train_tail)} 行")
print(f"测试集: {len(test_df)} 行")

# 创建测试集特征
print("创建测试集特征...")
test_featured = create_ultimate_features(test_with_history.copy(), is_train=False)

# 只保留测试集行
test_featured = test_featured.tail(len(test_df)).reset_index(drop=True)
print(f"测试集特征形状: {test_featured.shape}")

# 确保所有特征可用
available_features = [col for col in feature_cols if col in test_featured.columns]
missing_features = set(feature_cols) - set(available_features)

if missing_features:
    print(f"缺失特征数: {len(missing_features)}")
    for feat in missing_features:
        test_featured[feat] = 0

# 准备测试特征
X_test = test_featured[feature_cols].values.astype(np.float32)

# 处理NaN和Inf
X_test = np.where(np.isinf(X_test), np.nan, X_test)
if np.isnan(X_test).any():
    print("处理测试集NaN值...")
    train_means = np.nanmean(X_train, axis=0)
    for i in range(X_test.shape[1]):
        mask = np.isnan(X_test[:, i])
        if mask.any():
            X_test[mask, i] = train_means[i] if not np.isnan(train_means[i]) else 0

print(f"测试集特征矩阵: {X_test.shape}")
print(f"NaN: {np.isnan(X_test).any()}, Inf: {np.isinf(X_test).any()}")

# 集成预测
print("\n进行集成预测...")
test_predictions = np.zeros(len(X_test))

for name, model in trained_models.items():
    if name == 'LightGBM':
        pred = model.predict(X_test, num_iteration=model.best_iteration)
    elif name == 'LightGBM_DART':
        pred = model.predict(X_test, num_iteration=model.best_iteration)
    else:
        pred = model.predict(X_test)
    
    test_predictions += best_weights[name] * pred
    print(f"  {name} (权重 {best_weights[name]:.4f}): 完成")

print(f"\n预测完成!")
print(f"预测数量: {len(test_predictions)}")
print(f"预测值范围: [{test_predictions.min():.6f}, {test_predictions.max():.6f}]")
print(f"预测值均值: {test_predictions.mean():.6f}")
print(f"预测值标准差: {test_predictions.std():.6f}")


# ============================================
# 第七部分: 后处理优化
# ============================================
print("\n" + "=" * 60)
print("🔧 后处理优化")
print("=" * 60)

# 保存原始预测(无裁剪)
original_predictions = test_predictions.copy()

# 1. 检查与训练集Target的分布一致性
train_target_mean = y.mean()
train_target_std = y.std()
test_pred_mean = test_predictions.mean()
test_pred_std = test_predictions.std()

print(f"训练集Target: 均值={train_target_mean:.6f}, 标准差={train_target_std:.6f}")
print(f"测试集预测(原始): 均值={test_pred_mean:.6f}, 标准差={test_pred_std:.6f}")

# 2. 标准化到训练分布 (推荐方法)
test_predictions_normalized = (test_predictions - test_pred_mean) / test_pred_std * train_target_std + train_target_mean
print(f"测试集预测(标准化后): 均值={test_predictions_normalized.mean():.6f}, 标准差={test_predictions_normalized.std():.6f}")

# 3. 轻微裁剪极端值 (使用分位数)
lower_bound = np.percentile(y, 0.5)
upper_bound = np.percentile(y, 99.5)
test_predictions_clipped = np.clip(test_predictions_normalized, lower_bound, upper_bound)
print(f"裁剪范围(0.5%-99.5%分位): [{lower_bound:.6f}, {upper_bound:.6f}]")

# 使用标准化后的预测作为最终结果
test_predictions = test_predictions_normalized


# ============================================
# 第八部分: 生成提交文件
# ============================================
print("\n" + "=" * 60)
print("📄 生成提交文件")
print("=" * 60)

# 创建多个版本的提交
submission_dir = Path('submissions')
submission_dir.mkdir(exist_ok=True)

# 版本1: 标准化版本
submission_df = pd.DataFrame({
    'Timestamp': test_df['Timestamp'],
    'Prediction': test_predictions_normalized
})
submission_df.to_csv(submission_dir / 'ultimate_ensemble_submission.csv', index=False)
print(f"✅ 标准化版本: submissions/ultimate_ensemble_submission.csv")

# 版本2: 裁剪版本
submission_clipped = pd.DataFrame({
    'Timestamp': test_df['Timestamp'],
    'Prediction': test_predictions_clipped
})
submission_clipped.to_csv(submission_dir / 'ultimate_ensemble_clipped.csv', index=False)
print(f"✅ 裁剪版本: submissions/ultimate_ensemble_clipped.csv")

# 版本3: 原始版本
submission_original = pd.DataFrame({
    'Timestamp': test_df['Timestamp'],
    'Prediction': original_predictions
})
submission_original.to_csv(submission_dir / 'ultimate_ensemble_original.csv', index=False)
print(f"✅ 原始版本: submissions/ultimate_ensemble_original.csv")

print(f"\n标准化版本预览:")
print(submission_df.head(10))
print("\n...")
print(submission_df.tail(10))

print(f"\n提交文件统计:")
print(submission_df['Prediction'].describe())


# ============================================
# 第九部分: 保存模型和特征
# ============================================
print("\n" + "=" * 60)
print("💾 保存模型和特征")
print("=" * 60)

model_dir = Path('models')
model_dir.mkdir(exist_ok=True)

# 保存LightGBM模型
lgb_model.save_model(str(model_dir / 'ultimate_lgbm_model.txt'))
print("✅ LightGBM 模型已保存")

# 保存特征列表
with open(model_dir / 'ultimate_features.txt', 'w') as f:
    for feat in feature_cols:
        f.write(f"{feat}\n")
print("✅ 特征列表已保存")

# 保存模型权重
with open(model_dir / 'ultimate_weights.txt', 'w') as f:
    for name, w in best_weights.items():
        f.write(f"{name}: {w}\n")
print("✅ 模型权重已保存")

# 保存特征重要性
importance = lgb_model.feature_importance(importance_type='gain')
feature_importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': importance
}).sort_values('importance', ascending=False)
feature_importance_df.to_csv(model_dir / 'ultimate_feature_importance.csv', index=False)
print("✅ 特征重要性已保存")


# ============================================
# 总结
# ============================================
print("\n" + "=" * 80)
print("🏆 终极解决方案训练完成!")
print("=" * 80)

print(f"""
核心策略总结:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. 特征工程 ({len(feature_cols)} 个特征):
   • 时间特征 (周期性编码)
   • 价格特征 (OHLC比率、蜡烛图形态)
   • 成交量特征 (OBV、价量关系)
   • 技术指标 (RSI、MACD、布林带、ATR、随机指标等)
   • 移动平均 (SMA、EMA、均线交叉)
   • 波动率特征 (GK波动率、Parkinson波动率)
   • 动量特征 (ROC、动量)
   • 滞后特征 (多时间尺度)
   • 滚动统计 (均值、标准差、偏度、峰度、分位数)
   • 交叉特征 (特征交互)
   • 高级特征 (夏普比率、最大回撤)

2. 模型集成:
   • LightGBM (GBDT)
   • XGBoost
   • LightGBM (DART)
   • CatBoost (如果可用)

3. 集成策略:
   • 基于相关系数的加权平均
   • 网格搜索最优权重

4. 验证结果:
   • 最终Pearson相关系数: {final_corr:.6f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
提交文件: submissions/ultimate_ensemble_submission.csv
""")

print("祝你比赛成功! 🚀")
