import numpy as np
import pandas as pd
from src.config import PRICE_LAGS, WEATHER_TEMP_LAGS, WEATHER_PRECIP_LAGS

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Constructs past-only features at time t:
    1. Base features: price, temperature, precipitation
    2. Price lags: 1, 2, 3, 4, 8, 12, 24, 52
    3. Seasonal & Calendar features: week_of_year, month, quarter, sin_week, cos_week
    4. Technical indicators: SMA_7, SMA_21, EMA_12, MACD, Momentum_4, Bollinger Bands
    5. Weather features: temperature lags & rolling stats, precipitation lags & rolling stats
    
    All calculations are strictly backward-looking to prevent look-ahead data leakage.
    """
    feat_df = df.copy()
    
    # Calendar & Seasonal harmonics
    dt = feat_df['date'].dt
    feat_df['week_of_year'] = dt.isocalendar().week.astype(int)
    feat_df['month'] = dt.month
    feat_df['quarter'] = dt.quarter
    feat_df['sin_week'] = np.sin(2 * np.pi * feat_df['week_of_year'] / 52.0)
    feat_df['cos_week'] = np.cos(2 * np.pi * feat_df['week_of_year'] / 52.0)
    
    # Price Lags
    for lag in PRICE_LAGS:
        feat_df[f'price_lag_{lag}'] = feat_df['price'].shift(lag)
        
    # Technical Indicators (computed on backward rolling windows)
    feat_df['SMA_7'] = feat_df['price'].rolling(window=7, min_periods=1).mean()
    feat_df['SMA_21'] = feat_df['price'].rolling(window=21, min_periods=1).mean()
    feat_df['EMA_12'] = feat_df['price'].ewm(span=12, adjust=False).mean()
    
    # MACD (EMA_12 - EMA_26)
    ema_26 = feat_df['price'].ewm(span=26, adjust=False).mean()
    feat_df['MACD'] = feat_df['EMA_12'] - ema_26
    feat_df['MACD_signal'] = feat_df['MACD'].ewm(span=9, adjust=False).mean()
    
    # Momentum (4-week price change)
    feat_df['Momentum_4'] = feat_df['price'] - feat_df['price'].shift(4)
    
    # Bollinger Bands (20-week window, 2 std dev)
    roll_20_mean = feat_df['price'].rolling(window=20, min_periods=1).mean()
    roll_20_std = feat_df['price'].rolling(window=20, min_periods=1).std().fillna(0)
    feat_df['Bollinger_middle'] = roll_20_mean
    feat_df['Bollinger_upper'] = roll_20_mean + 2.0 * roll_20_std
    feat_df['Bollinger_lower'] = roll_20_mean - 2.0 * roll_20_std
    
    # Weather - Temperature Lags & Rolling
    for lag in WEATHER_TEMP_LAGS:
        feat_df[f'temp_lag_{lag}'] = feat_df['temperature'].shift(lag)
    feat_df['temp_rolling_mean_4'] = feat_df['temperature'].rolling(window=4, min_periods=1).mean()
    feat_df['temp_rolling_std_4'] = feat_df['temperature'].rolling(window=4, min_periods=1).std().fillna(0)
    
    # Weather - Precipitation Lags & Rolling
    for lag in WEATHER_PRECIP_LAGS:
        feat_df[f'precip_lag_{lag}'] = feat_df['precipitation'].shift(lag)
    feat_df['precip_rolling_mean_4'] = feat_df['precipitation'].rolling(window=4, min_periods=1).mean()
    feat_df['precip_rolling_sum_4'] = feat_df['precipitation'].rolling(window=4, min_periods=1).sum()
    feat_df['precip_rolling_std_4'] = feat_df['precipitation'].rolling(window=4, min_periods=1).std().fillna(0)
    
    # Backward fill the initial 52 weeks warmup so we don't lose data
    feat_df = feat_df.ffill().fillna(0)
    
    return feat_df

def get_feature_subsets():
    """
    Returns feature column subsets for ablation levels A, B, C, D, E.
    """
    price_base = ['price'] + [f'price_lag_{lag}' for lag in PRICE_LAGS]
    seasonal = ['sin_week', 'cos_week', 'month', 'quarter']
    
    weather = (
        ['temperature', 'precipitation'] +
        [f'temp_lag_{lag}' for lag in WEATHER_TEMP_LAGS] +
        ['temp_rolling_mean_4', 'temp_rolling_std_4'] +
        [f'precip_lag_{lag}' for lag in WEATHER_PRECIP_LAGS] +
        ['precip_rolling_mean_4', 'precip_rolling_sum_4', 'precip_rolling_std_4']
    )
    
    tech_indicators = [
        'SMA_7', 'SMA_21', 'EMA_12', 'MACD', 'MACD_signal', 'Momentum_4',
        'Bollinger_middle', 'Bollinger_upper', 'Bollinger_lower'
    ]
    
    level_A = price_base + seasonal
    level_B = price_base + seasonal + weather
    level_C = price_base + seasonal + tech_indicators
    level_D = price_base + seasonal + weather + tech_indicators
    # Level E adds HMM state features ('hmm_state', 'hmm_prob_...') to Level D
    
    return {
        "A": level_A,
        "B": level_B,
        "C": level_C,
        "D": level_D
    }
