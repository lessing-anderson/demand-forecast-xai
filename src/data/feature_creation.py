"""
Feature engineering module.
Handles feature creation and temporal transformations.
"""
import pandas as pd
import numpy as np

def encode_event_features(df):
    """Encode event categorical features as integers."""
    print("Encoding event features...")
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in event_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')
            df[col] = df[col].cat.codes 
            df[col] = df[col].astype(np.int16)
    
    return df

def create_calendar_features(df):
    """Create temporal features from date column."""
    print("Creating calendar features...")
    df['day_of_month'] = df['date'].dt.day.astype(np.int8)
    df['day_of_week'] = df['date'].dt.dayofweek.astype(np.int8)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(np.int8)
    
    return df

def create_lag_features(df, lags=[7, 28]):
    """Create lagged sales features."""
    print(f"Calculating lags: {lags}...")
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['id'])['sales'].shift(lag).astype(np.float16)
    
    return df

def create_rolling_features(df, windows=[7, 28], lag_col='sales_lag_28'):
    """Create rolling mean features anchored to a lag."""
    print(f"Calculating rolling means (windows {windows}, anchored to {lag_col})...")
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby(['id'])[lag_col].transform(
            lambda x: x.rolling(window).mean()
        ).astype(np.float16)
    
    return df

def create_features(df, lags=[7, 28], rolling_windows=[7, 28]):
    """
    Generate all temporal and event features.
    
    Args:
        df: Input dataframe with date and sales columns
        lags: List of lag values for lagged features
        rolling_windows: List of window sizes for rolling features
        
    Returns:
        Feature-engineered dataframe
    """
    print("Sorting data chronologically for temporal features...")
    df.sort_values(by=['id', 'date'], inplace=True)
    
    df = create_calendar_features(df)
    df = encode_event_features(df)
    df = create_lag_features(df, lags)
    df = create_rolling_features(df, rolling_windows, lag_col='sales_lag_28')
    
    return df
