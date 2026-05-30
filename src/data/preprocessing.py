"""
Feature engineering and data preprocessing module.
Handles memory optimization, feature creation, and temporal transformations.
"""
import pandas as pd
import numpy as np
import gc


def reduce_mem_usage(df):
    """
    Optimize dataframe memory usage by downcasting numeric types.
    
    Args:
        df: Input dataframe
        
    Returns:
        Memory-optimized dataframe
    """
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        if pd.api.types.is_numeric_dtype(col_type):
            try:
                c_min = df[col].min()
                c_max = df[col].max()
                
                if pd.api.types.is_integer_dtype(col_type):
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                elif pd.api.types.is_float_dtype(col_type):
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
            except (TypeError, OverflowError):
                pass
        elif col_type == object:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df


def create_calendar_features(df):
    """Create temporal features from date column."""
    print("Creating calendar features...")
    df['day_of_month'] = df['date'].dt.day.astype(np.int8)
    df['day_of_week'] = df['date'].dt.dayofweek.astype(np.int8)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(np.int8)
    
    return df


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
