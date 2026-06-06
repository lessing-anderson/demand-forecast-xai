"""
Data preprocessing module.
Handles memory optimization and data cleaning.
"""
import pandas as pd
import numpy as np


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

def process_dim_calendar(df, out_dir):
    """Create calendar dimension in parquet."""

    # Select and order required columns
    cols = [
        'd', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year',
        'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2'
    ]
    dim_calendar = df[cols].copy()

    # Rename 'd' to 'calendar_id' for clarity
    dim_calendar.rename(columns={'d': 'calendar_id'}, inplace=True)

    # Save to parquet
    df_path = out_dir / 'dim_calendar.parquet'
    dim_calendar.to_parquet(df_path, index=False)

    return dim_calendar

def process_dim_location(df, out_dir):
    """Create location dimension in parquet."""
    
    # Select and order required columns
    cols = [
        'store_id', 'state_id'
    ]
    df_location = df[cols].copy()

    # Drop duplicates to get unique store-state combinations
    dim_location = df_location.drop_duplicates().reset_index(drop=True)

    # Save to parquet
    df_path = out_dir / 'dim_location.parquet'
    dim_location.to_parquet(df_path, index=False)

    return dim_location

def process_dim_prices(df, out_dir):
    """Create price dimension in parquet."""

    # Select and order required columns
    cols = [
        'store_id', 'item_id', 'wm_yr_wk', 'sell_price'
    ]
    dim_prices = df[cols].copy()

    # Save to parquet
    df_path = out_dir / 'dim_prices.parquet'
    dim_prices.to_parquet(df_path, index=False)

    return dim_prices

def process_bridge_snap(df, out_dir):
    """Create bridge SNAP factless fact in parquet."""
    
    # Select and order required columns
    cols = [
        'd', 'snap_CA', 'snap_TX', 'snap_WI'
    ]
    df_bridge_snap = df[cols].copy()

    # Melt SNAP columns to create bridge table
    bridge_snap = df_bridge_snap.melt(
        id_vars=['d'],
        value_vars=['snap_CA', 'snap_TX', 'snap_WI'],
        var_name='state_id',
        value_name='is_snap'
    )

    # Clean state_id values (remove 'snap_' prefix)
    bridge_snap['state_id'] = bridge_snap['state_id'].str.replace('snap_', '')

    # Rename 'd' to 'calendar_id' for clarity
    bridge_snap.rename(columns={'d': 'calendar_id'}, inplace=True)

    # Save to parquet
    df_path = out_dir / 'bridge_snap.parquet'
    bridge_snap.to_parquet(df_path, index=False)

    return bridge_snap

def process_fact_sales(df, out_dir):
    """Create sales fact in parquet."""

    # Remove not required columns
    df_fact_sales = df.drop(columns=['id', 'state_id'])

    # Melt sales data to long format
    id_vars = ['store_id', 'item_id', 'dept_id', 'cat_id']
    fact_sales = pd.melt(df_fact_sales, id_vars=id_vars, var_name='d', value_name='sales')

    # Rename 'd' to 'calendar_id' for clarity
    fact_sales.rename(columns={'d': 'calendar_id'}, inplace=True)

    # Save to parquet
    df_path = out_dir / 'fact_sales.parquet'
    fact_sales.to_parquet(df_path, index=False)

    return fact_sales