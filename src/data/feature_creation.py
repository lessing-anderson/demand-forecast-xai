"""
Feature engineering module.
Handles feature creation and temporal transformations.
"""
from pathlib import Path

import pandas as pd
import numpy as np

def consolidate_processed(tables_dict, store_filter=None):
    """Join processed tables into a single DataFrame ready for feature engineering.

    Expected tables: 'fact_sales' (required), 'dim_calendar', 'dim_prices',
    'dim_location', 'bridge_snap'. 

    The returned DataFrame will contain at least
    ['store_id','item_id','calendar_id','date','sales'] and a generated 'id'
    column (combination of item_id and store_id) used by feature creation.
    """

    fact_sales = tables_dict['fact_sales'].copy()

    # Start with fact_sales as the base
    if store_filter:
        df = fact_sales[fact_sales['store_id'] == store_filter]
    else:
        df = fact_sales

    # Merge calendar to get date and week id
    dim_calendar = tables_dict['dim_calendar'].copy()

    df = df.merge(dim_calendar, on='calendar_id', how='left')

    # Merge prices on (store_id, item_id, wm_yr_wk)
    dim_prices = tables_dict['dim_prices'].copy()
    df = df.merge(dim_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

    # Merge location (state) info if present
    dim_location = tables_dict['dim_location'].copy()
    df = df.merge(dim_location, on='store_id', how='left')

    # Merge bridge SNAP info if present
    bridge_snap = tables_dict['bridge_snap'].copy()
    df = df.merge(bridge_snap, on=['calendar_id', 'state_id'], how='left')

    # Create composite id used by downstream functions
    if 'item_id' in df.columns and 'store_id' in df.columns:
        df['id'] = df['item_id'].astype(str) + '_' + df['store_id'].astype(str)

    return df

def _create_calendar_features(df):
    """Create temporal features from date column."""
    print("Creating calendar features...")
    df['day_of_month'] = df['date'].dt.day.astype(np.int8)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(np.int8)
    
    return df

def _create_lag_features(df, lags=[7, 28]):
    """Create lagged sales features."""
    print(f"Calculating lags: {lags}...")
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['id'])['sales'].shift(lag).astype(np.float16)
    
    return df

def _create_lag_rolling_features(df, windows=(7, 28), lag_col='sales_lag_28'):
    """Create rolling mean features anchored to a lagged sales feature."""

    if lag_col not in df.columns:
        raise ValueError(
            f"Required lag column '{lag_col}' not found. "
            "Create lag features before rolling features."
        )

    print(
        f"Calculating rolling means "
        f"(windows={windows}, anchored to {lag_col})..."
    )

    for window in windows:
        df[f'rolling_mean_{window}_{lag_col}'] = (
            df.groupby('id')[lag_col]
              .transform(lambda x: x.rolling(window).mean())
              .astype(np.float16)
        )

    return df

def _create_recent_rolling_features(df, windows=(7, 28)):
    """Create causal rolling means using only sales prior to each row."""

    print(f"Calculating recent rolling means: {windows}...")

    for window in windows:
        df[f'rolling_mean_{window}'] = (
            df.groupby('id')['sales']
              .transform(
                  lambda x: x.shift(1).rolling(window).mean()
              )
              .astype(np.float16)
        )

    return df

def _standardize_dtypes(df, lags, rolling_windows, lag_col_for_rolling):
    # --------------------------------------------------
    # Standardize categorical columns
    # --------------------------------------------------

    print("Standardize categorical columns...")
    categorical_cols = [
        'store_id',
        'item_id',
        'dept_id',
        'cat_id',
        'state_id',
        'event_name_1',
        'event_type_1',
        'event_name_2',
        'event_type_2',
        'is_snap',
    ]

    for col in categorical_cols:
        if col in df.columns:
            df[col] = df[col].astype('category')

    # --------------------------------------------------
    # Standardize integer columns
    # --------------------------------------------------

    print("Standardize integer columns...")
    int8_cols = [
        'day_of_week',
        'month',
        'day_of_month'
        'week_of_year',
    ]

    for col in int8_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                .astype(np.int8)
            )

    int16_cols = [
        'year'
    ]

    for col in int16_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                .astype(np.int16)
            )

    # --------------------------------------------------
    # Standardize continuous / temporal features
    # --------------------------------------------------

    print("Standardize continuous / temporal features...")
    float32_cols = [
        'sell_price',
        *[f'sales_lag_{lag}' for lag in lags],
        *[f'rolling_mean_{window}' for window in rolling_windows],
        *[
            f'rolling_mean_{window}_{lag_col_for_rolling}'
            for window in rolling_windows
        ],
    ]

    for col in float32_cols:
        if col in df.columns:
            df[col] = (
                pd.to_numeric(df[col], errors='coerce')
                .astype(np.float32)
            )

    return df


def create_features_from_processed(df, lags=[7, 28], rolling_windows=[7, 28]):
    """Prepare processed DataFrame, create features and standardize dtypes."""

    print('Preparing processed DataFrame for feature creation...')
    df = df.copy()

    # ==================================================
    # 1. REQUIREMENTS FOR FEATURE ENGINEERING
    # ==================================================

    # Ensure date column
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'])

    # If 'id' missing, create from item/store combination
    if 'id' not in df.columns and {'item_id', 'store_id'}.issubset(df.columns):
        df['id'] = (df['item_id'].astype(str) + '_' + df['store_id'].astype(str))

    # Ensure target column
    if 'sales' in df.columns:
        df['sales'] = (pd.to_numeric(df['sales'], errors='coerce').astype(np.int16))

    # Fill missing event values with 'NO_EVENT'
    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in event_cols:
        df[col] = df[col].fillna('NO_EVENT')

    # Sort before temporal features
    print("Sorting data chronologically for temporal features...")
    df.sort_values(by=['id', 'date'], inplace=True)

    # ==================================================
    # 2. FEATURE ENGINEERING
    # ==================================================
    lag_col_for_rolling = 'sales_lag_28'

    df = _create_calendar_features(df)
    df = _create_lag_features(df, lags)
    df = _create_lag_rolling_features(df, rolling_windows, lag_col_for_rolling)
    df = _create_recent_rolling_features(df, rolling_windows)

    # ==================================================
    # 3. RENAME COLUMNS
    # ==================================================

    df.rename(
        columns={
            'wday': 'day_of_week',
        },
        inplace=True
    )

    # ==================================================
    # 4. STANDARDIZE THE DTYPES
    # ==================================================

    df = _standardize_dtypes(df, lags, rolling_windows, lag_col_for_rolling)

    # ==================================================
    # 5. SELECT OUTPUT COLUMNS
    # ==================================================

    print("Selecting final feature columns...")

    output_cols = [
        # Control
        'id',
        'date',

        # Hierarchy   
        'cat_id', 
        'dept_id',   
        'item_id',        
        'state_id',
        'store_id',

        # Calendar
        'year',
        'month',
        'day_of_month',
        'week_of_year',
        'day_of_week',

        # Target
        'sales',

        # Events / exogenous
        'event_name_1',
        'event_type_1',
        'event_name_2',
        'event_type_2',
        'is_snap',
        'sell_price',

        # Lag features
        *[f'sales_lag_{lag}' for lag in lags],

        # Rolling means over lag
        *[
            f'rolling_mean_{window}_{lag_col_for_rolling}'
            for window in rolling_windows
        ],

        # Recent rolling means
        *[
            f'rolling_mean_{window}'
            for window in rolling_windows
        ],
    ]

    df = df[output_cols].copy()

    return df


def save_features(df, out_dir='data/features', filename='features.parquet', overwrite=True):
    """Save features DataFrame to a parquet file under `out_dir`.

    Writes a single consolidated file containing `store_id` and `id` so it can
    be filtered later.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fp = out_path / filename

    if fp.exists() and not overwrite:
        print(f"File {fp} exists and overwrite=False. Skipping.")
        return fp

    # Use pandas to_parquet
    df.to_parquet(fp, index=False)
    print(f"Saved features to {fp}")
    return fp
