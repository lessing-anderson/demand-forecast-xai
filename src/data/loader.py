"""
Data loading module for M5 demand forecasting dataset.
Handles raw CSV loading and basic merging operations.
"""
import pandas as pd
import numpy as np
import gc
import warnings

warnings.filterwarnings('ignore')


def load_calendar(raw_path):
    """Load calendar data from raw CSV."""
    return pd.read_csv(f'{raw_path}/calendar.csv')


def load_prices(raw_path):
    """Load sell prices data from raw CSV."""
    return pd.read_csv(f'{raw_path}/sell_prices.csv')


def load_sales(raw_path):
    """Load sales training data from raw CSV."""
    return pd.read_csv(f'{raw_path}/sales_train_evaluation.csv')


def filter_by_store(sales, prices, store_id):
    """Filter sales and prices data for a specific store."""
    sales_filtered = sales[sales['store_id'] == store_id].copy()
    prices_filtered = prices[prices['store_id'] == store_id].copy()
    return sales_filtered, prices_filtered


def melt_sales(sales):
    """Convert sales data from wide to long format."""
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = pd.melt(sales, id_vars=id_vars, var_name='d', value_name='sales')
    return df


def merge_with_calendar_and_prices(df, calendar, prices):
    """Merge sales data with calendar and price information."""
    df = pd.merge(df, calendar, on='d', how='left')
    df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    # Drop rows before product launch (missing prices)
    df.dropna(subset=['sell_price'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    return df


def load_and_merge(raw_path, store_filter=None):
    """
    Load raw M5 data and merge calendar + prices.
    
    Args:
        raw_path: Path to raw data directory
        store_filter: Optional store_id to filter data
        
    Returns:
        Merged dataframe with calendar and price info
    """
    print("Loading raw files...")
    calendar = load_calendar(raw_path)
    prices = load_prices(raw_path)
    sales = load_sales(raw_path)

    if store_filter:
        print(f"Filtering data for store: {store_filter}...")
        sales, prices = filter_by_store(sales, prices, store_filter)

    print("Melting sales data (Wide to Long)...")
    df = melt_sales(sales)
    
    del sales
    gc.collect()

    print("Merging with calendar and prices...")
    df = merge_with_calendar_and_prices(df, calendar, prices)
    
    return df
