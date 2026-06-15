"""
Data loading module for M5 demand forecasting dataset.
Handles raw CSV loading and processed data loading.
"""
import pandas as pd
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

def load_data_raw(raw_path):
    """
    Load raw M5 data.
    
    Args:
        raw_path: Path to raw data directory

    Returns:
        Calendar, price and sales dataframes
    """
    print("Loading raw files...")
    calendar = load_calendar(raw_path)
    prices = load_prices(raw_path)
    sales = load_sales(raw_path)
    
    return calendar, prices, sales

def load_fact_sales(processed_path):
    """Load fact_sales table."""
    return pd.read_parquet(f'{processed_path}/fact_sales.parquet')

def load_dim_calendar(processed_path):
    """Load calendar dimension table."""
    return pd.read_parquet(f'{processed_path}/dim_calendar.parquet')

def load_dim_prices(processed_path):
    """Load prices dimension table."""
    return pd.read_parquet(f'{processed_path}/dim_prices.parquet')

def load_dim_location(processed_path):
    """Load location dimension table."""
    return pd.read_parquet(f'{processed_path}/dim_location.parquet')

def load_bridge_snap(processed_path):
    """Load bridge SNAP factless fact table."""
    return pd.read_parquet(f'{processed_path}/bridge_snap.parquet')

def load_data_processed(processed_path):
    """
    Load processed data.
    
    Args:
        processed_path: Path to processed data directory
    Returns:
        Dictionary with loaded processed tables
    """

    fact_sales = load_fact_sales(processed_path)
    dim_calendar = load_dim_calendar(processed_path)
    dim_prices = load_dim_prices(processed_path)
    dim_location = load_dim_location(processed_path)
    bridge_snap = load_bridge_snap(processed_path)

    tables = {
        'fact_sales': fact_sales,
        'dim_calendar': dim_calendar,
        'dim_prices': dim_prices,
        'dim_location': dim_location,
        'bridge_snap': bridge_snap
    }

    return tables

def _normalize_store_filter(store_filter):
    """Normalize store_filter to a list of store IDs."""
    if store_filter is None:
        return None
    if isinstance(store_filter, str):
        return [store_filter]
    return list(store_filter)


def load_features(feature_path, store_filter=None):
    """Load features.
        
        Args:
            feature_path: Path to features data directory
    
        Returns:
            features dataframe"""

    df = pd.read_parquet(f'{feature_path}/features.parquet')

    if store_filter:
        stores = _normalize_store_filter(store_filter)
        df = df[df['store_id'].isin(stores)].copy()
        print(f'Applied store filter: {stores}')
        
    return df
