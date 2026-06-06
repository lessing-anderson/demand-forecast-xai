"""
Data loading module for M5 demand forecasting dataset.
Handles raw CSV loading and basic operations.
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


def filter_by_store(sales, prices, store_id):
    """Filter sales and prices data for a specific store."""
    sales_filtered = sales[sales['store_id'] == store_id].copy()
    prices_filtered = prices[prices['store_id'] == store_id].copy()
    return sales_filtered, prices_filtered

def load_data(raw_path, store_filter=None):
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
    
    return calendar, prices, sales
