import pandas as pd
import numpy as np
import gc
import warnings

# Suppress pandas chained assignment warnings for cleaner output
warnings.filterwarnings('ignore')

def reduce_mem_usage(df):
    """Iterates through all columns of a dataframe and modifies the data type to reduce memory usage."""
    start_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage of dataframe is {start_mem:.2f} MB')
    
    for col in df.columns:
        col_type = df[col].dtype
        
        # Check if column is numeric using pandas' built-in function
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
                # Skip columns that can't be compared numerically
                pass
        elif col_type == object:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    print(f'Decreased by {100 * (start_mem - end_mem) / start_mem:.1f}%')
    
    return df


def load_and_merge(raw_path, store_filter=None):
    """
    Loads raw data, optionally filters by store to save memory,
    melts the sales data, and merges with calendar and prices.
    """
    print("Loading raw files...")
    calendar = pd.read_csv(f'{raw_path}/calendar.csv')
    prices = pd.read_csv(f'{raw_path}/sell_prices.csv')
    sales = pd.read_csv(f'{raw_path}/sales_train_evaluation.csv')

    if store_filter:
        print(f"Filtering data for store: {store_filter}...")
        sales = sales[sales['store_id'] == store_filter].copy()
        prices = prices[prices['store_id'] == store_filter].copy()

    #calendar = reduce_mem_usage(calendar)
    #prices = reduce_mem_usage(prices)
    #sales = reduce_mem_usage(sales)

    print("Melting sales data (Wide to Long)...")
    id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    df = pd.melt(sales, id_vars=id_vars, var_name='d', value_name='sales')
    
    del sales
    gc.collect()

    print("Merging with calendar and prices...")
    df = pd.merge(df, calendar, on='d', how='left')
    df = pd.merge(df, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
    
    # Drop rows before product launch
    df.dropna(subset=['sell_price'], inplace=True)
    df['date'] = pd.to_datetime(df['date'])
    
    #return reduce_mem_usage(df)
    return df


def isolate_target_and_nullify(df, split_date):
    """
    Saves the ground truth for the test window and nullifies the target
    variable in the main dataframe to prevent data leakage.
    """
    print(f"Isolating ground truth and nullifying target after {split_date}...")
    
    df_ground_truth = df[df['date'] > split_date][['id', 'date', 'sales']].copy()
    df_ground_truth.rename(columns={'sales': 'actual_sales'}, inplace=True)
    
    df.loc[df['date'] > split_date, 'sales'] = np.nan
    
    return df, df_ground_truth


def create_features(df):
    """
    Generates calendar features, lags, and rolling means anchored to lag 28.
    """
    print("Creating calendar features...")
    df['day_of_month'] = df['date'].dt.day.astype(np.int8)
    df['day_of_week'] = df['date'].dt.dayofweek.astype(np.int8)
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(np.int8)

    event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
    for col in event_cols:
        df[col] = df[col].astype('category')
        df[col] = df[col].cat.codes 
        df[col] = df[col].astype(np.int16)

    print("Sorting data chronologically for temporal features...")
    df.sort_values(by=['id', 'date'], inplace=True)

    print("Calculating Lags...")
    lags = [7, 28]
    for lag in lags:
        df[f'sales_lag_{lag}'] = df.groupby(['id'])['sales'].shift(lag).astype(np.float16)

    print("Calculating Rolling Means (Anchored to Lag 28)...")
    windows = [7, 28]
    for window in windows:
        df[f'rolling_mean_{window}'] = df.groupby(['id'])['sales_lag_28'].transform(
            lambda x: x.rolling(window).mean()
        ).astype(np.float16)

    return df


def build_pipeline(raw_path, split_date='2016-04-24', store_filter=None):
    """
    Orchestrates the entire data preparation pipeline.
    """
    print(f"--- Starting Data Prep Pipeline ---")
    df = load_and_merge(raw_path, store_filter)
    df, df_ground_truth = isolate_target_and_nullify(df, split_date)
    df = create_features(df)
    print(f"--- Pipeline Completed! ---")
    
    return df, df_ground_truth

# Allow execution as a standalone script if needed
if __name__ == "__main__":
    PATH = "../data/raw"
    df, truth = build_pipeline(PATH, store_filter='CA_1')
    print("Processed dataframe shape:", df.shape)