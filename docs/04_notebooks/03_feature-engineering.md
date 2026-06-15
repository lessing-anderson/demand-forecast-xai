# Notebook 03: Feature Engineering

## Purpose

`notebooks/03_feature_engineering.ipynb` consolidates the processed dimensional tables and generates calendar, event, lag, and rolling mean features. It prepares and persists the modeling dataset `data/features/features.parquet`.

## Prerequisites

Run the notebook with `notebooks/` as the working directory. Its relative path configuration is:

```python
processed_path = '../data/processed'
out_dir = '../data/features'
split_date = '2016-04-24'
```

The following processed tables must be available in `data/processed/`:

- `dim_calendar.parquet`;
- `dim_location.parquet`;
- `dim_prices.parquet`;
- `bridge_snap.parquet`;
- `fact_sales.parquet`.

The notebook uses Python 3, Pandas, NumPy, Matplotlib, Seaborn, `src.data.loader`, and `src.data.feature_creation`.

## Inputs

| Variable | Source | Use |
|---|---|---|
| `tables` | `data/processed/*.parquet` | Processed dimensional and fact tables loaded via `load_data_processed()`. |

## Execution Steps

### 1. Initialize environment

Imports required packages, sets plotting styles (`whitegrid`, size 12x6), and adds the repository root to `sys.path`.

### 2. Load and consolidate data

Loads processed tables via `load_data_processed(processed_path)` and joins them using `consolidate_processed(tables)`:
- Base table: `fact_sales`.
- Joins:
  1. `fact_sales` LEFT JOIN `dim_calendar` on `calendar_id`;
  2. result LEFT JOIN `dim_prices` on `(store_id, item_id, wm_yr_wk)`;
  3. result LEFT JOIN `dim_location` on `store_id`;
  4. result LEFT JOIN `bridge_snap` on `(calendar_id, state_id)`.
- Composite Key: Generates `id = item_id + '_' + store_id`.

### 3. Create temporal and event features

Calls `create_features_from_processed(df, lags=[7, 28], rolling_windows=[7, 28])`:
- Date parsing: Ensures `date` is `datetime64`.
- Sorting: Sorts chronologically by `['id', 'date']`.
- Calendar features: Computes `day_of_month` (1-31), `day_of_week` (`wday` 1-7), and `week_of_year` (1-53).
- Event encoding: Standardizes `event_name_1`, `event_type_1`, `event_name_2`, and `event_type_2` as `category`.
- Exogenous indicators: Standardizes `is_snap` (`int8`) and `sell_price` (`float32`).
- Lag features: Computes `sales_lag_7` and `sales_lag_28` by shifting `sales` grouped by `id`.
- Rolling means over lag_28: Computes `rolling_mean_7_sales_lag_28` and `rolling_mean_28_sales_lag_28` using rolling windows over `sales_lag_28` grouped by `id`.
- Recent causal rolling means: Computes `rolling_mean_7` and `rolling_mean_28` over `sales.shift(1)` grouped by `id`.

### 4. Persist feature dataset

Saves the engineered features to `data/features/features.parquet` via `save_features()`.

### 5. Inspect features

Prints summary statistics, data shape `(59,181,090, 25)`, and column dtypes.

## Outputs

The notebook writes a single Parquet file:

| Output File | Rows | Columns | Description |
|---|---:|---:|---|
| `data/features/features.parquet` | 59,181,090 | 25 | Consolidated feature dataset across all 10 M5 stores. |

### Feature Columns (25 total):
- **Metadata & Control (2)**: `id`, `date`
- **Product Hierarchy (5)**: `cat_id`, `dept_id`, `item_id`, `state_id`, `store_id`
- **Calendar & Time (5)**: `year`, `month`, `day_of_month`, `week_of_year`, `day_of_week`
- **Target (1)**: `sales`
- **Events & Exogenous (6)**: `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`, `is_snap`, `sell_price`
- **Lag Features (2)**: `sales_lag_7`, `sales_lag_28`
- **Rolling Means on Lag_28 (2)**: `rolling_mean_7_sales_lag_28`, `rolling_mean_28_sales_lag_28`
- **Recent Rolling Means (2)**: `rolling_mean_7`, `rolling_mean_28`

## Dependencies

```text
03_feature_engineering.ipynb
 ├── src.data.loader (load_data_processed)
 ├── src.data.feature_creation (consolidate_processed, create_features_from_processed, save_features)
 └── data/processed/*.parquet
```

The next notebook, `04_lightgbm_baseline.ipynb`, consumes `data/features/features.parquet`.

## Limitations and Notes

- **Null Values**: Historical lags and rolling windows produce null values at the start of each time series (`sales_lag_7`: 21,343 nulls per store; `sales_lag_28`: 85,372 nulls per store).

## Related Documentation

- [Feature Dataset Contract](../02_data/03_feature-contract.md)
- [Data Lineage](../02_data/04_lineage.md)
- [Module Reference: `src.data.feature_creation`](../09_reference/data/03_feature-creation.md)
