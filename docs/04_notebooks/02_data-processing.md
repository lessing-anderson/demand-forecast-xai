# Notebook 02: Data Processing

## Purpose

`notebooks/02_data_processing.ipynb` converts the raw M5 CSV files into a structured, lightweight dimensional model (star schema) persisted as Parquet tables in `data/processed/`.

It optimizes memory consumption by downcasting numeric types and transforms sales data from a wide format (`d_1`, `d_2`, ...) into a long-form fact table.

## Prerequisites

Run the notebook with `notebooks/` as the working directory. Its relative path configuration is:

```python
raw_path = '../data/raw'
out_dir = Path('../data/processed')
```

The following raw source files must be present in `data/raw/`:

- `calendar.csv`;
- `sell_prices.csv`;
- `sales_train_evaluation.csv`.

The notebook requires Python 3, Pandas, PyArrow (for Parquet support), and the project modules `src.data.loader` and `src.data.data_processing`. It enables IPython autoreload.

## Inputs

| Variable | Source | Use |
|---|---|---|
| `calendar` | `calendar.csv` | Calendar dates, events, and state SNAP attributes. |
| `prices` | `sell_prices.csv` | Store-item weekly selling prices. |
| `sales` | `sales_train_evaluation.csv` | Product hierarchy and daily sales columns (`d_1` to `d_1941`). |

Data loading is executed via `load_data_raw(raw_path)` from `src.data.loader`.

## Execution Steps

### 1. Initialize environment

Sets `sys.path` to include the project root, imports loaders and processing functions, and ensures `data/processed/` directory exists.

### 2. Load raw data

Loads the three source CSVs into memory and prints their shapes:
- Calendar: (1,969, 14)
- Prices: (6,841,121, 4)
- Sales: (30,490, 1,947)

### 3. Create Calendar Dimension (`dim_calendar`)

Applies `reduce_mem_usage()` on a copy of `calendar` (reducing memory by ~34.4%). Calls `process_dim_calendar()` to select calendar attributes, rename `d` to `calendar_id`, and save to `dim_calendar.parquet`.

### 4. Create Location Dimension (`dim_location`)

Applies `reduce_mem_usage()` on `sales` (reducing memory by ~78.5%). Calls `process_dim_location()` to extract unique `store_id` and `state_id` pairs, deduplicate them, and save to `dim_location.parquet`.

### 5. Create Price Dimension (`dim_prices`)

Applies `reduce_mem_usage()` on `prices` (reducing memory by ~24.6%). Calls `process_dim_prices()` to select `store_id`, `item_id`, `wm_yr_wk`, and `sell_price`, then saves to `dim_prices.parquet`.

### 6. Create SNAP Bridge Factless Fact (`bridge_snap`)

Applies `reduce_mem_usage()` on `calendar`. Calls `process_bridge_snap()` to melt state SNAP columns (`snap_CA`, `snap_TX`, `snap_WI`) into `state_id` and `is_snap`, cleans state prefixes, renames `d` to `calendar_id`, and saves to `bridge_snap.parquet`.

### 7. Create Sales Fact (`fact_sales`)

Applies `reduce_mem_usage()` on `sales`. Calls `process_fact_sales()` to drop `id` and `state_id`, melt daily sales columns (`d_1`..`d_1941`) into `calendar_id` and `sales`, and save to `fact_sales.parquet`.

## Outputs

The notebook materializes five Parquet tables in `data/processed/`:

| Output File | Table Name | Rows | Key Columns |
|---|---|---:|---|
| `dim_calendar.parquet` | `dim_calendar` | 1,969 | `calendar_id`, `date`, `wm_yr_wk`, `event_name_1`, `event_type_1`, ... |
| `dim_location.parquet` | `dim_location` | 10 | `store_id`, `state_id` |
| `dim_prices.parquet` | `dim_prices` | 6,841,121 | `store_id`, `item_id`, `wm_yr_wk`, `sell_price` |
| `bridge_snap.parquet` | `bridge_snap` | 5,907 | `calendar_id`, `state_id`, `is_snap` |
| `fact_sales.parquet` | `fact_sales` | 59,181,090 | `store_id`, `item_id`, `dept_id`, `cat_id`, `calendar_id`, `sales` |

## Dependencies

```text
02_data_processing.ipynb
 ├── src.data.loader (load_data_raw)
 ├── src.data.data_processing (reduce_mem_usage, process_dim_*, process_bridge_snap, process_fact_sales)
 ├── data/raw/calendar.csv
 ├── data/raw/sell_prices.csv
 └── data/raw/sales_train_evaluation.csv
```

The next notebook, `03_feature_engineering.ipynb`, consumes the generated `dim_calendar`, `dim_location`, `dim_prices`, and `fact_sales` tables.

## Limitations and Notes

- **Memory Usage during Melt**: Unpivoting 1,941 daily sales columns into long format generates 59,181,090 rows in `fact_sales`, requiring significant RAM during execution.
- **Type Casting**: `reduce_mem_usage()` downcasts integers and float types and converts string columns to categorical.

## Related Documentation

- [Processed Data Schema](../02_data/02_processed-schema.md)
- [Data Lineage](../02_data/04_lineage.md)
- [Module Reference: `src.data.data_processing`](../09_reference/data/02_data-processing.md)
- [Module Reference: `src.data.loader`](../09_reference/data/01_loader.md)
