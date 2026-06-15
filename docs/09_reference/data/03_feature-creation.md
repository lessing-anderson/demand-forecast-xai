# `src.data.feature_creation`

## Purpose

Consolidates processed dimensional and fact tables into a unified modeling dataset and computes calendar, event, SNAP, lag, and rolling statistics.

## Public Functions

### `consolidate_processed(tables_dict, store_filter=None)`

Joins processed dimensional and fact tables:
- Base: `fact_sales`.
- Merges:
  1. LEFT JOIN `dim_calendar` on `calendar_id`;
  2. LEFT JOIN `dim_prices` on `(store_id, item_id, wm_yr_wk)`;
  3. LEFT JOIN `dim_location` on `store_id`;
  4. LEFT JOIN `bridge_snap` on `(calendar_id, state_id)`.
- Generates composite series identifier: `id = item_id + '_' + store_id`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `tables_dict` | `dict[str, pd.DataFrame]` | — | Dictionary containing processed tables from `load_data_processed()`. |
| `store_filter` | `str` or `None` | `None` | Optional store filter (e.g. `'CA_1'`). |

---

### `create_features_from_processed(df, lags=[7, 28], rolling_windows=[7, 28])`

Executes the complete feature-engineering sequence:
1. Validates and parses `date` (`datetime64`) and numeric `sales` (`int16`).
2. Sorts data chronologically by `['id', 'date']`.
3. Creates calendar attributes (`day_of_month`, `week_of_year`, `day_of_week`).
4. Creates lag features (`sales_lag_7`, `sales_lag_28`).
5. Creates rolling means anchored to `sales_lag_28` (`rolling_mean_7_sales_lag_28`, `rolling_mean_28_sales_lag_28`).
6. Creates recent causal rolling means on shifted sales (`rolling_mean_7`, `rolling_mean_28`).
7. Standardizes categorical and numeric dtypes.
8. Filters and returns the 25 target columns.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Consolidated DataFrame from `consolidate_processed()`. |
| `lags` | `list[int]` | `[7, 28]` | List of day shift lag intervals. |
| `rolling_windows` | `list[int]` | `[7, 28]` | List of rolling window day spans. |

---

### `save_features(df, out_dir='data/features', filename='features.parquet', overwrite=True)`

Saves the engineered DataFrame to a Parquet file under `out_dir`.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Feature DataFrame to write. |
| `out_dir` | `str` or `Path` | `'data/features'` | Target directory. |
| `filename` | `str` | `'features.parquet'` | Output filename. |
| `overwrite` | `bool` | `True` | Overwrite existing file if true. |

## Internal Helpers

- `_create_calendar_features(df)`: Computes `day_of_month` and ISO `week_of_year`.
- `_create_lag_features(df, lags)`: Computes sales lags per `id`.
- `_create_lag_rolling_features(df, windows, lag_col)`: Computes rolling means over `sales_lag_28`.
- `_create_recent_rolling_features(df, windows)`: Computes causal rolling means over `sales.shift(1)`.
- `_standardize_dtypes(df, lags, rolling_windows, lag_col_for_rolling)`: Downcasts numeric and categorizes string columns.

## Related Documentation

- [Feature Dataset Contract](../../02_data/03_feature-contract.md)
- [Data Lineage](../../02_data/04_lineage.md)
- [Notebook 03: Feature Engineering](../../04_notebooks/03_feature-engineering.md)
