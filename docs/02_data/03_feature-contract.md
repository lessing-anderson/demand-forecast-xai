# Feature Dataset Contract

## Artifact

The feature-engineering stage writes a single Parquet file:

```text
data/features/features.parquet
```

It is produced by `03_feature_engineering.ipynb` through `save_features()`.

## Grain, Volume, and Ordering

- **Grain**: One row per store-item-day.
- **Volume**: 59,181,090 rows (full 10 stores across 1,941 days) $\times$ 25 columns. When filtered for a single store (e.g., `store_id == 'CA_1'`), the dataset contains 5,918,109 rows.
- **Business Key**: `(store_id, item_id, date)`.
- **Derived Identifier**: `id = item_id + '_' + store_id`.

Before temporal features are generated, data is sorted chronologically by `['id', 'date']`.

## Full Column Schema (25 Columns)

| Group | Column | Data Type | Description |
|---|---|---|---|
| **Control / Metadata** | `id` | `string` | Composite series identifier (`item_id_store_id`). Dropped before modeling. |
| | `date` | `datetime64[us]` | Observation date. Used for temporal splitting and sorting. |
| **Product Hierarchy** | `cat_id` | `category` | Category identifier (`FOODS`, `HOBBIES`, `HOUSEHOLD`). |
| | `dept_id` | `category` | Department identifier (e.g., `FOODS_1`, `HOBBIES_1`). |
| | `item_id` | `category` | Unique item identifier (3,049 items). |
| | `state_id` | `category` | State code (`CA`, `TX`, `WI`). |
| | `store_id` | `category` | Store identifier (10 stores, e.g., `CA_1`). |
| **Calendar & Time** | `year` | `int16` | Year of observation (2011 to 2016). |
| | `month` | `int8` | Month of observation (1 to 12). |
| | `day_of_month` | `int8` | Day of the month (1 to 31). |
| | `week_of_year` | `int8` | ISO calendar week of the year (1 to 53). |
| | `day_of_week` | `int8` | Day of week integer code (`wday` from source, 1 to 7). |
| **Target Variable** | `sales` | `int16` | Observed unit sales. Supervised training target. |
| **Events & Exogenous** | `event_name_1` | `category` | Primary calendar event name (e.g., `SuperBowl`, `Easter`). |
| | `event_type_1` | `category` | Primary event type (`Sporting`, `Cultural`, `National`, `Religious`). |
| | `event_name_2` | `category` | Secondary calendar event name if overlapping. |
| | `event_type_2` | `category` | Secondary event type. |
| | `is_snap` | `int8` | Binary indicator (0/1) for SNAP food stamp benefits in the store's state. |
| | `sell_price` | `float32` | Weekly unit selling price for the store-item pair. |
| **Lag Features** | `sales_lag_7` | `float32` | Unit sales lagged by 7 days grouped by `id`. |
| | `sales_lag_28` | `float32` | Unit sales lagged by 28 days grouped by `id`. |
| **Rolling Means on Lag** | `rolling_mean_7_sales_lag_28` | `float32` | 7-day rolling mean computed over `sales_lag_28` grouped by `id`. |
| | `rolling_mean_28_sales_lag_28` | `float32` | 28-day rolling mean computed over `sales_lag_28` grouped by `id`. |
| **Recent Rolling Means** | `rolling_mean_7` | `float32` | Causal 7-day rolling mean on shifted sales (`sales.shift(1).rolling(7).mean()`). |
| | `rolling_mean_28` | `float32` | Causal 28-day rolling mean on shifted sales (`sales.shift(1).rolling(28).mean()`). |

## Baseline Model Feature Interface (22 Features)

The baseline LightGBM model drops metadata and target columns (`id`, `date`, `sales`) via `split_features_target()`. The resulting feature matrix $X$ contains **22 features**:

```text
1.  cat_id                        12. event_type_1
2.  dept_id                       13. event_name_2
3.  item_id                       14. event_type_2
4.  state_id                      15. is_snap
5.  store_id                      16. sell_price
6.  year                          17. sales_lag_7
7.  month                         18. sales_lag_28
8.  day_of_month                  19. rolling_mean_7_sales_lag_28
9.  week_of_year                  20. rolling_mean_28_sales_lag_28
10. day_of_week                   21. rolling_mean_7
11. event_name_1                  22. rolling_mean_28
```

## Encoding and Nullability Rules

- **Categoricals**: String and object columns are converted to `category` dtype, encoded as integer codes for LightGBM, LIME, and SHAP.
- **Null Values**: Historical lags and rolling windows produce null values at the start of each time series. For training, rows with valid targets are kept; for LIME and SHAP sampling, feature nulls are imputed with zero or medians.
- **Recursive Forecasting**: In holdout evaluation, `rolling_mean_7`, `rolling_mean_28`, and `sales_lag_7` beyond day 7 are calculated dynamically using recursive step-by-step predictions to prevent lookahead leakage.