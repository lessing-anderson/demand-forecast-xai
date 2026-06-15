# `src.data.data_processing`

## Purpose

Transforms raw M5 DataFrames into the Parquet tables stored in `data/processed/`.

## Public Functions

| Function | Responsibility |
|---|---|
| `reduce_mem_usage(df)` | Downcasts numeric columns and converts object columns to categories. |
| `process_dim_calendar(df, out_dir)` | Creates and writes `dim_calendar.parquet`. |
| `process_dim_location(df, out_dir)` | Creates and writes `dim_location.parquet`. |
| `process_dim_prices(df, out_dir)` | Creates and writes `dim_prices.parquet`. |
| `process_bridge_snap(df, out_dir)` | Melts SNAP columns and writes `bridge_snap.parquet`. |
| `process_fact_sales(df, out_dir)` | Melts daily sales and writes `fact_sales.parquet`. |

## Side Effects

All `process_*` functions write Parquet files to `out_dir` and return the created DataFrame. `reduce_mem_usage()` mutates the supplied DataFrame.
