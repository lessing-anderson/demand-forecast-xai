# `src.data.loader`

## Purpose

Provides thin loaders for raw M5 CSV files and processed Parquet tables.

## Public Functions

| Function | Input | Output |
|---|---|---|
| `load_calendar(raw_path)` | Raw-data directory | Calendar DataFrame. |
| `load_prices(raw_path)` | Raw-data directory | Prices DataFrame. |
| `load_sales(raw_path)` | Raw-data directory | Evaluation-sales DataFrame. |
| `load_data_raw(raw_path)` | Raw-data directory | Tuple: calendar, prices, sales. |
| `load_fact_sales(processed_path)` | Processed-data directory | Fact-sales DataFrame. |
| `load_dim_calendar(processed_path)` | Processed-data directory | Calendar dimension DataFrame. |
| `load_dim_prices(processed_path)` | Processed-data directory | Price dimension DataFrame. |
| `load_dim_location(processed_path)` | Processed-data directory | Location dimension DataFrame. |
| `load_bridge_snap(processed_path)` | Processed-data directory | SNAP bridge DataFrame. |
| `load_data_processed(processed_path)` | Processed-data directory | Dictionary of all processed tables. |
| `load_features(feature_path, store_filter=None)` | Features directory and optional store filter | Filtered features DataFrame. |

## Notes

Paths are assembled from the directory argument and fixed filenames. Missing files and schema errors propagate from pandas.
