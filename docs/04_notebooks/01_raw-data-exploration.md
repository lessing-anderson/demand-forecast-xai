# Notebook 01: Raw Data Exploration

## Purpose

`notebooks/01_raw_data_exploration.ipynb` is an exploratory, read-only
notebook for the M5 source data. It gives an initial view of the calendar,
prices, sales hierarchy, demand distribution, and store-level demand before
any data is transformed.

It is optional in the execution sequence: it does not create an input required
by later notebooks.

## Prerequisites

Run the notebook with `notebooks/` as the working directory. Its relative path
configuration is:

```python
raw_path = '../data/raw'
```

The following raw files must be available:

- `calendar.csv`;
- `sell_prices.csv`;
- `sales_train_evaluation.csv`.

The notebook uses Python 3, Pandas, Matplotlib, Seaborn, and
`src.data.loader`. It enables IPython autoreload so local changes to imported
modules are reflected without restarting the kernel.

## Inputs

| Variable | Source | Use |
|---|---|---|
| `calendar` | `calendar.csv` | Calendar dates, events, and SNAP attributes. |
| `prices` | `sell_prices.csv` | Store-item weekly selling prices. |
| `sales` | `sales_train_evaluation.csv` | Product hierarchy and daily sales columns. |

Data loading is delegated to `load_calendar()`, `load_prices()`, and
`load_sales()` in `src.data.loader`.

## Execution Steps

### 1. Initialize the environment

The notebook adds the repository root to `sys.path`, imports plotting libraries
and the data loaders, and applies the Seaborn `whitegrid` style.

### 2. Load source files

It loads the three CSV files and prints their row-and-column shapes. No memory
optimization is applied at this stage.

### 3. Inspect the calendar

The calendar section prints the first rows, lists all columns, reports the date
range, and shows the most frequent values in both event-type columns.

### 4. Inspect prices

The prices section prints the first rows, counts unique stores and items, and
reports descriptive statistics for `sell_price`.

### 5. Inspect sales hierarchy

The sales section prints the first rows and columns, then counts stores, items,
categories, departments, daily `d_*` columns, and the implied number of time
series.

### 6. Analyze demand distribution

The notebook derives `sales_cols` by selecting columns whose names start with
`d_`. It calculates mean sales for each day, maps M5 day identifiers to real
dates through the calendar table, and renders:

- a time series of average sales by day;
- a log-scale histogram of average sales per item-store series.

### 7. Compare stores

For each store, it computes the average of daily mean sales and renders a
horizontal bar chart sorted in descending order.

### 8. Print summary statistics

The final cell prints counts of periods, stores, categories, items,
store-item combinations, calendar dates, and unique primary event types.

## Outputs

The notebook produces console output and interactive plots only. It does not
write files, alter raw data, create processed tables, or persist figures.

Variables retained in the notebook kernel include `calendar`, `prices`,
`sales`, `sales_cols`, `daily_sales`, `item_avg_sales`, and `store_stats`.

## Dependencies

```text
01_raw_data_exploration.ipynb
 ├── src.data.loader
 ├── data/raw/calendar.csv
 ├── data/raw/sell_prices.csv
 └── data/raw/sales_train_evaluation.csv
```

The next notebook, `02_data_processing.ipynb`, reads the same raw sources but
does not depend on this notebook's in-memory variables or plots.

## Limitations and Notes

- The notebook loads the complete evaluation sales CSV into memory, which can
  be expensive on constrained machines.
- It provides descriptive exploration only; it does not validate schemas,
  missing values, duplicates, or data quality rules.
- The reported number of time series is calculated as unique stores multiplied
  by unique items. This assumes a complete store-item grid and should not
  replace a direct count of unique series in a generalized dataset.
- The two sales-distribution statistics are arithmetic means. They do not
  distinguish intermittent demand, promotion effects, or seasonal components.
- Figures are not saved, so results must be exported manually if they are
  needed in a report.

## Related Documentation

- [M5 Source Data](../02_data/01_m5-source-data.md)
- [Data Lineage](../02_data/04_lineage.md)
- [Module Reference: `src.data.loader`](../09_reference/data/01_loader.md)
