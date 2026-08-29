# M5 Source Data

## Dataset

This project uses the data from the
[M5 Forecasting Accuracy competition](https://www.kaggle.com/competitions/m5-forecasting-accuracy/data).
The original files belong in `data/raw/` and must be treated as immutable input
data. The project does not download or modify those files.

## Files

| File | Grain | Used by the current pipeline | Purpose |
|---|---|---:|---|
| `calendar.csv` | One row per M5 day | Yes | Maps day identifiers to dates and provides calendar, event, and SNAP attributes. |
| `sell_prices.csv` | Store, item, and week | Yes | Provides weekly item prices by store. |
| `sales_train_evaluation.csv` | Store-item series with one column per day | Yes | Supplies daily sales from `d_1` through `d_1941`. |
| `sales_train_validation.csv` | Store-item series with one column per day | No | Competition reference file with sales through `d_1913`. |
| `sample_submission.csv` | Forecast series and horizon | No | Competition submission template. |

## Current Modeling Window

The implemented baseline uses `sales_train_evaluation.csv`. It treats the
period through `2016-04-24` as historical training data and the later period as
the holdout set. This mirrors the public M5 validation window while retaining
actual values for error analysis and explainability research.

## Source Schemas

### Calendar

`calendar.csv` contains the following groups of variables:

- `date` and `d`: calendar date and M5 day identifier;
- `wm_yr_wk`, `weekday`, `wday`, `month`, and `year`: temporal attributes;
- `event_name_1`, `event_type_1`, `event_name_2`, and `event_type_2`: event
  information;
- `snap_CA`, `snap_TX`, and `snap_WI`: state-specific SNAP indicators.

### Prices

`sell_prices.csv` has the natural key `(store_id, item_id, wm_yr_wk)` and the
measure `sell_price`.

### Sales

`sales_train_evaluation.csv` contains product hierarchy attributes (`item_id`,
`dept_id`, and `cat_id`), location attributes (`store_id` and `state_id`), the
competition series identifier (`id`), and daily sales columns from `d_1` to
`d_1941`.

## Acquisition and Validation

Download the files from Kaggle and place them directly in `data/raw/` using the
exact names listed above. Before running the pipeline, confirm that the three
required files exist and that their headers match the expected schema. The
loader in `src/data/loader.py` reads CSV files without schema migration or
automatic fallbacks.

The raw-data README contains additional context about the M5 validation and
evaluation phases: [`data/raw/README.md`](../../data/raw/README.md).
