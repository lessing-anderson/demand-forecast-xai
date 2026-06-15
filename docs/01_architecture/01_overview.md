# Architecture Overview

## Purpose

**demand-forecast-xai** is an applied research project for demand forecasting
with the M5 Forecasting Accuracy dataset. It transforms the competition data
into an analytical dataset, trains a LightGBM model, produces local LIME
and SHAP explanations to contrast predictions with low and high errors, and measure key xAI metrics to compare both methods.

The implemented architecture is driven by sequential notebooks and supported
by reusable Python modules in `src/`. The notebooks orchestrate the process;
the modules encapsulate data operations, modeling, metrics, and explainability.

## Layers

| Layer | Location | Responsibility |
|---|---|---|
| Raw data | `data/raw/` | Original M5 files, which must remain immutable. |
| Processed data | `data/processed/` | Intermediate Parquet tables in a dimensional model. |
| Features | `data/features/` | Consolidated, enriched dataset for modeling. |
| Domain code | `src/` | Loading, transformation, features, splitting, model, and xAI. |
| Orchestration | `notebooks/` | End-to-end experiment execution. |
| Experiments | `experiments/` | Persisted models, predictions, metrics, and explanations. |

## Source Data

The workflow consumes `calendar.csv`, `sell_prices.csv`, and
`sales_train_evaluation.csv`. `sales_train_validation.csv` and
`sample_submission.csv` are competition references and are not used by the
current execution flow.

## Processing and Features

`02_data_processing.ipynb` materializes five tables: `dim_calendar`, `dim_location`, `dim_prices`, `bridge_snap`, and `fact_sales`. The fact table transforms sales from wide format (`d_1`, `d_2`, ...) to long format.

`03_feature_engineering.ipynb` joins sales, calendar, prices, location, and SNAP bridge into `features.parquet` (59,181,090 rows $\times$ 25 columns). Features include product hierarchy, calendar variables, encoded events, SNAP food stamp eligibility (`is_snap`), selling price, lags of 7 and 28 days, rolling means anchored to `sales_lag_28`, and recent causal rolling means.

## Modeling and Explainability

`LightGBMModel` implements the contract defined by `BaseModel`. The baseline uses **22 features**, a 28-day temporal validation window, and a multi-step recursive forecasting holdout period after the `2016-04-24` cutoff.

LIME and SHAP reuse the persisted baseline model and predictions. Both methods deterministically select 200 cases, split between low and high absolute errors (`sample_bucket` `excellent` vs. `bad`), and use up to 2,000 historical rows as background data.

See [execution-flow.md](02_execution-flow.md) for the operational sequence,
[module-dependencies.md](03_module-dependencies.md) for code relationships,
and [decisions.md](04_decisions.md) for technical decisions and limitations.
