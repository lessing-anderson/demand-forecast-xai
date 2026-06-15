# Decisions, Limitations, and Evolution

## Implemented Decisions

### Layered Data

Raw data is preserved as CSV, intermediate transformations are materialized as Parquet, and the modeling dataset is also persisted as Parquet. This reduces later read costs and makes data lineage explicit.

### Temporal Evaluation

The `2016-04-24` cutoff separates training from holdout data. Within training, the final 28 days are used as the validation window for LightGBM early stopping.

### Error-Oriented Explanations

LIME and SHAP do not explain a random sample. They select 200 observations, balancing excellent and poor absolute-error cases. This concentrates analysis on the behavior most relevant to the explainability study.

### Multi-Step Recursive Forecasting

To prevent lookahead data leakage during holdout evaluation across the 28-day forecast horizon, the baseline model dynamically iterates day-by-day in `04_lightgbm_baseline.ipynb`. It recalculates `sales_lag_7` and recent rolling means using previously predicted sales $\hat{y}$ rather than ground-truth sales.

### Artifact Persistence

Models, predictions, metrics, and explanations are persisted per experiment in `experiments/<name>/artifacts/`, allowing LIME and SHAP to run without retraining the baseline.

## Known Limitations

### MAPE with Zero Sales

Recorded MAPE is unstable because zero sales are common in retail demand series ($y_i = 0$). To address this limitation, the project uses RMSE and WRMSSE as stable, scaled evaluation benchmarks.
