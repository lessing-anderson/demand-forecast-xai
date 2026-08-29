# Notebook 04: LightGBM Baseline Training

## Purpose

`notebooks/04_lightgbm_baseline.ipynb` trains and evaluates the baseline LightGBM demand forecasting model. It performs temporal train/validation/holdout splitting, serializes the trained model, and saves holdout predictions, evaluation metrics, and feature importances.

## Prerequisites

Run the notebook with `notebooks/` as the working directory. Its relative path configuration is:

```python
feature_path = Path('../data/features/features.parquet')
artifact_dir = Path('../experiments/exp_001_baseline_lgbm/artifacts')
split_date = pd.Timestamp('2016-04-24')
```

The feature dataset `data/features/features.parquet` must exist.

The notebook uses Python 3, Pandas, NumPy, LightGBM, `src.data.loader`, `src.models.splitting`, `src.models.lightgbm_model`, `src.utils.metrics`, and `src.utils.helpers`.

## Inputs

| Variable | Source | Use |
|---|---|---|
| `df` | `data/features/features.parquet` | Engineered feature dataset loaded via `src.data.loader.load_features()`. |
| `split_date` | `Timestamp('2016-04-24')` | Cutoff separating training/validation from holdout evaluation. |

## Execution Steps

### 1. Initialize environment

Configures autoreload, imports modeling and metric functions, and prepares output directories under `experiments/exp_001_baseline_lgbm/artifacts/`.

### 2. Preprocess features for LightGBM

- Converts date column to `datetime64`.
- Encodes non-date object/string columns as categorical integer codes.
- Converts boolean columns to `int8`.
- Sorts data by `date`.

### 3. Perform temporal train/validation/holdout split

- Calls `split_train_holdout(df, split_date='2016-04-24')` to separate historical training (`date <= 2016-04-24`) from holdout (`date > 2016-04-24`).
- Splits historical training into training subset and validation subset (`date <= 2016-03-27` vs `2016-03-28` to `2016-04-24`, 85,372 rows).
- Drops non-feature metadata columns (`id`, `date`, `sales`) via `split_features_target()`.
- Feature matrix: **22 features** across 5 categories (hierarchy, calendar, events/exogenous, lags, rolling means).
- Train rows: 5,747,365 (pre-split) / 5,661,993 (train subset).
- Val rows: 85,372 (28 days).
- Holdout test rows: 85,372 (28 days $\times$ 3,049 items).

### 4. Train LightGBM model

Instantiates `LightGBMModel(name='lightgbm_baseline_exp001')` and trains with parameters:
- `objective`: `'regression'`, `metric`: `'rmse'`
- `learning_rate`: 0.05, `num_leaves`: 31
- `num_rounds`: 500, `early_stopping`: 50
- Training completes with best iteration at **iteration 483** (valid RMSE: 2.01288).

### 5. Multi-step recursive forecasting on holdout

To eliminate lookahead leakage in the 28-day test period:
- Initial lag 7 and recent rolling means (`rolling_mean_7`, `rolling_mean_28`) from the test period are masked with `NaN`.
- Iterates chronologically day-by-day across all 28 holdout dates (`2016-04-25` to `2016-05-22`).
- On each day, dynamically recalculates `sales_lag_7` and causal rolling means (`rolling_mean_7`, `rolling_mean_28`) using available historical sales concatenated with previous days' predictions $\hat{y}$.
- Generates predictions $\hat{y}_{\text{pred}}$ for all 3,049 items and appends them to the historical sales series.

### 6. Persist experiment artifacts

Saves trained model, predictions, and top-20 feature importances to `experiments/exp_001_baseline_lgbm/artifacts/`.

## Outputs

The notebook generates three persisted artifacts:

| Output File | Format | Description |
|---|---|---|
| `lightgbm_CA_1.pkl` | Pickle | Serialized trained LightGBM booster object (~1.8 MB). |
| `predictions_CA_1.parquet` | Parquet | Holdout predictions containing `id`, `date`, `y_true`, `y_pred` (85,372 rows). |
| `feature_importance_CA_1.parquet` | Parquet | Top 20 features ranked by gain importance (20 rows $\times$ 2 columns). |

## Dependencies

```text
04_lightgbm_baseline.ipynb
 ├── src.data.loader (load_features)
 ├── src.models.splitting (split_train_holdout, split_features_target)
 ├── src.models.lightgbm_model (LightGBMModel)
 ├── src.utils.metrics (evaluate_forecast)
 ├── src.utils.helpers (save_pickle)
 └── data/features/features.parquet
```

Notebooks `05_LIME_explainer.ipynb` and `06_SHAP_explainer.ipynb` consume the artifacts produced here.

## Limitations and Notes

- **Unpersisted Categorical Mappings**: Categorical feature codes are computed in-memory and not saved to disk alongside the model artifact.
- **Fixed Store Filter**: Currently trained specifically on store `CA_1`.

## Related Documentation

- [LightGBM Baseline Model](../05_modeling/01_lightgbm-baseline.md)
- [Temporal Validation Strategy](../05_modeling/02_temporal-validation.md)
- [Evaluation Metrics](../05_modeling/03_metrics.md)
- [Module Reference: `src.models.lightgbm_model`](../09_reference/models/02_lightgbm-model.md)
- [Module Reference: `src.models.splitting`](../09_reference/models/03_splitting.md)
- [Module Reference: `src.utils.metrics`](../09_reference/utils/01_metrics.md)
