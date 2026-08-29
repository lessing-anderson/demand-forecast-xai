# LightGBM Baseline Model

## Overview

The baseline forecasting model for store `CA_1` is implemented using **LightGBM** (Light Gradient Boosting Machine). It predicts daily unit sales based on product hierarchy, calendar/event attributes, selling price, and historical demand lag and rolling statistics.

The model implementation follows the abstract contract defined by `BaseModel` in `src/models/base_model.py` and is instantiated via `LightGBMModel` in `src/models/lightgbm_model.py`.

## Model Architecture and Interface

`LightGBMModel` inherits from `BaseModel` and encapsulates the `lightgbm.train()` API:

```python
class LightGBMModel(BaseModel):
    def __init__(self, name='lightgbm_forecast', **lgb_params): ...
    def train(self, X_train, y_train, X_val=None, y_val=None, num_rounds=1000, early_stopping=50): ...
    def predict(self, X): ...
    def get_feature_importance(self, importance_type='gain', top_k=20): ...
```

## Hyperparameter Configuration

The baseline experiment (`exp_001_baseline_lgbm`) uses the following hyperparameter settings:

| Parameter | Value | Description |
|---|---|---|
| `objective` | `'regression'` | L2 loss objective for continuous target prediction. |
| `metric` | `'rmse'` | Evaluation metric for validation early stopping. |
| `learning_rate` | `0.05` | Boosting shrinkage rate. |
| `num_leaves` | `31` | Maximum tree leaves per base learner. |
| `feature_fraction` | `0.8` | Subsampling fraction of features per iteration. |
| `bagging_fraction` | `0.8` | Subsampling fraction of rows per iteration. |
| `bagging_freq` | `5` | Frequency for bagging iteration. |
| `num_boost_round` | `500` (max) | Maximum boosting iterations. |
| `early_stopping` | `50` rounds | Early stopping threshold on validation set. |

During training on store `CA_1`, the best iteration was reached at **iteration 483** with a validation RMSE of **2.01288**.

## Input Feature Set (22 Features)

The baseline model drops non-predictive identifiers (`id`, `date`) and the target (`sales`), training on **22 features** in the following categories:

| Feature Category | Feature Names | Description |
|---|---|---|
| **Product Hierarchy** | `cat_id`, `dept_id`, `item_id`, `state_id`, `store_id` | Categorical item and store hierarchical levels. |
| **Calendar & Time** | `year`, `month`, `day_of_month`, `week_of_year`, `day_of_week` | Chronological time and seasonality representations. |
| **Events & Exogenous** | `event_name_1`, `event_type_1`, `event_name_2`, `event_type_2`, `is_snap`, `sell_price` | Cultural/sporting events, SNAP food stamp eligibility, and weekly price. |
| **Lag Features** | `sales_lag_7`, `sales_lag_28` | Historical unit sales shifted by 7 and 28 days per series `id`. |
| **Rolling Means on Lag_28** | `rolling_mean_7_sales_lag_28`, `rolling_mean_28_sales_lag_28` | 7-day and 28-day rolling averages computed over `sales_lag_28`. |
| **Recent Rolling Means** | `rolling_mean_7`, `rolling_mean_28` | Causal 7-day and 28-day rolling averages computed over recent shifted sales (`sales.shift(1)`). |

## Multi-Step Recursive Forecasting

In the 28-day holdout horizon (`2016-04-25` to `2016-05-22`), future observed sales are unavailable. To prevent lookahead leakage:
1. `sales_lag_7` (from day 7 onward) and recent rolling means (`rolling_mean_7`, `rolling_mean_28`) are dynamically recalculated day-by-day.
2. At each date $t$, the model predicts $\hat{y}_t$ for all 3,049 items in store `CA_1`.
3. $\hat{y}_t$ is concatenated into the available sales history to update rolling means and lags for subsequent dates $t+1, \dots, t+28$.

## Persisted Artifacts

Training produces three serialized experiment outputs under `experiments/exp_001_baseline_lgbm/artifacts/`:

1. **`lightgbm_CA_1.pkl`**: Serialized LightGBM booster object (~1.8 MB).
2. **`predictions_CA_1.parquet`**: Holdout predictions (`id`, `date`, `y_true`, `y_pred` across 85,372 rows).
3. **`feature_importance_CA_1.parquet`**: Top 20 features ranked by gain importance (`feature`, `importance`).

## Feature Importance Extraction

`LightGBMModel` extracts two types of feature importance via `get_feature_importance(importance_type, top_k)`:
- **Gain Importance**: Total gain of splits using the feature (measures predictive power).
- **Split Importance**: Number of times a feature is used to split data across trees.

## Related Documentation

- [Execution Flow](../01_architecture/02_execution-flow.md)
- [Temporal Validation Strategy](02_temporal-validation.md)
- [Evaluation Metrics](03_metrics.md)
- [Module Reference: `src.models.lightgbm_model`](../09_reference/models/02_lightgbm-model.md)
