# Evaluation Metrics

## Overview

The forecasting performance of the baseline LightGBM model is evaluated using standard regression error metrics (RMSE, MAE, MAPE) as well as M5 competition metrics specifically designed for zero-inflated and intermittent demand (**RMSSE** and **WRMSSE**). The evaluation module `src/utils/metrics.py` provides functions to compute these metrics.

## Metric Definitions

### 1. Root Mean Squared Error (RMSE)

RMSE measures the square root of the average squared difference between actual sales $y_i$ and predicted sales $\hat{y}_i$:

$$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2}$$

- **Characteristics**: Heavily penalizes large errors/outliers due to squaring.
- **Role in Baseline**: Primary loss function (`metric='rmse'`) for LightGBM training and early stopping.

### 2. Mean Absolute Error (MAE)

MAE computes the average absolute difference between actual sales $y_i$ and predicted sales $\hat{y}_i$:

$$\text{MAE} = \frac{1}{N} \sum_{i=1}^{N} |y_i - \hat{y}_i|$$

- **Characteristics**: Provides a linear penalty for errors and is expressed in natural units (unit sales).
- **Role in Baseline**: Primary metric for ranking holdout prediction errors to select the 200 LIME and SHAP analysis instances.

### 3. Mean Absolute Percentage Error (MAPE)

MAPE measures the average percentage error relative to actual values:

$$\text{MAPE} = \frac{1}{N} \sum_{i=1}^{N} \left| \frac{y_i - \hat{y}_i}{y_i} \right| \times 100\%$$

- **Characteristics**: Scale-independent percentage metric.
- **Limitation in M5 Dataset**: In retail demand forecasting, daily sales $y_i = 0$ occur frequently (intermittent demand). Division by zero produces undefined or extremely inflated values ($> 10^{15}$).

### 4. Root Mean Squared Scaled Error (RMSSE)

RMSSE scales the forecast root mean squared error by the in-sample naive forecast error calculated on the historical training set:

$$\text{RMSSE}_i = \sqrt{ \frac{\frac{1}{H} \sum_{t=1}^{H} (y_{i, t} - \hat{y}_{i, t})^2}{ \frac{1}{N - t_0} \sum_{t=t_0 + 1}^{N} (y_{i, t} - y_{i, t-1})^2 } }$$

- **Characteristics**: Scale-independent, robust to zero sales days because the denominator uses historical squared differences rather than division by actual daily sales $y_i$.
- **M5 Standard**: Scaling starts at index $t_0$, the first day with non-zero sales in the training set for series $i$.

### 5. Weighted Root Mean Squared Scaled Error (WRMSSE)

WRMSSE is the primary metric of the Kaggle M5 competition. It computes a weighted average of individual series RMSSE values across the product/series hierarchy:

$$\text{WRMSSE} = \sum_{i=1}^{K} w_i \cdot \text{RMSSE}_i$$

- **Characteristics**: Series weights $w_i \ge 0$ (normalized such that $\sum w_i = 1$) reflect the commercial importance or sales volume of each time series.
- **Role in M5**: Evaluates model performance across all hierarchical aggregation levels without being skewed by zero-demand periods.

## Baseline Holdout Performance (`exp_001_baseline_lgbm`)

Evaluation on the 28-day holdout set (85,372 store-item-day combinations for store `CA_1` post `2016-04-24`) produced:

| Metric | Holdout Value | Note |
|---|---:|---|
| **RMSE** | **2.1694** | Unscaled quadratic error |
| **MAE** | **1.1507** | Mean absolute sales error |
| **RMSSE** | **0.3784** | Scaled against naive in-sample baseline |
| **WRMSSE** | **0.6931** | Volume/dollar-weighted series scaled error |
| **MAPE** | Inflated ($> 10^{15}$) | Distorted by zero-sales days ($y_i = 0$) |

## Code Reference

The evaluation helper functions are defined in `src/utils/metrics.py`:

```python
from src.utils.metrics import rmsse, wrmsse, evaluate_forecast

# Single series RMSSE
score_rmsse = rmsse(y_true, y_pred, y_train)

# Multi-series WRMSSE
score_wrmsse = wrmsse(y_true, y_pred, y_train, series_ids=series_ids_holdout, series_ids_train=series_ids_train)

# Comprehensive evaluation dictionary
metrics = evaluate_forecast(
    y_true, y_pred,
    dataset_name='Holdout',
    y_train=y_train,
    series_ids=series_ids_holdout,
    series_ids_train=series_ids_train
)
```

## Related Documentation

- [LightGBM Baseline Model](01_lightgbm-baseline.md)
- [Temporal Validation Strategy](02_temporal-validation.md)
- [Module Reference: `src.utils.metrics`](../09_reference/utils/01_metrics.md)

