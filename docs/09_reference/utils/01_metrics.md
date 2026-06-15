# `src.utils.metrics`

## Purpose

Defines forecast-evaluation metrics based on scikit-learn and M5 competition benchmarks (RMSSE and WRMSSE for zero-inflated demand).

## Public Functions

| Function | Output |
|---|---|
| `rmse(y_true, y_pred)` | Root mean squared error. |
| `mae(y_true, y_pred)` | Mean absolute error. |
| `mape(y_true, y_pred)` | Mean absolute percentage error. |
| `rmsse(y_true, y_pred, y_train, eps=1e-8)` | Root Mean Squared Scaled Error relative to in-sample historical naive error. |
| `wrmsse(y_true, y_pred, y_train, series_ids, ...)` | Weighted Root Mean Squared Scaled Error across time series (M5 primary metric). |
| `wrmsse_dataframe(df_train, df_test, id_col, target_col, pred_col, weight_col)` | DataFrame helper for calculating WRMSSE across time series. |
| `evaluate_forecast(y_true, y_pred, dataset_name, y_train=None, series_ids=None, ...)` | Dictionary containing RMSE, MAE, MAPE, and optionally RMSSE/WRMSSE; prints summary. |

`evaluate_forecast()` removes paired NaN values before computing metrics. MAPE can be unstable when actual sales contain zeros ($y_i = 0$), which is resolved by using `rmsse` and `wrmsse`.

