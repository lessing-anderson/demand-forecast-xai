# `src.models.lightgbm_model`

## Purpose

Implements the `BaseModel` contract with LightGBM regression.

## `LightGBMModel`

`LightGBMModel(name='lightgbm_forecast', **lgb_params)` uses default regression parameters when no parameters are supplied. `train()` accepts optional validation data, boosting rounds, and early stopping. It records feature names from the training matrix and returns the fitted wrapper.

`predict(X)` raises an error before fitting. `get_feature_importance()` returns a DataFrame with `feature` and `importance`, ordered descending and optionally limited by `top_k`. Importance type can be `gain` or `split`.

## Output

`get_model()` returns the underlying LightGBM booster, which is serialized by the baseline notebook with `save_pickle()`.
