# `src.models.splitting`

## Purpose

Provides dataset partitioning and train/test separation utilities for temporal forecasting models.

## Public Functions

### `split_train_holdout(df, split_date)`

Splits a DataFrame into historical training/validation data (`date <= split_date`) and out-of-time holdout evaluation data (`date > split_date`).

```python
train_df, holdout_df = split_train_holdout(df, split_date='2016-04-24')
```

| Parameter | Type | Description |
|---|---|---|
| `df` | `pd.DataFrame` | Dataset containing a datetime `date` column. |
| `split_date` | `str` or `pd.Timestamp` | Cutoff date separating historical from future holdout observations. |

**Returns:** `tuple[pd.DataFrame, pd.DataFrame]` representing `(train_df, holdout_df)`.

---

### `split_features_target(df, target_col='sales', drop_cols=None)`

Separates the feature matrix $X$ from the target vector $y$, dropping specified metadata columns.

```python
X, y = split_features_target(
    df,
    target_col='sales',
    drop_cols=['id', 'date', 'd', 'sales']
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `df` | `pd.DataFrame` | — | Input DataFrame. |
| `target_col` | `str` | `'sales'` | Column name containing the prediction target. |
| `drop_cols` | `list[str]` or `None` | `None` | List of non-feature metadata columns to exclude from $X$. |

**Returns:** `tuple[pd.DataFrame, pd.Series]` representing `(X, y)`.

## Related Documentation

- [Temporal Validation Strategy](../../05_modeling/02_temporal-validation.md)
- [LightGBM Baseline Model](../../05_modeling/01_lightgbm-baseline.md)
- [Notebook 04: LightGBM Baseline](../../04_notebooks/04_lightgbm-baseline.md)
