# ADR 002: Out-of-Time Cutoff Temporal Split Strategy

- **Status**: Accepted
- **Date**: 2026-07-26
- **Deciders**: Project Architecture Team
- **Technical Area**: Modeling & Validation Strategy

## Context

In time-series demand forecasting, standard $K$-fold cross-validation or random train/test splitting causes severe lookahead bias and data leakage, as future observations leak information into past predictions.

To accurately reflect real-world retail forecasting deployment, evaluation must be conducted on an out-of-time holdout window occurring chronologically after all training observations.

## Decision

We decided to enforce a strict **out-of-time temporal cutoff** date at **`2016-04-24`**:

1. **Training & Validation Set (`date <= '2016-04-24'`)**:
   - Total rows: 5,747,365 for store `CA_1`.
   - **Validation Window**: The final 28 days (`2016-03-28` to `2016-04-24`, 85,372 rows) are reserved for early stopping during LightGBM model training.
2. **Holdout Evaluation Set (`date > '2016-04-24'`)**:
   - Total rows: 85,372 (28 days $\times$ 3,049 items for `CA_1`).
   - Used exclusively for model evaluation and xAI explanation sampling.

```text
2011-01-29                        2016-03-27   2016-04-24        2016-05-22
 |-------------------------------------|------------|-----------------|
            Training Subset             Validation       Holdout Set
             (5.66M rows)                (85k rows)       (85k rows)
```

## Consequences

### Positive
- **Zero Lookahead Bias**: Model training relies exclusively on past historical observations.
- **Realistic Horizon**: The 28-day holdout matches the official M5 competition forecasting window.
- **Early Stopping Protection**: Reserving a 28-day validation window prevents overfitting during gradient boosting iterations.
- **Leakage-Free Multi-Step Evaluation**: The 28-day holdout evaluation is executed via recursive dynamic forecasting in `04_lightgbm_baseline.ipynb`, masking future test features and recalculating lags and rolling means over predicted values $\hat{y}$ to guarantee zero ground-truth leakage.

### Negative / Trade-Offs
- **Computational Overhead of Recursive Loop**: Iterating day-by-day across 28 steps and dynamically recalculating rolling means increases holdout evaluation runtime compared to single-shot batch inference.

## Related Documentation

- [Temporal Validation Strategy](../05_modeling/02_temporal-validation.md)
- [LightGBM Baseline Model](../05_modeling/01_lightgbm-baseline.md)
- [Decisions & Limitations](../01_architecture/04_decisions.md)
