# ADR 003: Deterministic Error-Tail Sampling for Local Explanations

- **Status**: Accepted
- **Date**: 2026-07-26
- **Deciders**: Project Architecture Team
- **Technical Area**: Explainable AI (xAI) & Sampling Strategy

## Context

Evaluating local explanation methods (LIME and SHAP) on a random sample of retail demand predictions is ineffective. In large demand datasets, random sampling is heavily skewed toward typical, low-volume sales days with low errors, failing to provide insight into model failure modes or extreme demand regimes.

Furthermore, stochastic or unaligned sampling prevents direct, instance-by-instance comparison between LIME and SHAP attributions.

## Decision

We decided to implement **Deterministic Error-Tail Sampling** (`select_error_tail_sample()` in `src/explainers/lime_explainer.py`):

1. **Error Metric**: Calculate absolute prediction error $\text{abs\_error} = |y_{\text{actual}} - \hat{y}_{\text{pred}}|$ on the holdout evaluation set.
2. **Deterministic Sort**: Sort holdout rows by `abs_error` with a fixed seed (`seed=42`) tie-breaker.
3. **Tail Selection**:
   - Select 100 instances with the smallest absolute error (`sample_bucket == 'excellent'`).
   - Select 100 instances with the largest absolute error (`sample_bucket == 'bad'`).
4. **Shared Sample Alignment**: Both LIME (Notebook 05) and SHAP (Notebook 06) consume the exact same 200 selected instances.

```text
Holdout Predictions (85,372 rows)
  ├── 100 Lowest Error Rows (sample_bucket = 'excellent')  ──┐
  └── 100 Highest Error Rows (sample_bucket = 'bad')       ──┴──> 200 Evaluation Instances (LIME & SHAP)
```

## Consequences

### Positive
- **High-Contrast Analysis**: Allows contrasting explanation dynamics between scenarios where the model is highly accurate vs. scenarios where it fails significantly.
- **Perfect Alignment**: Enables direct per-instance comparisons between LIME linear surrogate weights and SHAP Shapley values.
- **100% Reproducible**: Fixed seed ensures identical instance selection across reruns.

### Negative / Trade-Offs
- **Non-Representative Sample**: The 200-instance sample intentionally overrepresents extreme error cases and does not reflect the overall error distribution of the dataset.

## Related Documentation

- [Error-Tail Sampling Strategy](../06_explainability/03_sampling-by-error.md)
- [LIME Local Explanations](../06_explainability/01_lime.md)
- [SHAP Local Explanations](../06_explainability/02_shap.md)
