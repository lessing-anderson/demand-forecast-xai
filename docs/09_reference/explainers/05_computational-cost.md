# `src.explainers.computational_cost`

## Purpose

Provides high-precision profiling and comparative benchmarking of wall-clock execution time (in seconds) for generating **local** explanations using **LIME Local** and **SHAP Local**.

The analysis focuses strictly on local instance-level explanation latency, isolating SHAP TreeExplainer local values from global TreeSHAP calculations.

## Timing Utilities

### `Timer`

A context manager measuring wall-clock duration in seconds with high precision (`time.perf_counter()`):

```python
with Timer() as t:
    # explanation computation
print(f"Elapsed: {t.interval:.4f} seconds")
```

### `timer(func)`

A function decorator that returns `(function_result, elapsed_time_seconds)`.

## Public Functions

### `explain_local_shap_single(explainer, instance_features)`

Generates local SHAP values for a single instance using a pre-built `TreeExplainer` and records elapsed time in seconds.

### `explain_local_lime_single(explainer, predict_fn, instance_series, num_features=10)`

Generates a local LIME explanation for a single instance using a pre-built `LimeTabularExplainer` and records elapsed time in seconds.

### `benchmark_computational_cost(model, sample_frame, background_frame, feature_names, categorical_columns=None, categorical_names=None, num_lime_features=10, random_state=42)`

Runs iterative benchmark profiling across the 200 error-tail holdout instances for both SHAP Local and LIME Local under identical conditions (explainers pre-built once, simulating a production model-serving deployment). Returns `(benchmark_df, summary_df)`.

### `plot_computational_cost_comparison(benchmark_df, summary_df)`

Generates a dual-panel figure:
1. **Bar Plot**: Mean execution time per instance in seconds with annotated values.
2. **Box Plot**: Dispersion and outlier analysis of execution time across instances.

### `save_computational_cost_artifacts(benchmark_df, summary_df, fig, artifact_dir)`

Persists raw benchmark runs (`computational_cost_results.parquet`), summary statistics (`computational_cost_metrics.json`), and the comparison figure (`fig_computational_cost_comparison.png`).

## Related Documentation

- [Notebook 09: Computational Cost Measuring](../../04_notebooks/09_computational-cost-measuring.md)
- [XAI Evaluation Protocol](../../06_explainability/04_evaluation-protocol.md)
