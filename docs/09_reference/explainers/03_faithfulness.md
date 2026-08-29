# `src.explainers.faithfulness`

## Purpose

Implements the **faithfulness (fidelity)** evaluation protocol using **iterative feature deletion (ablation)** to compare local LIME and SHAP explanations.

For each selected instance in the high-error holdout sample (`sample_bucket == 'bad'`), top-ranked features identified by each explainer are sequentially obfuscated. The baseline LightGBM model is re-evaluated at every step and predictive degradation is measured via RMSE and WRMSSE.

## Configuration Class

### `FaithfulnessConfig`

```python
@dataclass(frozen=True)
class FaithfulnessConfig:
    max_steps: int = 10
    error_bucket: str = "bad"
    actual_col: str = "actual_sales"
    id_col: str = "id"
    target_col: str = "sales"
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `max_steps` | `int` | `10` | Maximum number of sequential feature deletion steps. |
| `error_bucket` | `str` | `'bad'` | Target error tail bucket to evaluate (high-error cases). |
| `actual_col` | `str` | `'actual_sales'` | Ground-truth column name in prediction/sample tables. |
| `id_col` | `str` | `'id'` | Time-series identifier column name. |
| `target_col` | `str` | `'sales'` | Historical target column name in training set. |

## Public Functions

### `select_faithfulness_instances(sample_frame, error_bucket='bad')`

Filters the error-tail analysis sample to retain instances belonging to the specified error bucket, sorted descending by absolute prediction error.

### `get_feature_rankings(explanations, method, importance_col=None)`

Extracts per-instance feature deletion order from long-form explanations (`weight` for LIME, `shap_value` for SHAP), sorted by descending absolute attribution magnitude.

### `compute_obfuscation_values(background_frame, feature_names, categorical_columns=None)`

Computes baseline neutral replacement values for obfuscated features:
- **Continuous features**: Median of training background distribution.
- **Categorical features**: Mode of training background distribution.

### `run_iterative_deletion(model, sample_frame, feature_names, rankings, fill_values, method_name, max_steps=None, actual_col='actual_sales')`

Executes the step-by-step feature removal and re-prediction loop for a specific explanation method:
1. Obfuscates top-$k$ features ($k \in [0, \text{max\_steps}]$).
2. Runs model inference on perturbed feature vector.
3. Records new predictions and absolute errors per instance and step.

### `aggregate_deletion_metrics(deletion_results, train_targets_by_series, actual_col='actual_sales', id_col='id')`

Aggregates instance-level deletion results into global RMSE and WRMSSE degradation curves along with deltas relative to step 0.

### `compute_faithfulness_summary_metrics(metric_curve)`

Calculates scalar summary indicators from deletion curves, including total degradation ($\Delta\text{RMSE}$, $\Delta\text{WRMSSE}$) and Area Under the Degradation Curve (AUC) via trapezoidal integration.

### `evaluate_faithfulness_by_bucket(model, sample_frame, lime_explanations, shap_explanations, feature_names, background_frame, train_frame, categorical_columns=None, config=None)`

High-level end-to-end wrapper running the full comparative faithfulness evaluation pipeline between LIME and SHAP. Returns `(deletion_results, metric_curve, summary_metrics)`.

### `plot_faithfulness_curves(metric_curve, metrics=('rmse', 'wrmsse'), ax=None, title=None)`

Generates publication-ready Matplotlib comparison plots showing performance degradation as a function of the number of removed top features.

### `save_faithfulness_artifacts(deletion_results, metric_curve, summary_metrics, output_dir)`

Persists generated tables (`faithfulness_deletion_results.parquet`, `faithfulness_metric_curve.parquet`) and metrics (`faithfulness_metrics.json`) to the experiment artifact directory.

## Related Documentation

- [Notebook 07: Faithfulness Measuring](../../04_notebooks/07_faithfulness-measuring.md)
- [XAI Evaluation Protocol](../../06_explainability/04_evaluation-protocol.md)
