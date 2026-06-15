# SHAP Local Explanations

## Overview

SHAP (SHapley Additive exPlanations) is a game-theoretic approach to feature attribution that assigns each feature an exact Shapley value representing its contribution to a model prediction.

This project implements SHAP explanations for the baseline LightGBM model using `shap.TreeExplainer`. The implementation is encapsulated in `src/explainers/shap_explainer.py` and executed via `notebooks/06_SHAP_explainer.ipynb`.

## Additive Efficiency Equation

For any instance $x$, SHAP satisfies the additive property:

$$f(x) = \phi_0 + \sum_{i=1}^{M} \phi_i$$

where:
- $f(x)$ is the model output prediction (`model_output`).
- $\phi_0$ is the base value / expected value (`expected_value`), representing the average prediction over the background dataset.
- $\phi_i$ is the SHAP value (`shap_value`) for feature $i$.
- $M$ is the total number of features ($M = 22$).

## Key Functions (`src/explainers/shap_explainer.py`)

- **`build_shap_explainer(model, background_frame)`**: Instantiates `shap.TreeExplainer(model, data=background_matrix)`.
- **`explain_shap_sample(model, sample_frame, background_frame, ...)`**: Computes SHAP values, checks additivity, and returns long-form explanation tables.
- **`_normalize_shap_values(shap_values)`**: Ensures 2D NumPy array structure for single-output regression models.
- **`save_shap_explanations(explanations, output_path)`**: Persists output DataFrames to Parquet format.

## Configuration Parameters (`ShapRunConfig`)

```python
@dataclass(frozen=True)
class ShapRunConfig:
    sample_size: int = 200
    background_size: int = 2000
    seed: int = 42
```

- **`sample_size`**: Identical 200 error-tail instances as used in LIME.
- **`background_size`**: 2,000 historical training rows used to compute marginal feature expectations.
- **`seed`**: Fixed seed (42) for deterministic sampling.

## Output Schema (`shap_explanations.parquet`)

The output table contains 4,400 rows (200 instances $\times$ 22 features):

| Column | Type | Description |
|---|---|---|
| `row_index` | `int64` | Index of the instance (0 to 199). |
| `id`, `date` | `string`, `datetime64` | Identifier and date. |
| `actual_sales`, `pred_saved` | `float64` | Actual sales and baseline prediction. |
| `feature_name`, `feature_value` | `string`, `float64` | Feature name and raw value. |
| `shap_value` | `float64` | Signed SHAP value (contribution to prediction). |
| `abs_shap_value` | `float64` | Absolute SHAP value $|\phi_i|$ used for ranking. |
| `feature_rank` | `int64` | Feature importance rank for this instance (1 to 22). |
| `expected_value` | `float64` | Base value $\phi_0$. |
| `model_output` | `float64` | Model prediction $f(x)$. |
| `shap_sum` | `float64` | Sum of all SHAP values $\sum \phi_i$. |
| `shap_reconstruction` | `float64` | Reconstructed value $\phi_0 + \sum \phi_i$. |
| `reconstruction_delta` | `float64` | Additivity delta: $f(x) - (\phi_0 + \sum \phi_i)$. |

## Additivity Validation

`notebooks/06_SHAP_explainer.ipynb` performs validation of output reconstruction, saving summary statistics in `experiments/exp_001_baseline_lgbm/artifacts/shap_metrics_CA_1.json`:
- **Expected Value**: ~1.365 unit sales across background training data.
- **Max Reconstruction Error**: $< 10^{-8}$ (confirming exact mathematical additivity).

## Related Documentation

- [Error-Tail Sampling Strategy](03_sampling-by-error.md)
- [LIME Local Explanations](01_lime.md)
- [Notebook 06: SHAP Explainer](../04_notebooks/06_shap-explainer.md)
- [Module Reference: `src.explainers.shap_explainer`](../09_reference/explainers/02_shap-explainer.md)
