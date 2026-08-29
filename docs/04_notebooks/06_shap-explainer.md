# Notebook 06: SHAP Explainer

## Purpose

`notebooks/06_SHAP_explainer.ipynb` generates local SHAP (SHapley Additive exPlanations) values for the baseline LightGBM model using `TreeExplainer`. It reuses the error-tail sampling procedure (200 instances) to facilitate direct comparison with LIME and validates additive output reconstruction.

## Prerequisites

Run the notebook with `notebooks/` as the working directory. Its relative path configuration is:

```python
artifact_dir = Path('../experiments/exp_001_baseline_lgbm/artifacts')
feature_path = Path('../data/features/features.parquet')
prediction_path = artifact_dir / 'predictions_CA_1.parquet'
model_path = artifact_dir / 'lightgbm_CA_1.pkl'
```

The following input files are required:
- `data/features/features.parquet`;
- `experiments/exp_001_baseline_lgbm/artifacts/predictions_CA_1.parquet`;
- `experiments/exp_001_baseline_lgbm/artifacts/lightgbm_CA_1.pkl`.

The notebook requires Python 3, SHAP (`shap.TreeExplainer`), Pandas, NumPy, and `src.explainers.shap_explainer`.

## Configuration Parameters

| Parameter | Value | Description |
|---|---|---|
| `sample_size` | 200 | Total instances selected (100 `excellent`, 100 `bad`). |
| `background_size` | 2,000 | Training instances used as background reference for SHAP. |
| `seed` | 42 | Random seed for deterministic sample selection. |

## Execution Steps

### 1. Load data and select error-tail sample

- Reads `features.parquet` (filtered for `CA_1`) and encodes categorical columns.
- Reads holdout predictions from `predictions_CA_1.parquet`.
- Uses `select_error_tail_sample()` with `seed=42` to select the identical set of 200 instances as Notebook 05.

### 2. Build SHAP TreeExplainer & calculate SHAP values

- Loads `lightgbm_CA_1.pkl`.
- Samples 2,000 background rows from historical training data (`date <= '2016-04-24'`).
- Instantiates `shap.TreeExplainer(model, data=background_matrix)`.
- Calls `explain_shap_sample()` to compute exact Shapley values for all 22 features across all 200 instances.

### 3. Verify SHAP additivity & reconstruction

- Verifies that $\text{Expected Value} + \sum \text{SHAP Values} = \text{Model Output}$.
- Computes reconstruction deltas and records summary metrics.

### 4. Persist SHAP artifacts

Saves sample rows, full long-form SHAP explanations, and validation metrics to `experiments/exp_001_baseline_lgbm/artifacts/`.

## Outputs

The notebook generates three artifacts:

| Output File | Format | Rows/Size | Description |
|---|---|---:|---|
| `shap_sample_rows.parquet` | Parquet | 200 rows | Analysis sample instances. |
| `shap_explanations.parquet` | Parquet | 4,400 rows | Long-form SHAP values (200 instances $\times$ 22 features). |
| `shap_metrics_CA_1.json` | JSON | Metadata | SHAP expected value, mean absolute delta, and max reconstruction error. |

## Dependencies

```text
06_SHAP_explainer.ipynb
 ├── src.explainers.shap_explainer (explain_shap_sample, save_shap_explanations, build_shap_explainer)
 ├── src.explainers.lime_explainer (select_error_tail_sample, encode_for_model, load_saved_model, load_prediction_artifact)
 ├── data/features/features.parquet
 ├── experiments/exp_001_baseline_lgbm/artifacts/predictions_CA_1.parquet
 └── experiments/exp_001_baseline_lgbm/artifacts/lightgbm_CA_1.pkl
```

## Limitations and Notes

- **TreeExplainer Specificity**: Uses tree structure optimization; exact additivity is maintained ($|\text{delta}| < 10^{-10}$).
- **Shared Utilities**: Reuses amostragem and encoding helper functions from `src/explainers/lime_explainer.py`.

## Related Documentation

- [SHAP Local Explanations](../06_explainability/02_shap.md)
- [Error-Tail Sampling Strategy](../06_explainability/03_sampling-by-error.md)
- [XAI Evaluation Protocol](../06_explainability/04_evaluation-protocol.md)
- [Module Reference: `src.explainers.shap_explainer`](../09_reference/explainers/02_shap-explainer.md)
