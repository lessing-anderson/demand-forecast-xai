# Notebook 05: LIME Explainer

## Purpose

`notebooks/05_LIME_explainer.ipynb` generates local LIME (Local Interpretable Model-agnostic Explanations) explanations for the baseline LightGBM model. It selects 200 holdout observations based on extreme prediction errors (100 low error, 100 high error) and computes feature-level explanation weights.

## Prerequisites

Run the notebook with `notebooks/` as the working directory. Its relative path configuration is:

```python
artifact_dir = Path('../experiments/exp_001_baseline_lgbm/artifacts')
feature_path = Path('../data/features/features.parquet')
prediction_path = artifact_dir / 'predictions_CA_1.parquet'
model_path = artifact_dir / 'lightgbm_CA_1.pkl'
```

The following files must exist:
- `data/features/features.parquet`;
- `experiments/exp_001_baseline_lgbm/artifacts/predictions_CA_1.parquet`;
- `experiments/exp_001_baseline_lgbm/artifacts/lightgbm_CA_1.pkl`.

The notebook requires Python 3, LIME (`lime.lime_tabular`), Joblib, Pandas, NumPy, and `src.explainers.lime_explainer`.

## Configuration Parameters

| Parameter | Value | Description |
|---|---|---|
| `sample_size` | 200 | Total instances selected (100 `excellent`, 100 `bad`). |
| `background_size` | 2,000 | Historical training rows sampled for LIME background distribution. |
| `num_features` | 10 | Top features returned per LIME explanation. |
| `random_state` | 42 | Random seed for sampling and LIME explainer. |
| `n_jobs` | 4 | Parallel threads for parallel explanation generation. |

## Execution Steps

### 1. Load data and align sample

- Loads `features.parquet` (filtered for `CA_1`) and converts string columns to categorical integer codes via `encode_for_model()`.
- Loads holdout predictions from `predictions_CA_1.parquet`.
- Calls `select_error_tail_sample()` to deterministically select 200 instances:
  - 100 rows with smallest absolute error (`sample_bucket == 'excellent'`);
  - 100 rows with largest absolute error (`sample_bucket == 'bad'`).
- Merges sample predictions back with feature table on `['id', 'date']`.

### 2. Prepare background data & verify model predictions

- Samples 2,000 background rows from training data (`date <= '2016-04-24'`) to estimate feature distributions.
- Loads `lightgbm_CA_1.pkl` and predicts on the sample to ensure consistency with `pred_saved` (max prediction delta: 0.00000000).

### 3. Generate LIME explanations in parallel

- Calls `explain_lime_sample()`:
  - Instantiates `LimeTabularExplainer(mode='regression', discretize_continuous=True)`.
  - Computes LIME feature weights for each row using 4 parallel threads.
  - Normalizes results into a long-format DataFrame (2,000 rows = 200 instances $\times$ 10 features).

### 4. Persist LIME artifacts

Saves sample rows and explanation results to Parquet in `experiments/exp_001_baseline_lgbm/artifacts/`.

## Outputs

The notebook generates two Parquet files:

| Output File | Rows | Cols | Description |
|---|---:|---:|---|
| `lime_sample_rows.parquet` | 200 | 35 | Metadata and features of the 200 selected analysis instances. |
| `lime_explanations.parquet` | 2,000 | 19 | Long-form LIME weights (`feature_name`, `feature_value`, `weight`, `feature_rank`, `intercept`). |

## Dependencies

```text
05_LIME_explainer.ipynb
 ├── src.explainers.lime_explainer (encode_for_model, select_error_tail_sample, explain_lime_sample, load_saved_model, load_prediction_artifact, save_lime_explanations)
 ├── data/features/features.parquet
 ├── experiments/exp_001_baseline_lgbm/artifacts/predictions_CA_1.parquet
 └── experiments/exp_001_baseline_lgbm/artifacts/lightgbm_CA_1.pkl
```

## Limitations and Notes

- **Discretization Sensitivity**: LIME continuous feature discretization can cause small variations if background samples change.
- **Computation Time**: Parallelized across 4 workers; CPU bound due to local surrogate model fitting for each instance.

## Related Documentation

- [LIME Local Explanations](../06_explainability/01_lime.md)
- [Error-Tail Sampling Strategy](../06_explainability/03_sampling-by-error.md)
- [XAI Evaluation Protocol](../06_explainability/04_evaluation-protocol.md)
- [Module Reference: `src.explainers.lime_explainer`](../09_reference/explainers/01_lime-explainer.md)
