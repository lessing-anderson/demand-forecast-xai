# Notebook 07: Faithfulness Measuring

## Purpose

`notebooks/07_faithfulness_measuring.ipynb` evaluates the **faithfulness (fidelity)** of local LIME and SHAP explanations for the LightGBM demand forecasting model using **iterative feature deletion (ablation)**.

The protocol concentrates on holdout instances where the model incurred the largest prediction errors (`sample_bucket == 'bad'`), testing whether masking features identified as most influential by each explainer produces a corresponding degradation in prediction accuracy.

## Prerequisites

Run the notebook with `notebooks/` as the working directory. Its relative path configuration is:

```python
feature_path = Path('../data/features/features.parquet')
artifact_dir = Path('../experiments/exp_001_baseline_lgbm/artifacts')
split_date = pd.Timestamp('2016-04-24')
```

The following input artifacts must exist from prior notebook runs:
- `data/features/features.parquet` (produced by Notebook 03);
- `experiments/exp_001_baseline_lgbm/artifacts/lightgbm_CA_1.pkl` (produced by Notebook 04);
- `experiments/exp_001_baseline_lgbm/artifacts/predictions_CA_1.parquet` (produced by Notebook 04);
- `experiments/exp_001_baseline_lgbm/artifacts/lime_explanations.parquet` (produced by Notebook 05);
- `experiments/exp_001_baseline_lgbm/artifacts/shap_explanations.parquet` (produced by Notebook 06).

The notebook uses Python 3, Pandas, NumPy, Matplotlib, Seaborn, `src.explainers` (`FaithfulnessConfig`, `evaluate_faithfulness_by_bucket`, `plot_faithfulness_curves`, `save_faithfulness_artifacts`), and `src.utils.metrics`.

## Protocol Configuration (`FaithfulnessConfig`)

```python
config = FaithfulnessConfig(
    max_steps=10,
    error_bucket='bad',
    actual_col='actual_sales',
    id_col='id',
    target_col='sales',
)
```

- **`max_steps`**: 10 sequential feature deletion steps (removing top-1 to top-10 features).
- **`error_bucket`**: `'bad'` (100 high-error holdout cases).
- **Obfuscation Strategy**: Continuous variables are replaced by their historical training median; categorical variables by their mode.

## Execution Steps

### 1. Load Data and Explanations
- Loads the feature matrix, baseline model, holdout predictions, and both LIME and SHAP long-form explanation tables.
- Encodes categorical columns consistently via `encode_for_model()`.

### 2. Feature Ranking Extraction
- Extracts per-instance deletion rankings:
  - **LIME**: Sorted by descending absolute surrogate weight ($|\text{weight}|$).
  - **SHAP**: Sorted by descending absolute Shapley value ($|\phi_i|$).

### 3. Iterative Feature Deletion
- For each instance and step $k \in [0, 10]$:
  - Replaces top-$k$ features with background neutral values.
  - Re-evaluates the LightGBM model on the modified feature vector.
  - Records the perturbed prediction and error.

### 4. Metric Aggregation
- Pools instance predictions at each step and computes global **RMSE** and **WRMSSE** degradation relative to the unperturbed baseline (step 0).
- Calculates the Area Under the Degradation Curve (AUC) for both metrics.

### 5. Visualisation and Artifact Persistence
- Renders comparative degradation curves for LIME and SHAP.
- Persists output tables and summary metrics to `experiments/exp_001_baseline_lgbm/artifacts/`.

## Generated Outputs

| Artifact | Type | Description |
|---|---|---|
| `faithfulness_deletion_results.parquet` | Parquet | Detailed row-level predictions across all 10 deletion steps for both methods. |
| `faithfulness_metric_curve.parquet` | Parquet | Aggregated step-by-step RMSE, WRMSSE, and error deltas. |
| `faithfulness_metrics.json` | JSON | Summary indicators including baseline/final errors and AUC metrics. |

## Related Documentation

- [XAI Evaluation Protocol](../../06_explainability/04_evaluation-protocol.md)
- [Module Reference: `src.explainers.faithfulness`](../../09_reference/explainers/03_faithfulness.md)
