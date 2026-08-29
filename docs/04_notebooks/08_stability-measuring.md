# Notebook 08: Stability Measuring

## Purpose

`notebooks/08_stability_measuring.ipynb` evaluates the **stability (robustness)** of local LIME and SHAP explanations under continuous feature perturbation.

In explainable machine learning, explanation stability assesses whether feature attribution rankings $\mathbf{\Phi}(\mathbf{x})$ remain consistent when small, controlled variations $\mathbf{x}' = \mathbf{x} + \boldsymbol{\epsilon}$ are introduced to the input instances.

## Prerequisites

Run the notebook with `notebooks/` as the working directory. Its relative path configuration is:

```python
feature_path = Path('../data/features/features.parquet')
artifact_dir = Path('../experiments/exp_001_baseline_lgbm/artifacts')
split_date = pd.Timestamp('2016-04-24')
```

The following input files are required:
- `data/features/features.parquet` (produced by Notebook 03);
- `experiments/exp_001_baseline_lgbm/artifacts/lightgbm_CA_1.pkl` (produced by Notebook 04);
- `experiments/exp_001_baseline_lgbm/artifacts/predictions_CA_1.parquet` (produced by Notebook 04);
- `experiments/exp_001_baseline_lgbm/artifacts/lime_explanations.parquet` (produced by Notebook 05);
- `experiments/exp_001_baseline_lgbm/artifacts/shap_explanations.parquet` (produced by Notebook 06).

The notebook uses Python 3, Pandas, NumPy, SciPy (`scipy.stats.spearmanr`), Seaborn, Matplotlib, and `src.explainers` (`StabilityConfig`, `add_continuous_perturbation`, `recalculate_perturbed_explanations`, `compute_ranking_stability`, `plot_stability_comparison`, `save_stability_artifacts`).

## Protocol Configuration (`StabilityConfig`)

```python
config = StabilityConfig(
    noise_scale=0.03,  # 3% of feature empirical std
    seed=42,
    num_lime_features=10,
    n_jobs=4,
)
```

- **`noise_scale`**: 0.03 (adds zero-mean Gaussian noise $\epsilon_j \sim \mathcal{N}(0, (0.03 \cdot \sigma_j)^2)$ to continuous features).
- **Categorical Preservation**: Categorical integer codes (store, item, event, SNAP) are kept strictly unperturbed to avoid out-of-domain artifacts.
- **Consistency Metric**: **Spearman Rank Correlation** ($\rho$) comparing feature importance orders before and after noise.

## Execution Steps

### 1. Load Data & Prepare Background
- Loads the feature matrix and selects the 200 error-tail instances (`lime_sample_rows.parquet` / `shap_sample_rows.parquet`).
- Prepares a 2,000-row historical training background dataset.

### 2. Inject Continuous Gaussian Noise
- Calls `add_continuous_perturbation()` to apply noise to continuous variables (`sell_price`, `sales_lag_*`, `rolling_mean_*`), enforcing non-negativity where appropriate.

### 3. Recalculate Explanations
- Re-computes local SHAP values using `TreeExplainer`.
- Re-computes local LIME surrogate weights using `LimeTabularExplainer` (4 parallel threads).

### 4. Measure Ranking Consistency
- Calls `compute_ranking_stability()` to compute per-instance Spearman rank correlations ($\rho_{\text{SHAP}}$ and $\rho_{\text{LIME}}$).
- Aggregates overall and bucket-wise (`excellent` vs `bad`) summary metrics (mean, median, standard deviation).

### 5. Visualisation and Artifact Persistence
- Plots side-by-side boxplots (by error bucket) and KDE distribution curves.
- Saves results to `experiments/exp_001_baseline_lgbm/artifacts/`.

## Generated Outputs

| Artifact | Type | Description |
|---|---|---|
| `stability_results.parquet` | Parquet | Per-instance Spearman rank correlation scores ($\rho_{\text{SHAP}}$, $\rho_{\text{LIME}}$, $\Delta\rho$). |
| `stability_metrics.json` | JSON | Aggregated stability metrics overall and segmented by error bucket. |

## Related Documentation

- [XAI Evaluation Protocol](../../06_explainability/04_evaluation-protocol.md)
- [Module Reference: `src.explainers.stability`](../../09_reference/explainers/04_stability.md)
