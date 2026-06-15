# Notebook 09: Computational Cost Measuring

## Purpose

`notebooks/09_computational_cost_measuring.ipynb` benchmarks and compares the **computational cost** (wall-clock latency in seconds) required to generate local explanations using **LIME Local** vs. **SHAP Local**.

The evaluation isolates the instance-level inference cost under identical production-serving assumptions (explainers pre-built once).

## Key Guidelines

- **Strictly Local Scope**: Directly compares instance-level local explanations (TreeExplainer local values vs. local linear surrogate).
- **Metric**: Execution time in seconds per instance, measured with the high-precision `Timer` utility.
- **Amostragem**: Executed over the 200 holdout error-tail instances (100 `excellent` + 100 `bad`).

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
- `experiments/exp_001_baseline_lgbm/artifacts/lime_sample_rows.parquet` (produced by Notebook 05);
- `experiments/exp_001_baseline_lgbm/artifacts/shap_sample_rows.parquet` (produced by Notebook 06).

The notebook uses Python 3, Pandas, NumPy, Seaborn, Matplotlib, and `src.explainers` (`benchmark_computational_cost`, `plot_computational_cost_comparison`, `save_computational_cost_artifacts`).

## Execution Steps

### 1. Initialize & Pre-build Explainers
- Loads the trained LightGBM booster and sample instances.
- Pre-builds `shap.TreeExplainer` and `LimeTabularExplainer` using a 2,000-row historical training background.

### 2. Benchmark Local Explanation Latency
- Iterates across all 200 instances measuring wall-clock duration:
  - **SHAP Local**: Calls `explain_local_shap_single()` timing TreeExplainer inference for the single feature vector.
  - **LIME Local**: Calls `explain_local_lime_single()` timing surrogate perturbation, model scoring, and linear fitting.

### 3. Compute Summary Statistics
- Calculates mean, standard deviation, median, 25th/75th percentiles, minimum, and maximum processing time for both methods.

### 4. Visualisation and Artifact Persistence
- Plots side-by-side mean latency bar chart and runtime dispersion boxplot.
- Saves benchmark tables, JSON metrics, and figure to `experiments/exp_001_baseline_lgbm/artifacts/`.

## Generated Outputs

| Artifact | Type | Description |
|---|---|---|
| `computational_cost_results.parquet` | Parquet | Individual wall-clock execution times (seconds) per instance and method. |
| `computational_cost_metrics.json` | JSON | Aggregated latency metrics (mean, median, std, quartiles, min/max). |
| `fig_computational_cost_comparison.png` | PNG | Dual-panel visualization comparing mean runtime and dispersion. |

## Related Documentation

- [XAI Evaluation Protocol](../../06_explainability/04_evaluation-protocol.md)
- [Module Reference: `src.explainers.computational_cost`](../../09_reference/explainers/05_computational-cost.md)
