# `src.explainers.stability`

## Purpose

Implements the **stability (robustness)** measurement module for local XAI methods (SHAP Local vs. LIME Local) under continuous feature perturbation.

Stability assesses whether local feature attributions remain consistent and robust when small, controlled Gaussian noise is introduced to continuous attributes of test instances, while preserving categorical variables on the valid data manifold.

## Configuration Class

### `StabilityConfig`

```python
@dataclass(frozen=True)
class StabilityConfig:
    noise_scale: float = 0.03
    seed: int = 42
    num_lime_features: int = 10
    n_jobs: int = 4
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `noise_scale` | `float` | `0.03` | Relative noise magnitude (3% of feature empirical standard deviation $\sigma_j$). |
| `seed` | `int` | `42` | Random seed for reproducible Gaussian noise generation. |
| `num_lime_features` | `int` | `10` | Top features explained per instance in LIME. |
| `n_jobs` | `int` | `4` | Number of worker threads for parallel LIME execution. |

## Public Functions

### `add_continuous_perturbation(sample_frame, feature_names, categorical_columns, background_frame=None, noise_scale=0.03, seed=42)`

Introduces zero-mean Gaussian noise $\epsilon_j \sim \mathcal{N}(0, (\eta \cdot \sigma_j)^2)$ to continuous variables, clipping non-negative fields (prices and lag sales) at zero. Categorical attributes (event codes, store/item codes, SNAP indicators) remain unchanged.

### `recalculate_perturbed_explanations(model, perturbed_sample, background_frame, feature_names, categorical_columns=None, categorical_names=None, num_lime_features=10, seed=42, n_jobs=4)`

Recalculates local SHAP (TreeExplainer) and LIME (local linear surrogate) explanations on the perturbed feature dataset. Returns `(perturbed_shap_df, perturbed_lime_df)`.

### `compute_ranking_stability(orig_shap_df, pert_shap_df, orig_lime_df, pert_lime_df, feature_names)`

Computes feature ranking consistency before and after noise using the **Spearman Rank Correlation Coefficient** ($\rho$) for each instance. Returns per-instance scores DataFrame and aggregated summary metrics (mean, median, std overall and by error bucket).

### `plot_stability_comparison(stability_df, title=..., save_path=None)`

Generates publication-quality comparison charts:
1. **Boxplots**: Grouped by error bucket (`excellent` vs `bad`).
2. **KDE / Density Plots**: Showing empirical distribution of Spearman rank correlations for SHAP and LIME.

### `save_stability_artifacts(stability_results, summary_metrics, output_dir)`

Persists per-instance stability scores to `stability_results.parquet` and summary statistics to `stability_metrics.json`.

## Related Documentation

- [Notebook 08: Stability Measuring](../../04_notebooks/08_stability-measuring.md)
- [XAI Evaluation Protocol](../../06_explainability/04_evaluation-protocol.md)
