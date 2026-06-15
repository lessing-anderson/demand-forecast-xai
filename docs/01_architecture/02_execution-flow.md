# Execution Flow

## Effective Pipeline

```mermaid
flowchart LR
    n01["data/raw\nRaw CSVs"] --> n02
    n02["02_data_processing Notebook"] --> processed["data/processed\nDimensional Parquet"]
    processed --> n03["03_feature_engineering Notebook"]
    n03 --> features["data/features\nFeatures Parquet"]
    features --> n04["04_lightgbm_baseline Notebook"]
    n04 --> artifacts["experiments/.../artifacts\nModel, Predictions, and Metrics"]
    features --> n05["05_LIME_explainer Notebook"]
    artifacts --> n05
    artifacts --> n06
    features --> n06["06_SHAP_explainer Notebook"]
    n05 --> lime["experiments/.../artifacts\nLIME Parquet"]
    n06 --> shap["experiments/.../artifacts\nSHAP Parquet"]
    lime --> n07["07_faithfulness_measuring Notebook"]
    lime --> n08["08_stability_measuring Notebook"]
    lime --> n09["09_computational_cost_measuring Notebook"]
    shap --> n07
    shap --> n08
    shap --> n09
```

## Required Order

1. `02_data_processing.ipynb` generates the Parquet tables in `data/processed/`.
2. `03_feature_engineering.ipynb` generates `data/features/features.parquet`.
3. `04_lightgbm_baseline.ipynb` generates the baseline model artifacts.
4. `05_LIME_explainer.ipynb` and `06_SHAP_explainer.ipynb` can run independently after the baseline.
5. `07_faithfulness_measuring.ipynb`, `08_stability_measuring.ipynb`, and `09_computational_cost_measuring.ipynb` run after the baseline, LIME, and SHAP explanations are persisted.

## Inputs and Outputs

| Stage | Inputs | Persisted outputs |
|---|---|---|
| Processing | Three M5 CSVs | Five dimensional/fact parquet files. |
| Feature engineering | Processed parquet tables | One feature parquet file. |
| Baseline | Parquet file with features | Pickled model, holdout predictions, metrics JSON, and feature importance. |
| LIME | Features, model, and predictions | Parquet with LIME sample rows and local explanations. |
| SHAP | Features, model, and predictions | Parquet with SHAP sample rows, local explanations, and metrics JSON. |
| Faithfulness | Model, features, holdout predictions, LIME/SHAP explanations | `faithfulness_deletion_results.parquet`, `faithfulness_metric_curve.parquet`, `faithfulness_metrics.json`. |
| Stability | Model, features, LIME/SHAP explanations | `stability_results.parquet`, `stability_metrics.json`. |
| Computational Cost | Model, features, sample rows | `computational_cost_results.parquet`, `computational_cost_metrics.json`, `fig_computational_cost_comparison.png`. |

The notebooks use relative paths and assume `notebooks/` as their working directory. When run elsewhere, their paths must be adjusted.

