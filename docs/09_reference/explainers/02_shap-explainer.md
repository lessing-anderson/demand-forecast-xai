# `src.explainers.shap_explainer`

## Purpose

Builds TreeSHAP explanations for the persisted LightGBM booster and serializes them in long format.

## Main API

| Function or class | Responsibility |
|---|---|
| `ShapRunConfig` | Immutable configuration container for SHAP runs. |
| `load_saved_model(path)` | Loads a pickled model. |
| `build_shap_explainer(model, background_frame)` | Creates a `shap.TreeExplainer`. |
| `explain_shap_sample(...)` | Produces long-form SHAP contributions. |
| `save_shap_explanations(explanations, path)` | Persists explanations as Parquet. |

## Output Contract

Each output row contains case metadata, feature contribution, absolute contribution, rank, expected value, model output, SHAP sum, reconstructed output, and reconstruction delta. The delta is the primary additivity consistency check.

## Reuse

The module imports common artifact loading, encoding, and error-sampling helpers from `lime_explainer`.
