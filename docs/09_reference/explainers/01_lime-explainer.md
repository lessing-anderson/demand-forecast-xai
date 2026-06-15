# `src.explainers.lime_explainer`

## Purpose

Provides reusable utilities for selecting error-stratified cases and generating local LIME explanations for a persisted LightGBM model.

## Main API

| Function or class | Responsibility |
|---|---|
| `LimeRunConfig` | Immutable configuration container for LIME runs. |
| `load_saved_model(path)` | Loads a pickled model. |
| `load_prediction_artifact(path)` | Loads predictions and adds error metadata. |
| `add_error_columns(df)` | Adds residual, absolute error, and error quantile. |
| `encode_for_model(df)` | Encodes text and boolean features for model use. |
| `select_error_tail_sample(predictions, sample_size, seed)` | Selects balanced excellent and poor error cases. |
| `build_lime_explainer(...)` | Creates a regression `LimeTabularExplainer`. |
| `explain_lime_sample(...)` | Generates normalized long-form LIME output. |
| `save_lime_explanations(explanations, path)` | Persists explanations as Parquet. |

## Output Contract

Each explanation row contains row metadata, the feature name and value, LIME weight, rank, label, and intercept. `explain_lime_sample()` uses joblib threading for parallel case explanation.
