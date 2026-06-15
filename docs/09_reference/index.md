# Python Code Reference

This section documents the executable Python modules organized in `src/`.

| Area | Modules |
|---|---|
| **Data (`src/data`)** | [loader](data/01_loader.md), [data processing](data/02_data-processing.md), [feature creation](data/03_feature-creation.md) |
| **Models (`src/models`)** | [base model](models/01_base-model.md), [LightGBM model](models/02_lightgbm-model.md), [splitting](models/03_splitting.md) |
| **Explainers (`src/explainers`)** | [LIME explainer](explainers/01_lime-explainer.md), [SHAP explainer](explainers/02_shap-explainer.md), [faithfulness](explainers/03_faithfulness.md), [stability](explainers/04_stability.md), [computational cost](explainers/05_computational-cost.md) |
| **Utilities (`src/utils`)** | [metrics](utils/01_metrics.md), [helpers](utils/03_helpers.md) |

`src/__init__.py` and the package-level `__init__.py` files structure the namespace. `src/explainers/__init__.py` re-exports public explainer and measurement functions; the detailed signatures and usage are documented in the respective explainer reference guides.
