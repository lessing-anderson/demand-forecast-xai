# `src.models.base_model`

## Purpose

Defines the abstract interface for forecasting models.

## `BaseModel`

The class stores the model name, underlying model object, feature names, and fit state. Implementations must provide `train(X_train, y_train, **kwargs)`, `predict(X)`, and `get_feature_importance()`. It also provides `set_feature_names(feature_names)` and `get_model()`.
