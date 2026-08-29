import numpy as np
import pandas as pd


def get_local_ranking(
    explanations_df,
    instance_id,
    instance_col="instance_id",
    feature_col="feature",
    importance_col="importance",
):
    local_explanation = explanations_df[
        explanations_df[instance_col] == instance_id
    ].copy()

    local_explanation["abs_importance"] = (
        local_explanation[importance_col].abs()
    )

    return (
        local_explanation
        .sort_values("abs_importance", ascending=False)
        [feature_col]
        .tolist()
    )


def perturb_instance(
    instance,
    reference_data,
    features,
    num_samples=100,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    perturbed = pd.concat(
        [instance] * num_samples,
        ignore_index=True,
    )

    sampled_idx = rng.choice(
        reference_data.index,
        size=num_samples,
        replace=True,
    )

    sampled = (
        reference_data
        .loc[sampled_idx, features]
        .reset_index(drop=True)
    )

    for feature in features:
        perturbed[feature] = sampled[feature].values

    # Preserve categorical metadata expected by LightGBM
    categorical_cols = instance.select_dtypes(
        include="category"
    ).columns

    for col in categorical_cols:
        perturbed[col] = pd.Categorical(
            perturbed[col],
            categories=instance[col].cat.categories,
            ordered=instance[col].cat.ordered,
        )

    return perturbed


def evaluate_local_fidelity(
    model,
    instance,
    reference_data,
    ranked_features,
    steps=(1, 3, 5, 10),
    num_samples=100,
    random_state=42,
):
    baseline_pred = model.predict(instance)[0]

    results = []

    for k in steps:
        features = ranked_features[:k]

        perturbed = perturb_instance(
            instance=instance,
            reference_data=reference_data,
            features=features,
            num_samples=num_samples,
            random_state=random_state,
        )

        perturbed_preds = model.predict(perturbed)

        prediction_change = np.abs(
            perturbed_preds - baseline_pred
        )

        results.append({
            "k": k,
            "baseline_prediction": baseline_pred,
            "mean_prediction_change": prediction_change.mean(),
        })

    return pd.DataFrame(results)