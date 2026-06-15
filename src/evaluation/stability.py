import numpy as np
from scipy.stats import spearmanr


def perturb_instance(
    instance,
    numeric_features,
    perturbation_pct=0.01,
    random_state=42,
):
    rng = np.random.default_rng(random_state)

    perturbed = instance.copy()

    for feature in numeric_features:
        noise = rng.uniform(
            -perturbation_pct,
            perturbation_pct,
        )

        perturbed.loc[:, feature] = (
            perturbed[feature] * (1 + noise)
        )

    return perturbed


def calculate_stability(
    original_explanation,
    perturbed_explanation,
):
    merged = original_explanation[
        ["feature", "importance"]
    ].merge(
        perturbed_explanation[
            ["feature", "importance"]
        ],
        on="feature",
        suffixes=("_original", "_perturbed"),
    )

    stability, _ = spearmanr(
        merged["importance_original"].abs(),
        merged["importance_perturbed"].abs(),
    )

    return stability