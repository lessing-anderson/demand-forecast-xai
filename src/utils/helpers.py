"""
Helper utilities and common functions.
"""
import sys
import pickle
from pathlib import Path
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path.cwd().parent))


def save_pickle(obj, filepath):
    """Save object to pickle file."""
    Path(filepath).parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(obj, f)
    print(f"Saved to {filepath}")


def load_pickle(filepath):
    """Load object from pickle file."""
    with open(filepath, 'rb') as f:
        obj = pickle.load(f)
    print(f"Loaded from {filepath}")
    return obj


def select_prediction_samples(
    predictions,
    item_col='id',
    true_col='y_true',
    pred_col='y_pred',
    n_good=100,
    n_bad=100,
    max_per_item=3,
    zero_prediction_threshold=0.5,
    random_state=42
):
    df = predictions.copy()

    # Prediction errors
    df['error'] = df[pred_col] - df[true_col]
    df['absolute_error'] = df['error'].abs()

    df['error_direction'] = np.where(
        df['error'] >= 0,
        'over',
        'under'
    )

    # -------------------------
    # Bad predictions
    # -------------------------

    n_bad_over = n_bad // 2
    n_bad_under = n_bad - n_bad_over

    bad_over = (
        df[df['error_direction'] == 'over']
        .sort_values('absolute_error', ascending=False)
        .groupby(item_col, group_keys=False)
        .head(max_per_item)
        .head(n_bad_over)
    )

    bad_under = (
        df[df['error_direction'] == 'under']
        .sort_values('absolute_error', ascending=False)
        .groupby(item_col, group_keys=False)
        .head(max_per_item)
        .head(n_bad_under)
    )

    bad = pd.concat([bad_over, bad_under])

    bad['prediction_quality'] = 'bad'

    # -------------------------
    # Good predictions
    # -------------------------

    good_candidates = df.drop(index=bad.index)

    # Remove trivial zero-demand cases
    good_candidates = good_candidates[
        ~(
            (good_candidates[true_col] == 0) &
            (good_candidates[pred_col].abs() < zero_prediction_threshold)
        )
    ]

    good = (
        good_candidates
        .sort_values('absolute_error', ascending=True)
        .groupby(item_col, group_keys=False)
        .head(max_per_item)
        .head(n_good)
    )

    good['prediction_quality'] = 'good'

    # -------------------------
    # Final sample
    # -------------------------

    selected = pd.concat([good, bad])

    selected = selected.sample(
        frac=1,
        random_state=random_state
    ).copy()

    # Preserve the original row position from the full predictions/X_test table.
    selected['instance_id'] = selected.index.to_numpy()
    selected = selected.reset_index(drop=True)

    return selected