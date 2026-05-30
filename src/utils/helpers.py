"""
Helper utilities and common functions.
"""
import pickle
from pathlib import Path


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
