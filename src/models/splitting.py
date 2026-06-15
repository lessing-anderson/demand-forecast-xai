"""
Data splitting and train/test separation module.
Handles temporal train/test splits and ground truth isolation.
"""
import pandas as pd
import numpy as np

def split_train_holdout(df, split_date):
    train_df = df[df['date'] <= split_date].copy()
    holdout_df = df[df['date'] > split_date].copy()

    return train_df, holdout_df

def split_features_target(df, target_col='sales', drop_cols=None):
    drop_cols = [col for col in drop_cols if col in df.columns]

    X = df.drop(columns=drop_cols)
    y = df[target_col].copy()

    return X, y