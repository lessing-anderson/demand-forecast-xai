"""
Data splitting and train/test separation module.
Handles temporal train/test splits and ground truth isolation.
"""
import pandas as pd
import numpy as np


def isolate_ground_truth(df, split_date):
    """
    Separate ground truth labels from training data to prevent leakage.
    
    Args:
        df: Input dataframe with sales target
        split_date: Date string (YYYY-MM-DD) to split train/test
        
    Returns:
        Tuple of (training_df with nullified future sales, ground_truth_df)
    """
    print(f"Isolating ground truth and nullifying target after {split_date}...")
    
    df_ground_truth = df[df['date'] > split_date][['id', 'date', 'sales']].copy()
    df_ground_truth.rename(columns={'sales': 'actual_sales'}, inplace=True)
    
    df.loc[df['date'] > split_date, 'sales'] = np.nan
    
    return df, df_ground_truth


def get_train_test_split(df, split_date):
    """
    Split data into training and test sets by date.
    
    Args:
        df: Input dataframe
        split_date: Date string (YYYY-MM-DD) to split
        
    Returns:
        Tuple of (train_df, test_df)
    """
    train_df = df[df['date'] <= split_date].copy()
    test_df = df[df['date'] > split_date].copy()
    
    return train_df, test_df


def get_features_and_target(df, target_col='sales', drop_cols=None):
    """
    Separate features from target variable.
    
    Args:
        df: Input dataframe with features and target
        target_col: Name of target column
        drop_cols: List of columns to exclude from features
        
    Returns:
        Tuple of (X_features, y_target)
    """
    if drop_cols is None:
        drop_cols = ['id', 'date', 'd', target_col]
    
    # Remove cols that don't exist
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    X = df.drop(columns=drop_cols)
    y = df[target_col].copy() if target_col in df.columns else None
    
    return X, y


def prepare_train_data(df, split_date, target_col='sales', drop_cols=None):
    """
    Prepare training data: drop NaN targets, split features/target.
    
    Args:
        df: Full processed dataframe
        split_date: Date for train/test split
        target_col: Name of target column
        drop_cols: Columns to exclude from features
        
    Returns:
        Tuple of (X_train, y_train, test_df)
    """
    train_df, test_df = get_train_test_split(df, split_date)
    
    # Remove rows with NaN target in training set
    train_df = train_df.dropna(subset=[target_col])
    
    X_train, y_train = get_features_and_target(train_df, target_col, drop_cols)
    
    return X_train, y_train, test_df


def prepare_test_data(test_df, target_col='sales', drop_cols=None):
    """
    Prepare test data for inference.
    
    Args:
        test_df: Test set dataframe
        target_col: Name of target column
        drop_cols: Columns to exclude from features
        
    Returns:
        Tuple of (X_test, test_df_with_metadata)
    """
    X_test, _ = get_features_and_target(test_df, target_col, drop_cols)
    
    return X_test, test_df
