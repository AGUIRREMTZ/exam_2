"""
Utility functions for loading and processing the NSL-KDD dataset.
"""
import arff
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def load_kdd_dataset(data_path):
    """
    Load the NSL-KDD dataset from an ARFF file.
    
    Args:
        data_path (str): Path to the ARFF file
        
    Returns:
        pd.DataFrame: Dataset as a pandas DataFrame
    """
    with open(data_path, 'r') as train_set:
        dataset = arff.load(train_set)
        attributes = [attr[0] for attr in dataset["attributes"]]
    return pd.DataFrame(dataset["data"], columns=attributes)


def train_val_test_split(df, rstate=42, shuffle=True, stratify=None):
    """
    Split dataset into train, validation, and test sets.
    
    Args:
        df (pd.DataFrame): Input dataset
        rstate (int): Random state for reproducibility
        shuffle (bool): Whether to shuffle the data
        stratify (str): Column name to use for stratified sampling
        
    Returns:
        tuple: (train_set, val_set, test_set)
    """
    strat = df[stratify] if stratify else None
    train_set, test_set = train_test_split(
        df, test_size=0.4, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    strat = test_set[stratify] if stratify else None
    val_set, test_set = train_test_split(
        test_set, test_size=0.5, random_state=rstate, shuffle=shuffle, stratify=strat
    )
    return train_set, val_set, test_set


def add_null_values_for_demo(X_train):
    """
    Add null values to demonstrate imputation (for demo purposes).
    
    Args:
        X_train (pd.DataFrame): Training data
        
    Returns:
        pd.DataFrame: Data with added null values
    """
    X_train_copy = X_train.copy()
    X_train_copy.loc[
        (X_train_copy["src_bytes"] > 400) & (X_train_copy["src_bytes"] < 800), 
        "src_bytes"
    ] = np.nan
    X_train_copy.loc[
        (X_train_copy["dst_bytes"] > 500) & (X_train_copy["dst_bytes"] < 200), 
        "dst_bytes"
    ] = np.nan
    return X_train_copy
