"""
Utility functions for Natural Cycles assignment analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score


def preprocess_features(X, categorical_columns=None):
    """
    Preprocess features for machine learning models.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    categorical_columns : list, optional
        List of categorical column names to encode

    Returns:
    --------
    X_processed : pandas.DataFrame
        Processed feature matrix
    label_encoders : dict
        Dictionary of fitted label encoders
    """
    X_processed = X.copy()
    label_encoders = {}

    # Handle categorical variables with label encoding
    if categorical_columns is None:
        categorical_columns = X.select_dtypes(include=["object"]).columns

    for col in categorical_columns:
        if col in X_processed.columns:
            le = LabelEncoder()
            X_processed[col] = le.fit_transform(X_processed[col].astype(str))
            label_encoders[col] = le

    return X_processed, label_encoders


def get_top_features(feature_importance_df, top_n=8):
    """
    Get top N features based on importance.

    Parameters:
    -----------
    feature_importance_df : pandas.DataFrame
        DataFrame with feature importance
    top_n : int
        Number of top features to return

    Returns:
    --------
    top_features_list : list
        List of top feature names
    """
    top_n_features = min(top_n, len(feature_importance_df))
    top_features_list = feature_importance_df.head(top_n_features)["Feature"].tolist()
    return top_features_list
