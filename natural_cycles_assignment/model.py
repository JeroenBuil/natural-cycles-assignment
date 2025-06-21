"""
Model training and evaluation functions for Natural Cycles assignment analysis.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor


def evaluate_model_cv(X, y, model, n_splits=5, random_state=42):
    """
    Evaluate a model using cross-validation.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    model : sklearn estimator
        Model to evaluate
    n_splits : int
        Number of CV folds
    random_state : int
        Random state for reproducibility

    Returns:
    --------
    metrics : list
        List of dictionaries with metrics for each fold
    feature_importances : numpy.ndarray
        Average feature importance across folds
    all_predictions : list
        All predictions from CV
    all_true_values : list
        All true values from CV
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    metrics = []
    feature_importances = np.zeros(X.shape[1])
    all_predictions = []
    all_true_values = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train model
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        metrics.append({"fold": fold + 1, "rmse": rmse, "r2": r2})

        # Accumulate feature importance (for models that support it)
        if hasattr(model, "coef_"):
            feature_importances += np.abs(model.coef_)
        elif hasattr(model, "feature_importances_"):
            feature_importances += model.feature_importances_

        all_predictions.extend(y_pred)
        all_true_values.extend(y_test)

    # Average feature importances
    feature_importances /= n_splits

    return metrics, feature_importances, all_predictions, all_true_values


def train_ridge_model_with_cv(X, y, param_grid=None, n_splits=5):
    """
    Train Ridge regression model with cross-validation and hyperparameter tuning.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    param_grid : dict, optional
        Hyperparameter grid for Ridge regression
    n_splits : int
        Number of CV folds

    Returns:
    --------
    metrics : list
        List of metric dictionaries
    feature_importance_df : pandas.DataFrame
        Feature importance DataFrame
    all_predictions : list
        All predictions from CV
    all_true_values : list
        All true values from CV
    """
    if param_grid is None:
        param_grid = {
            "alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_coefficients = np.zeros(X.shape[1])
    all_y_test = []
    all_y_pred = []
    metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Scale features for linear models
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Hyperparameter tuning for this fold
        base_model = Ridge(random_state=42)

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train_scaled, y_train)
        best_model = grid_search.best_estimator_

        print(f"Best alpha: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.2f}")

        y_pred = best_model.predict(X_test_scaled)

        # Accumulate feature coefficients (absolute values for importance)
        feature_coefficients += np.abs(best_model.coef_)
        # Collect all predictions and true values
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        # Collect metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Fold {fold + 1} R²: {r2:.3f}, RMSE: {rmse:.3f}")

        metrics.append(
            {
                "rmse": rmse,
                "r2": r2,
            }
        )

    # Average feature coefficients
    feature_coefficients /= n_splits
    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_coefficients}
    ).sort_values(by="Importance", ascending=False)

    return metrics, feature_importance_df, all_y_pred, all_y_test


def train_xgboost_model_with_cv(X, y, param_grid=None, n_splits=5):
    """
    Train XGBoost model with cross-validation and hyperparameter tuning.

    Parameters:
    -----------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target variable
    param_grid : dict, optional
        Hyperparameter grid for XGBoost
    n_splits : int
        Number of CV folds

    Returns:
    --------
    metrics : list
        List of metric dictionaries
    feature_importance_df : pandas.DataFrame
        Feature importance DataFrame
    all_predictions : list
        All predictions from CV
    all_true_values : list
        All true values from CV
    """
    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1, 0.2],
            "subsample": [0.8, 0.9, 1.0],
            "colsample_bytree": [0.8, 0.9, 1.0],
            "reg_alpha": [0, 0.1, 0.5],
            "reg_lambda": [0, 0.1, 0.5],
        }

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_importances = np.zeros(X.shape[1])
    all_y_test = []
    all_y_pred = []
    metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\nFold {fold + 1}/{n_splits}")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Hyperparameter tuning for this fold
        base_model = XGBRegressor(
            objective="reg:squarederror", random_state=42, eval_metric="rmse"
        )

        # Use GridSearchCV for hyperparameter tuning
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_

        print(f"Best parameters: {grid_search.best_params_}")
        print(f"Best CV score: {-grid_search.best_score_:.2f}")

        y_pred = best_model.predict(X_test)

        # Accumulate feature importances
        feature_importances += best_model.feature_importances_
        # Collect all predictions and true values
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        # Collect metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        print(f"Fold {fold + 1} R²: {r2:.3f}, RMSE: {rmse:.3f}")

        metrics.append(
            {
                "rmse": rmse,
                "r2": r2,
            }
        )

    # Average feature importances
    feature_importances /= n_splits
    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    return metrics, feature_importance_df, all_y_pred, all_y_test


def train_top_features_model(X, y, top_features_list, model_type="ridge", n_splits=5):
    """
    Train model using only top features.

    Parameters:
    -----------
    X : pandas.DataFrame
        Full feature matrix
    y : pandas.Series
        Target variable
    top_features_list : list
        List of top feature names to use
    model_type : str
        Type of model to train ("ridge" or "xgboost")
    n_splits : int
        Number of CV folds

    Returns:
    --------
    metrics : list
        List of metric dictionaries
    all_predictions : list
        All predictions from CV
    all_true_values : list
        All true values from CV
    feature_importance_df : pandas.DataFrame, optional
        Feature importance DataFrame (only for XGBoost models)
    """
    # Create new feature matrix with only top features
    X_top = X[top_features_list].copy()

    # Cross-validated model with top features only
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    all_y_test_top = []
    all_y_pred_top = []
    metrics_top = []

    # For feature importance tracking (XGBoost only)
    if model_type == "xgboost":
        feature_importances = np.zeros(len(top_features_list))

    for fold, (train_idx, test_idx) in enumerate(kf.split(X_top)):
        print(f"\nFold {fold + 1}/{n_splits} (Top Features Model)")
        X_train_top, X_test_top = X_top.iloc[train_idx], X_top.iloc[test_idx]
        y_train_top, y_test_top = y.iloc[train_idx], y.iloc[test_idx]

        if model_type == "ridge":
            # Scale features for linear models
            scaler_top = StandardScaler()
            X_train_top_scaled = scaler_top.fit_transform(X_train_top)
            X_test_top_scaled = scaler_top.transform(X_test_top)

            # Hyperparameter tuning for this fold
            base_model_top = Ridge(random_state=42)

            # Parameter grid for Ridge regression
            param_grid_top = {
                "alpha": [0.001, 0.01, 0.1, 1.0, 10.0],
            }

            X_train_final, X_test_final = X_train_top_scaled, X_test_top_scaled

        elif model_type == "xgboost":
            # Hyperparameter tuning for this fold (simplified grid for faster training)
            base_model_top = XGBRegressor(
                objective="reg:squarederror", random_state=42, eval_metric="rmse"
            )

            # Simplified parameter grid for faster training
            param_grid_top = {
                "n_estimators": [100, 200],
                "max_depth": [3, 5],
                "learning_rate": [0.1, 0.2],
                "subsample": [0.8, 1.0],
                "colsample_bytree": [0.8, 1.0],
            }

            X_train_final, X_test_final = X_train_top, X_test_top

        grid_search_top = GridSearchCV(
            base_model_top,
            param_grid_top,
            cv=3,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )

        grid_search_top.fit(X_train_final, y_train_top)
        best_model_top = grid_search_top.best_estimator_

        if model_type == "ridge":
            print(f"Best alpha: {grid_search_top.best_params_}")
        else:
            print(f"Best parameters: {grid_search_top.best_params_}")
        print(f"Best CV score: {-grid_search_top.best_score_:.2f}")

        y_pred_top = best_model_top.predict(X_test_final)

        # Collect all predictions and true values
        all_y_test_top.extend(y_test_top)
        all_y_pred_top.extend(y_pred_top)

        # Accumulate feature importances for XGBoost
        if model_type == "xgboost":
            feature_importances += best_model_top.feature_importances_

        # Collect metrics
        rmse_top = np.sqrt(mean_squared_error(y_test_top, y_pred_top))
        r2_top = r2_score(y_test_top, y_pred_top)

        print(f"Fold {fold + 1} R²: {r2_top:.3f}, RMSE: {rmse_top:.3f}")

        metrics_top.append(
            {
                "rmse": rmse_top,
                "r2": r2_top,
            }
        )

    # Create feature importance DataFrame for XGBoost models
    if model_type == "xgboost":
        feature_importances /= n_splits
        feature_importance_df = pd.DataFrame(
            {"Feature": top_features_list, "Importance": feature_importances}
        ).sort_values(by="Importance", ascending=False)
        return metrics_top, all_y_pred_top, all_y_test_top, feature_importance_df
    else:
        return metrics_top, all_y_pred_top, all_y_test_top, None


def print_model_performance(metrics, model_name="Model"):
    """
    Print model performance metrics.

    Parameters:
    -----------
    metrics : list
        List of metric dictionaries from cross-validation
    model_name : str
        Name of the model for display
    """
    avg_rmse = np.mean([m["rmse"] for m in metrics])
    avg_r2 = np.mean([m["r2"] for m in metrics])

    print(f"\n{model_name} PERFORMANCE")
    print("=" * 60)
    print(f"Cross-Validated Model Performance ({len(metrics)} folds):")
    print(f"  Root Mean Squared Error (RMSE): {avg_rmse:.2f}")
    print(f"  R² Score: {avg_r2:.3f}")

    return avg_rmse, avg_r2
