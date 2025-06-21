import pandas as pd
import numpy as np
from natural_cycles_assignment.import_data import load_and_clean_data
from natural_cycles_assignment.plotting import (
    setup_plotting_style,
    plot_predicted_vs_actual,
    plot_feature_importance,
    plot_model_comparison,
)
from natural_cycles_assignment.utils import (
    preprocess_features,
    get_top_features,
)
from natural_cycles_assignment.model import (
    train_xgboost_model_with_cv,
    train_top_features_model,
    print_model_performance,
)
import matplotlib.pyplot as plt


def question_4_ML_approach_factors_impacting_conception_time(df):
    """Answer: What factors impact the time it takes to get pregnant?"""
    print("\n" + "=" * 60)
    print(
        "QUESTION 4: ML Approach: what factors impact the time it takes to get pregnant?"
    )
    print("=" * 60)

    # Only consider women who got pregnant for this analysis
    df_pregnant = df[df["pregnant"] == 1].copy()

    if len(df_pregnant) == 0:
        print("No pregnant women found in the dataset!")
        return None

    # Data preprocessing and feature engineering
    y = df_pregnant["n_cycles_trying"]
    X = df_pregnant.drop(columns=["pregnant", "n_cycles_trying", "country"])

    # Preprocess features
    X, label_encoders = preprocess_features(X)

    print(f"Feature matrix shape: {X.shape}")
    print(f"Target variable range: {y.min()} to {y.max()}")
    print(f"Target variable mean: {y.mean():.2f}")
    print(f"Target variable std: {y.std():.2f}")

    # Train main model with all features
    metrics, feature_importance_df, all_y_pred, all_y_test, best_params = (
        train_xgboost_model_with_cv(X, y)
    )

    # Print model performance
    avg_rmse, avg_r2 = print_model_performance(metrics, "XGBOOST")

    # Plot predicted vs actual
    plot_predicted_vs_actual(
        all_y_test,
        all_y_pred,
        title="XGBoost Regression (CV): Predicted vs Actual",
        save_path="reports/figures/q4_xgboost_pred_vs_actual.png",
        r2_score=avg_r2,
        model_params=best_params,
    )

    # Print and plot feature importance
    plot_feature_importance(
        feature_importance_df,
        title="Top 10 Feature Importance (XGBoost, CV)",
        save_path="reports/figures/q4_XGBoost_feature_importance.png",
    )

    return df_pregnant


def main():
    """Main analysis function"""
    print("NATURAL CYCLES PREGNANCY ANALYSIS")
    print("=" * 60)

    # Set up plotting style
    setup_plotting_style()

    csv_file = "data/external/ncdatachallenge-2021-v1.csv"

    # Load and clean data + remove na
    df = load_and_clean_data(csv_file=csv_file, clean_outliers=True, remove_na=True)

    # Answer question 4
    factors_analysis = question_4_ML_approach_factors_impacting_conception_time(df)

    # Keep all figures open until user closes them
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - Close figure windows to exit")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    main()
