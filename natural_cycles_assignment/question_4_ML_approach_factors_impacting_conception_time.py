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
    metrics, feature_importance_df, all_y_pred, all_y_test = (
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
    )

    # Print and plot feature importance
    plot_feature_importance(
        feature_importance_df,
        title="Top 10 Feature Importance (XGBoost, CV)",
        save_path="reports/figures/q4_XGBoost_feature_importance.png",
    )

    # Second model with only top features
    print("\n" + "=" * 60)
    print("SECOND MODEL: Using Only Top Features")
    print("=" * 60)

    # Get top features
    top_features_list = get_top_features(feature_importance_df, top_n=5)

    print(f"Selected top {len(top_features_list)} features:")
    for i, feature in enumerate(top_features_list, 1):
        importance = feature_importance_df[feature_importance_df["Feature"] == feature][
            "Importance"
        ].iloc[0]
        print(f"  {i}. {feature}: {importance:.4f}")

    # Train model with top features
    metrics_top, all_y_pred_top, all_y_test_top, feature_importance_df_top = (
        train_top_features_model(X, y, top_features_list, model_type="xgboost")
    )

    # Print performance for top features model
    avg_rmse_top, avg_r2_top = print_model_performance(
        metrics_top, "TOP FEATURES XGBOOST"
    )

    # Plot predicted vs actual for top features model
    plot_predicted_vs_actual(
        all_y_test_top,
        all_y_pred_top,
        title="Top Features XGBoost Regression (CV): Predicted vs Actual",
        save_path="reports/figures/q4_top_features_xgboost_pred_vs_actual.png",
        r2_score=avg_r2_top,
    )

    # Plot feature importance for top features model
    if feature_importance_df_top is not None:
        plot_feature_importance(
            feature_importance_df_top,
            title="Top Features XGBoost Feature Importance (CV)",
            save_path="reports/figures/q4_top_features_xgboost_feature_importance.png",
        )

    # Print comparison
    print("\n" + "=" * 60)
    print("COMPARISON: All Features vs Top Features")
    print("=" * 60)
    print("All Features Model:")
    print(f"  R² Score: {avg_r2:.3f}")
    print(f"  RMSE: {avg_rmse:.3f}")
    print("\nTop Features Model:")
    print(f"  R² Score: {avg_r2_top:.3f}")
    print(f"  RMSE: {avg_rmse_top:.3f}")

    # Calculate improvement
    r2_improvement = avg_r2_top - avg_r2
    rmse_improvement = avg_rmse - avg_rmse_top  # Lower RMSE is better

    print(f"\nImprovement:")
    print(f"  R² improvement: {r2_improvement:+.3f}")
    print(f"  RMSE improvement: {rmse_improvement:+.3f}")

    # Plot comparison
    metrics_dict = {
        "All Features": [avg_r2, avg_rmse],
        "Top Features": [avg_r2_top, avg_rmse_top],
    }
    plot_model_comparison(
        metrics_dict,
        title="Model Comparison",
        save_path="reports/figures/q4_model_comparison.png",
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
