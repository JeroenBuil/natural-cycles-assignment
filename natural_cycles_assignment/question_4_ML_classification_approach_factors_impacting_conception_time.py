import pandas as pd
import numpy as np
from natural_cycles_assignment.import_data import load_and_clean_data
from natural_cycles_assignment.plotting import (
    setup_plotting_style,
    plot_classification_heatmap,
    plot_feature_importance,
    plot_model_comparison,
)
from natural_cycles_assignment.utils import (
    preprocess_features,
)
from natural_cycles_assignment.model import (
    train_xgboost_classification_with_cv,
    print_classification_performance,
)
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE
from collections import Counter
import seaborn as sns


def question_4_ML_classification_approach_factors_impacting_conception_time(df):
    """Answer: What factors impact the time it takes to get pregnant? (Classification approach)"""
    print("\n" + "=" * 60)
    print(
        "QUESTION 4: ML Classification Approach: what factors impact the time it takes to get pregnant?"
    )
    print("=" * 60)

    # Only consider women who got pregnant for this analysis
    df_pregnant = df[df["pregnant"] == 1].copy()

    if len(df_pregnant) == 0:
        print("No pregnant women found in the dataset!")
        return None

    # Create binned target variable
    def bin_cycles(cycles):
        if cycles <= 3:
            return 0  # Fast conception (0-3 cycles)
        elif cycles <= 6:
            return 1  # Medium conception (4-6 cycles)
        else:
            return 2  # Slow conception (7+ cycles)

    df_pregnant["conception_speed"] = df_pregnant["n_cycles_trying"].apply(bin_cycles)

    # Print class distribution
    class_counts = df_pregnant["conception_speed"].value_counts().sort_index()
    class_names = ["Fast (0-3 cycles)", "Medium (4-6 cycles)", "Slow (7+ cycles)"]
    print("\nClass Distribution:")
    for i, (class_idx, count) in enumerate(class_counts.items()):
        percentage = (count / len(df_pregnant)) * 100
        print(f"  {class_names[class_idx]}: {count} samples ({percentage:.1f}%)")

    # Data preprocessing and feature engineering
    y = df_pregnant["conception_speed"]
    X = df_pregnant.drop(
        columns=["pregnant", "n_cycles_trying", "conception_speed", "country"]
    )

    # Preprocess features
    X, label_encoders = preprocess_features(X)

    print(f"\nFeature matrix shape: {X.shape}")
    print(f"Target variable classes: {y.unique()}")
    print(f"Target variable distribution: {y.value_counts().sort_index().tolist()}")

    # Balance classes using SMOTE
    print("\n" + "=" * 60)
    print("BALANCING CLASSES USING SMOTE")
    print("=" * 60)

    # Apply SMOTE to balance classes
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_balanced, y_balanced = smote.fit_resample(X, y)

    print(f"Original class distribution: {Counter(y)}")
    print(f"Balanced class distribution: {Counter(y_balanced)}")
    print(f"Balanced feature matrix shape: {X_balanced.shape}")

    # Calculate samples per class after balancing
    balanced_counts = Counter(y_balanced)
    total_samples = len(y_balanced)
    print("\nBalanced Class Distribution:")
    for class_idx, count in sorted(balanced_counts.items()):
        percentage = (count / total_samples) * 100
        print(f"  {class_names[class_idx]}: {count} samples ({percentage:.1f}%)")

    # Plot class distribution before and after SMOTE
    plt.figure(figsize=(4, 3))
    plt.subplot(1, 2, 1)
    plt.bar(
        ["Fast", "Medium", "Slow"],
        [sum(y == 0), sum(y == 1), sum(y == 2)],
        color=["green", "orange", "red"],
        alpha=0.7,
    )
    plt.title("Class Distribution (Before SMOTE)")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    plt.bar(
        ["Fast", "Medium", "Slow"],
        [sum(y_balanced == 0), sum(y_balanced == 1), sum(y_balanced == 2)],
        color=["green", "orange", "red"],
        alpha=0.7,
    )
    plt.title("Class Distribution (After SMOTE)")
    plt.ylabel("Count")

    plt.tight_layout()
    plt.savefig(
        "reports/figures/q4_ML_classification_class_distribution_before_after_smote.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show(block=False)

    # Train main model with all features (using balanced data)
    metrics, feature_importance_df, all_y_pred, all_y_test, best_params = (
        train_xgboost_classification_with_cv(X_balanced, y_balanced)
    )

    # Print model performance
    avg_accuracy, avg_f1 = print_classification_performance(
        metrics, "XGBOOST CLASSIFICATION"
    )

    # Define class names for the heatmap
    class_names = ["Fast (0-3 cycles)", "Medium (4-6 cycles)", "Slow (7+ cycles)"]

    # Plot classification heatmap (for classification, we'll show confusion matrix)
    plot_classification_heatmap(
        all_y_test,
        all_y_pred,
        title="XGBoost Classification (CV): Classification Results",
        save_path="reports/figures/q4_ML_classification_xgboost_heatmap.png",
        model_params=best_params,
        class_names=class_names,
        accuracy=avg_accuracy,
        f1_score=avg_f1,
    )

    # Print and plot feature importance
    plot_feature_importance(
        feature_importance_df,
        title="Top 10 Feature Importance (XGBoost Classification, CV)",
        save_path="reports/figures/q4_ML_classification_xgboost_feature_importance.png",
    )

    # Print final results
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print("XGBoost Classification Model (All Features, Balanced Classes):")
    print(f"  Accuracy: {avg_accuracy:.3f}")
    print(f"  F1 Score: {avg_f1:.3f}")

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
    factors_analysis = (
        question_4_ML_classification_approach_factors_impacting_conception_time(df)
    )

    # Keep all figures open until user closes them
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - Close figure windows to exit")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    main()
