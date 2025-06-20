import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from natural_cycles_assignment.import_data import load_and_clean_data
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


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

    # XGBoost feature importance and selection
    X = df_pregnant.drop(columns=["pregnant", "n_cycles_trying", "country"])
    X = pd.get_dummies(X)
    y = df_pregnant["n_cycles_trying"]

    # Cross-validated XGBoost feature importance and performance
    n_splits = 5
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    feature_importances = np.zeros(X.shape[1])
    all_y_test = []
    all_y_pred = []
    metrics = []

    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model = XGBRegressor(objective="reg:squarederror", random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # Accumulate feature importances
        feature_importances += model.feature_importances_
        # Collect all predictions and true values
        all_y_test.extend(y_test)
        all_y_pred.extend(y_pred)
        # Collect metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        metrics.append((mse, rmse, mae, r2))

    # Average feature importances
    feature_importances /= n_splits
    feature_importance_df = pd.DataFrame(
        {"Feature": X.columns, "Importance": feature_importances}
    ).sort_values(by="Importance", ascending=False)

    # Average metrics
    avg_mse = np.mean([m[0] for m in metrics])
    avg_rmse = np.mean([m[1] for m in metrics])
    avg_mae = np.mean([m[2] for m in metrics])
    avg_r2 = np.mean([m[3] for m in metrics])

    print("\nCross-Validated Model Performance ({} folds):".format(n_splits))
    print(f"  Mean Squared Error (MSE): {avg_mse:.2f}")
    print(f"  Root Mean Squared Error (RMSE): {avg_rmse:.2f}")
    print(f"  Mean Absolute Error (MAE): {avg_mae:.2f}")
    print(f"  R² Score: {avg_r2:.2f}")

    # Plot predicted vs actual (all folds combined)
    all_y_test = np.array(all_y_test)
    all_y_pred = np.array(all_y_pred)
    plt.figure(figsize=(7, 7))
    plt.scatter(all_y_test, all_y_pred, alpha=0.6)
    plt.plot(
        [all_y_test.min(), all_y_test.max()],
        [all_y_test.min(), all_y_test.max()],
        "r--",
        lw=2,
    )
    plt.xlabel("Actual Cycles to Pregnancy")
    plt.ylabel("Predicted Cycles to Pregnancy")
    plt.title("XGBoost Regression (CV): Predicted vs Actual")
    plt.tight_layout()
    plt.savefig(
        "reports/figures/xgboost_pred_vs_actual.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    # Print and plot mean feature importance
    print("\nMean Feature Importance (across folds):")
    print(feature_importance_df)
    plt.figure(figsize=(10, 6))
    sns.barplot(
        data=feature_importance_df,
        y="Feature",
        x="Importance",
        palette="viridis",
        orient="h",
    )
    plt.title("Mean Feature Importance (XGBoost, CV)")
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(
        "reports/figures/XGBoost_feature_importance.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return df_pregnant


def main():
    """Main analysis function"""
    print("NATURAL CYCLES PREGNANCY ANALYSIS")
    print("=" * 60)

    csv_file = "data/external/ncdatachallenge-2021-v1.csv"

    # Load and clean data + remove na
    df = load_and_clean_data(csv_file=csv_file, remove_na=True)

    # Answer question 3
    factors_analysis = question_4_ML_approach_factors_impacting_conception_time(df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)


if __name__ == "__main__":
    main()
