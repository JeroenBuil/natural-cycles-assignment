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


def question_3_factors_impacting_conception_time(df):
    """Answer: What factors impact the time it takes to get pregnant?
    Args:
        df: pd.DataFrame, the cleaned dataset
    Returns:
        df_pregnant: pd.DataFrame, the dataset with only pregnant participants
    """
    print("\n" + "=" * 60)
    print("QUESTION 3: What factors impact the time it takes to get pregnant?")
    print("=" * 60)

    # Only consider women who got pregnant for this analysis
    df_pregnant = df[df["pregnant"] == 1].copy()

    if len(df_pregnant) == 0:
        print("No pregnant women found in the dataset!")
        return None

    # Analyze different factors
    factors_analysis = {}

    # 1. Age analysis
    print("\n1. AGE ANALYSIS:")
    age_bins = pd.IntervalIndex.from_tuples([(18, 25), (26, 30), (31, 35), (36, 50)])
    age_labels = ["18-25", "26-30", "31-35", "36-50"]

    df_pregnant["age_group"] = pd.cut(
        df_pregnant["age"], bins=age_bins, labels=age_labels, include_lowest=True
    )
    age_analysis = (
        df_pregnant.groupby("age_group")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(age_analysis)

    # Statistical test for age
    age_groups = df_pregnant["age_group"].dropna()
    cycles_by_age = df_pregnant["n_cycles_trying"].dropna()
    if len(age_groups) > 1:
        f_stat, p_value = stats.f_oneway(
            *[
                group["n_cycles_trying"].values
                for name, group in df_pregnant.groupby("age_group")
                if len(group) > 0
            ]
        )
        print(f"ANOVA p-value for age groups: {p_value:.4f}")

    # 2. BMI analysis
    print("\n2. BMI ANALYSIS:")
    bmi_bins = [0, 18.5, 24, 30, 50]
    bmi_labels = ["Underweight (<18.5)", "Normal (18.5-24)", "Overweight (24-30)", "Obese (>30)"]

    df_pregnant["bmi_group"] = pd.cut(
        df_pregnant["bmi"], bins=bmi_bins, labels=bmi_labels, include_lowest=True
    )
    bmi_analysis = (
        df_pregnant.groupby("bmi_group")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(bmi_analysis)

    # 3. Previous pregnancy analysis
    print("\n3. PREVIOUS PREGNANCY ANALYSIS:")
    prev_preg_analysis = (
        df_pregnant.groupby("been_pregnant_before")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(prev_preg_analysis)

    # Statistical test for previous pregnancy
    if len(df_pregnant["been_pregnant_before"].dropna().unique()) > 1:
        groups = [
            group["n_cycles_trying"].values
            for name, group in df_pregnant.groupby("been_pregnant_before")
            if len(group) > 0
        ]
        if len(groups) == 2:
            t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
            print(f"T-test p-value for previous pregnancy: {p_value:.4f}")

    # 4. Cycle regularity analysis
    print("\n4. CYCLE REGULARITY ANALYSIS:")
    cycle_analysis = (
        df_pregnant.groupby("regular_cycle")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(cycle_analysis)

    # 5. Dedication analysis
    print("\n5. DEDICATION ANALYSIS:")
    dedication_bins = [0, 0.5, 0.8, 1.0]
    dedication_labels = ["Low", "Medium", "High"]

    df_pregnant["dedication_group"] = pd.cut(
        df_pregnant["dedication"],
        bins=dedication_bins,
        labels=dedication_labels,
        include_lowest=True,
    )
    dedication_analysis = (
        df_pregnant.groupby("dedication_group")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(dedication_analysis)

    # 6. Intercourse frequency analysis
    print("\n6. INTERCOURSE FREQUENCY ANALYSIS:")
    freq_bins = [0, 0.05, 0.15, 0.3, 1.0]
    freq_labels = ["Very Low", "Low", "Medium", "High"]

    df_pregnant["freq_group"] = pd.cut(
        df_pregnant["intercourse_frequency"],
        bins=freq_bins,
        labels=freq_labels,
        include_lowest=True,
    )
    freq_analysis = (
        df_pregnant.groupby("freq_group")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(freq_analysis)

    # Create visualizations
    plt.figure(figsize=(12, 8))

    # Age vs cycles
    plt.subplot(2, 3, 1)
    age_means = df_pregnant.groupby("age_group")["n_cycles_trying"].mean()
    plt.bar(age_means.index, age_means.values, color="skyblue", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # BMI vs cycles
    plt.subplot(2, 3, 2)
    bmi_means = df_pregnant.groupby("bmi_group")["n_cycles_trying"].mean()
    plt.bar(bmi_means.index, bmi_means.values, color="lightcoral", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by BMI Group")
    plt.xlabel("BMI Group")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # Previous pregnancy vs cycles
    plt.subplot(2, 3, 3)
    prev_means = df_pregnant.groupby("been_pregnant_before")["n_cycles_trying"].mean()
    plt.bar(prev_means.index, prev_means.values, color="lightgreen", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Previous Pregnancy")
    plt.xlabel("Been Pregnant Before")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # Cycle regularity vs cycles
    plt.subplot(2, 3, 4)
    cycle_means = df_pregnant.groupby("regular_cycle")["n_cycles_trying"].mean()
    plt.bar(cycle_means.index, cycle_means.values, color="gold", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Cycle Regularity")
    plt.xlabel("Regular Cycle")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # Dedication vs cycles
    plt.subplot(2, 3, 5)
    ded_means = df_pregnant.groupby("dedication_group")["n_cycles_trying"].mean()
    plt.bar(ded_means.index, ded_means.values, color="plum", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Dedication")
    plt.xlabel("Dedication Level")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # Intercourse frequency vs cycles
    plt.subplot(2, 3, 6)
    freq_means = df_pregnant.groupby("freq_group")["n_cycles_trying"].mean()
    plt.bar(freq_means.index, freq_means.values, color="orange", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Intercourse Frequency")
    plt.xlabel("Intercourse Frequency")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("reports/figures/factors_impact.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Correlation analysis
    print("\n7. CORRELATION ANALYSIS:")
    numeric_cols = [
        "age",
        "bmi",
        "dedication",
        "average_cycle_length",
        "cycle_length_std",
        "intercourse_frequency",
    ]
    correlation_matrix = df_pregnant[numeric_cols + ["n_cycles_trying"]].corr()
    print("Correlation with cycles to pregnancy:")
    print(correlation_matrix["n_cycles_trying"].sort_values(ascending=False))

    return df_pregnant


def main():
    """Main analysis function"""
    print("NATURAL CYCLES PREGNANCY ANALYSIS")
    print("=" * 60)

    csv_file = "data/external/ncdatachallenge-2021-v1.csv"

    # Load and clean data + remove na
    df = load_and_clean_data(csv_file=csv_file, remove_na=True)

    # Answer question 3
    factors_analysis = question_3_factors_impacting_conception_time(df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    print("3. Key factors affecting time to pregnancy:")
    print("   - Age (older women may take longer)")
    print("   - Previous pregnancy history")
    print("   - BMI (both underweight and overweight)")
    print("   - Cycle regularity")
    print("   - App dedication and intercourse frequency")


if __name__ == "__main__":
    main()
