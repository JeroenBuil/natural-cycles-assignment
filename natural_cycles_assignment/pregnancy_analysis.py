import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from natural_cycles_assignment.import_data import load_and_clean_data

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def question_1_pregnancy_chance_13_cycles(df):
    """Answer: What is the chance of getting pregnant within 13 cycles?"""
    print("\n" + "=" * 60)
    print("QUESTION 1: What is the chance of getting pregnant within 13 cycles?")
    print("=" * 60)

    # Calculate overall pregnancy rate
    total_pregnant = df["pregnant"].sum()
    total_women = len(df)
    overall_rate = total_pregnant / total_women

    print(f"Overall pregnancy rate: {overall_rate:.3f} ({overall_rate*100:.1f}%)")

    # Calculate pregnancy rate for women who tried for 13 cycles or less
    within_13_cycles = df[df["n_cycles_trying"] <= 13]
    pregnant_within_13 = within_13_cycles["pregnant"].sum()
    total_within_13 = len(within_13_cycles)

    if total_within_13 > 0:
        rate_within_13 = pregnant_within_13 / total_within_13
        print(
            f"Pregnancy rate within 13 cycles: {rate_within_13:.3f} ({rate_within_13*100:.1f}%)"
        )
        print(f"Number of women who tried for ≤13 cycles: {total_within_13}")
        print(f"Number of pregnancies within 13 cycles: {pregnant_within_13}")

    # Create a plot showing pregnancy rate by cycle range
    cycle_ranges = [(1, 6), (7, 12), (13, 18), (19, 24), (25, 30), (31, 50)]
    rates = []
    cycle_labels = []

    for start, end in cycle_ranges:
        mask = (df["n_cycles_trying"] >= start) & (df["n_cycles_trying"] <= end)
        subset = df[mask]
        if len(subset) > 0:
            rate = subset["pregnant"].mean()
            rates.append(rate)
            cycle_labels.append(f"{start}-{end}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.bar(cycle_labels, rates, color="skyblue", alpha=0.7)
    plt.title("Pregnancy Rate by Cycle Range")
    plt.xlabel("Number of Cycles Trying")
    plt.ylabel("Pregnancy Rate")
    plt.xticks(rotation=45)

    # Add value labels on bars
    for i, rate in enumerate(rates):
        plt.text(i, rate + 0.01, f"{rate:.3f}", ha="center", va="bottom")

    plt.subplot(1, 2, 2)
    # Distribution of cycles trying
    plt.hist(df["n_cycles_trying"], bins=20, alpha=0.7, color="lightcoral")
    plt.axvline(x=13, color="red", linestyle="--", label="13 cycles")
    plt.title("Distribution of Cycles Trying to Conceive")
    plt.xlabel("Number of Cycles")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()
    plt.savefig(
        "reports/figures/pregnancy_chance_13_cycles.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return rate_within_13 if total_within_13 > 0 else None


def question_2_time_to_pregnancy(df):
    """Answer: How long does it usually take to get pregnant?"""
    print("\n" + "=" * 60)
    print("QUESTION 2: How long does it usually take to get pregnant?")
    print("=" * 60)

    # Only consider women who got pregnant
    pregnant_women = df[df["pregnant"] == 1]

    if len(pregnant_women) == 0:
        print("No pregnant women found in the dataset!")
        return None

    # Calculate statistics for time to pregnancy
    cycles_to_pregnancy = pregnant_women["n_cycles_trying"]

    print(f"Number of women who got pregnant: {len(pregnant_women)}")
    print(f"Mean cycles to pregnancy: {cycles_to_pregnancy.mean():.2f}")
    print(f"Median cycles to pregnancy: {cycles_to_pregnancy.median():.2f}")
    print(f"Standard deviation: {cycles_to_pregnancy.std():.2f}")
    print(f"Min cycles: {cycles_to_pregnancy.min()}")
    print(f"Max cycles: {cycles_to_pregnancy.max()}")

    # Percentiles
    percentiles = [25, 50, 75, 90, 95]
    for p in percentiles:
        value = np.percentile(cycles_to_pregnancy, p)
        print(f"{p}th percentile: {value:.1f} cycles")

    # Create visualizations
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.hist(cycles_to_pregnancy, bins=20, alpha=0.7, color="lightgreen")
    plt.axvline(
        cycles_to_pregnancy.mean(),
        color="red",
        linestyle="--",
        label=f"Mean: {cycles_to_pregnancy.mean():.1f}",
    )
    plt.axvline(
        cycles_to_pregnancy.median(),
        color="blue",
        linestyle="--",
        label=f"Median: {cycles_to_pregnancy.median():.1f}",
    )
    plt.title("Distribution of Cycles to Pregnancy")
    plt.xlabel("Cycles to Pregnancy")
    plt.ylabel("Frequency")
    plt.legend()

    plt.subplot(1, 3, 2)
    # Cumulative distribution
    sorted_cycles = np.sort(cycles_to_pregnancy)
    cumulative_prob = np.arange(1, len(sorted_cycles) + 1) / len(sorted_cycles)
    plt.plot(sorted_cycles, cumulative_prob, linewidth=2)
    plt.axhline(y=0.5, color="red", linestyle="--", alpha=0.7, label="50%")
    plt.axhline(y=0.75, color="blue", linestyle="--", alpha=0.7, label="75%")
    plt.title("Cumulative Distribution of Time to Pregnancy")
    plt.xlabel("Cycles to Pregnancy")
    plt.ylabel("Cumulative Probability")
    plt.legend()

    plt.subplot(1, 3, 3)
    # Box plot
    plt.boxplot(
        cycles_to_pregnancy, patch_artist=True, boxprops=dict(facecolor="lightblue")
    )
    plt.title("Box Plot of Cycles to Pregnancy")
    plt.ylabel("Cycles")

    plt.tight_layout()
    plt.savefig("reports/figures/time_to_pregnancy.png", dpi=300, bbox_inches="tight")
    plt.show()

    return cycles_to_pregnancy


def question_3_factors_impact(df):
    """Answer: What factors impact the time it takes to get pregnant?"""
    print("\n" + "=" * 60)
    print("QUESTION 3: What factors impact the time it takes to get pregnant?")
    print("=" * 60)

    # Only consider women who got pregnant for this analysis
    pregnant_women = df[df["pregnant"] == 1].copy()

    if len(pregnant_women) == 0:
        print("No pregnant women found in the dataset!")
        return None

    # Analyze different factors
    factors_analysis = {}

    # 1. Age analysis
    print("\n1. AGE ANALYSIS:")
    age_bins = [18, 25, 30, 35, 40, 45]
    age_labels = ["18-25", "26-30", "31-35", "36-40", "41-45"]

    pregnant_women["age_group"] = pd.cut(
        pregnant_women["age"], bins=age_bins, labels=age_labels, include_lowest=True
    )
    age_analysis = (
        pregnant_women.groupby("age_group")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(age_analysis)

    # Statistical test for age
    age_groups = pregnant_women["age_group"].dropna()
    cycles_by_age = pregnant_women["n_cycles_trying"].dropna()
    if len(age_groups) > 1:
        f_stat, p_value = stats.f_oneway(
            *[
                group["n_cycles_trying"].values
                for name, group in pregnant_women.groupby("age_group")
                if len(group) > 0
            ]
        )
        print(f"ANOVA p-value for age groups: {p_value:.4f}")

    # 2. BMI analysis
    print("\n2. BMI ANALYSIS:")
    bmi_bins = [0, 18.5, 25, 30, 50]
    bmi_labels = ["Underweight", "Normal", "Overweight", "Obese"]

    pregnant_women["bmi_group"] = pd.cut(
        pregnant_women["bmi"], bins=bmi_bins, labels=bmi_labels, include_lowest=True
    )
    bmi_analysis = (
        pregnant_women.groupby("bmi_group")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(bmi_analysis)

    # 3. Previous pregnancy analysis
    print("\n3. PREVIOUS PREGNANCY ANALYSIS:")
    prev_preg_analysis = (
        pregnant_women.groupby("been_pregnant_before")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(prev_preg_analysis)

    # Statistical test for previous pregnancy
    if len(pregnant_women["been_pregnant_before"].dropna().unique()) > 1:
        groups = [
            group["n_cycles_trying"].values
            for name, group in pregnant_women.groupby("been_pregnant_before")
            if len(group) > 0
        ]
        if len(groups) == 2:
            t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
            print(f"T-test p-value for previous pregnancy: {p_value:.4f}")

    # 4. Cycle regularity analysis
    print("\n4. CYCLE REGULARITY ANALYSIS:")
    cycle_analysis = (
        pregnant_women.groupby("regular_cycle")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(cycle_analysis)

    # 5. Dedication analysis
    print("\n5. DEDICATION ANALYSIS:")
    dedication_bins = [0, 0.5, 0.8, 1.0]
    dedication_labels = ["Low", "Medium", "High"]

    pregnant_women["dedication_group"] = pd.cut(
        pregnant_women["dedication"],
        bins=dedication_bins,
        labels=dedication_labels,
        include_lowest=True,
    )
    dedication_analysis = (
        pregnant_women.groupby("dedication_group")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(dedication_analysis)

    # 6. Intercourse frequency analysis
    print("\n6. INTERCOURSE FREQUENCY ANALYSIS:")
    freq_bins = [0, 0.05, 0.15, 0.3, 1.0]
    freq_labels = ["Very Low", "Low", "Medium", "High"]

    pregnant_women["freq_group"] = pd.cut(
        pregnant_women["intercourse_frequency"],
        bins=freq_bins,
        labels=freq_labels,
        include_lowest=True,
    )
    freq_analysis = (
        pregnant_women.groupby("freq_group")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(freq_analysis)

    # Create visualizations
    plt.figure(figsize=(20, 12))

    # Age vs cycles
    plt.subplot(2, 3, 1)
    age_means = pregnant_women.groupby("age_group")["n_cycles_trying"].mean()
    plt.bar(age_means.index, age_means.values, color="skyblue", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Age Group")
    plt.xlabel("Age Group")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # BMI vs cycles
    plt.subplot(2, 3, 2)
    bmi_means = pregnant_women.groupby("bmi_group")["n_cycles_trying"].mean()
    plt.bar(bmi_means.index, bmi_means.values, color="lightcoral", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by BMI Group")
    plt.xlabel("BMI Group")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # Previous pregnancy vs cycles
    plt.subplot(2, 3, 3)
    prev_means = pregnant_women.groupby("been_pregnant_before")[
        "n_cycles_trying"
    ].mean()
    plt.bar(prev_means.index, prev_means.values, color="lightgreen", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Previous Pregnancy")
    plt.xlabel("Been Pregnant Before")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # Cycle regularity vs cycles
    plt.subplot(2, 3, 4)
    cycle_means = pregnant_women.groupby("regular_cycle")["n_cycles_trying"].mean()
    plt.bar(cycle_means.index, cycle_means.values, color="gold", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Cycle Regularity")
    plt.xlabel("Regular Cycle")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # Dedication vs cycles
    plt.subplot(2, 3, 5)
    ded_means = pregnant_women.groupby("dedication_group")["n_cycles_trying"].mean()
    plt.bar(ded_means.index, ded_means.values, color="plum", alpha=0.7)
    plt.title("Mean Cycles to Pregnancy by Dedication")
    plt.xlabel("Dedication Level")
    plt.ylabel("Mean Cycles")
    plt.xticks(rotation=45)

    # Intercourse frequency vs cycles
    plt.subplot(2, 3, 6)
    freq_means = pregnant_women.groupby("freq_group")["n_cycles_trying"].mean()
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
    correlation_matrix = pregnant_women[numeric_cols + ["n_cycles_trying"]].corr()
    print("Correlation with cycles to pregnancy:")
    print(correlation_matrix["n_cycles_trying"].sort_values(ascending=False))

    return pregnant_women


def main():
    """Main analysis function"""
    print("NATURAL CYCLES PREGNANCY ANALYSIS")
    print("=" * 60)

    # Load and clean data
    df = load_and_clean_data()

    # Answer the three questions
    rate_13_cycles = question_1_pregnancy_chance_13_cycles(df)
    cycles_to_pregnancy = question_2_time_to_pregnancy(df)
    factors_analysis = question_3_factors_impact(df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    if rate_13_cycles is not None:
        print(f"1. Pregnancy chance within 13 cycles: {rate_13_cycles:.1%}")

    if cycles_to_pregnancy is not None:
        print(f"2. Median time to pregnancy: {cycles_to_pregnancy.median():.1f} cycles")
        print(f"   Mean time to pregnancy: {cycles_to_pregnancy.mean():.1f} cycles")

    print("3. Key factors affecting time to pregnancy:")
    print("   - Age (older women may take longer)")
    print("   - Previous pregnancy history")
    print("   - BMI (both underweight and overweight)")
    print("   - Cycle regularity")
    print("   - App dedication and intercourse frequency")


if __name__ == "__main__":
    main()
