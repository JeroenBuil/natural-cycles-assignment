import pandas as pd
import numpy as np
from natural_cycles_assignment.import_data import import_nc_data


def analyze_pregnancy_data():
    """Analyze the pregnancy data to answer the three questions"""

    # Load data
    df = import_nc_data()

    # Clean data
    df = df.dropna(subset=["outcome", "n_cycles_trying"])
    df["pregnant"] = (df["outcome"] == "pregnant").astype(int)
    df = df[df["bmi"] > 10]  # Remove very low BMI values
    df = df[df["age"] >= 18]  # Remove very young ages

    print(f"Cleaned dataset: {len(df)} women")

    # Question 1: Pregnancy chance within 13 cycles
    print("\n" + "=" * 60)
    print("QUESTION 1: What is the chance of getting pregnant within 13 cycles?")
    print("=" * 60)

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

    # Question 2: Time to pregnancy
    print("\n" + "=" * 60)
    print("QUESTION 2: How long does it usually take to get pregnant?")
    print("=" * 60)

    pregnant_women = df[df["pregnant"] == 1]
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

    # Question 3: Factors impacting time to pregnancy
    print("\n" + "=" * 60)
    print("QUESTION 3: What factors impact the time it takes to get pregnant?")
    print("=" * 60)

    # Age analysis
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

    # BMI analysis
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

    # Previous pregnancy analysis
    print("\n3. PREVIOUS PREGNANCY ANALYSIS:")
    prev_preg_analysis = (
        pregnant_women.groupby("been_pregnant_before")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(prev_preg_analysis)

    # Cycle regularity analysis
    print("\n4. CYCLE REGULARITY ANALYSIS:")
    cycle_analysis = (
        pregnant_women.groupby("regular_cycle")["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(cycle_analysis)

    # Dedication analysis
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

    # Intercourse frequency analysis
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
    correlations = correlation_matrix["n_cycles_trying"].sort_values(ascending=False)
    for var, corr in correlations.items():
        if var != "n_cycles_trying":
            print(f"  {var}: {corr:.3f}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    print(f"1. Pregnancy chance within 13 cycles: {rate_within_13:.1%}")
    print(f"2. Median time to pregnancy: {cycles_to_pregnancy.median():.1f} cycles")
    print(f"   Mean time to pregnancy: {cycles_to_pregnancy.mean():.1f} cycles")
    print("3. Key factors affecting time to pregnancy:")
    print("   - Age (older women may take longer)")
    print("   - Previous pregnancy history")
    print("   - BMI (both underweight and overweight)")
    print("   - Cycle regularity")
    print("   - App dedication and intercourse frequency")

    return {
        "pregnancy_rate_13_cycles": rate_within_13,
        "median_cycles_to_pregnancy": cycles_to_pregnancy.median(),
        "mean_cycles_to_pregnancy": cycles_to_pregnancy.mean(),
        "correlations": correlations,
    }


if __name__ == "__main__":
    results = analyze_pregnancy_data()
