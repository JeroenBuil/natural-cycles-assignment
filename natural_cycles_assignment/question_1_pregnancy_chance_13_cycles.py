import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from natural_cycles_assignment.import_data import load_and_clean_data

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def calculate_pregnancy_chance_within_13_cycles(df: pd.DataFrame):
    """Answer: What is the chance of getting pregnant within 13 cycles?"""
    print("\n" + "=" * 60)
    print("QUESTION 1: What is the chance of getting pregnant within 13 cycles?")
    print("=" * 60)

    # Calculate percentage of women who got pregnant in the dataset
    total_pregnant = df["pregnant"].sum()
    total_women = len(df)
    overall_pregnancy_rate = total_pregnant / total_women

    print(
        f"Overall pregnancy rate:\t\t{overall_pregnancy_rate*100:.1f}%\t({total_pregnant} pregnancie / {total_women} women)"
    )

    # Calculate pregnancy rate for women who tried for 13 cycles or less
    df_within_13_cycles = df[df["n_cycles_trying"] <= 13]
    pregnant_within_13 = df_within_13_cycles["pregnant"].sum()
    total_within_13 = len(df_within_13_cycles)

    if total_within_13 > 0:
        rate_within_13 = pregnant_within_13 / total_within_13
        print(
            f"Pregnancy rate <= 13 cycles:\t{rate_within_13*100:.1f}%\t({pregnant_within_13} pregnancies / {total_within_13} women\t"
        )

    # PLOTTING
    plt.figure(figsize=(12, 6))
    # Create histogram of cycles trying
    plt.subplot(1, 3, 1)
    sns.histplot(
        df, x="n_cycles_trying", alpha=0.7, discrete=True, kde=True, color="lightgreen"
    )
    plt.axvline(x=13, color="red", linestyle="--", label="13 cycles")
    plt.title("Histogram of Cycles needed to Conceive")
    plt.xlabel("Number of Cycles Trying")
    plt.ylabel("Count")
    plt.legend()

    # Cumulative histogram in percentage
    # BUG: the cumulative histogram is not correct, it should not total to 100%, because not all women got pregnant!
    plt.subplot(1, 3, 2)
    sns.histplot(
        df,
        x="n_cycles_trying",
        alpha=0.7,
        discrete=True,
        cumulative=True,
        stat="percent",
        thresh=90,
    )
    plt.axvline(x=13, color="red", linestyle="--", label="13 cycles")
    plt.axhline(
        y=overall_pregnancy_rate * 100,
        color="black",
        linestyle="--",
        label=f"{overall_pregnancy_rate*100:.1f}%",
    )
    plt.title("Cumulative Pregnancy Rate over Cycles needed to Conceive")
    plt.xlabel("Number of Cycles Trying")
    plt.ylabel("Pregnancy Rate [%]")
    plt.legend()

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

    # Add value labels on bars
    for i, rate in enumerate(rates):
        plt.text(i, rate + 0.01, f"{rate:.3f}", ha="center", va="bottom")

    plt.subplot(1, 3, 3)
    plt.bar(cycle_labels, rates, color="skyblue", alpha=0.7)
    plt.title("Pregnancy Rate by Cycle Range")
    plt.xlabel("Number of Cycles Trying")
    plt.ylabel("Pregnancy Rate")
    plt.xticks(rotation=45)

    # Save figure
    plt.tight_layout()
    plt.savefig(
        "reports/figures/pregnancy_chance_13_cycles.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return rate_within_13 if total_within_13 > 0 else None


def main():
    """Main analysis function"""
    print("Natural Cycles Assessment - Question 1 Analysis")
    print("=" * 60)

    csv_file = "data/external/ncdatachallenge-2021-v1.csv"

    # Load and clean data
    df = load_and_clean_data(csv_file=csv_file)

    # Answer question 1
    rate_13_cycles = calculate_pregnancy_chance_within_13_cycles(df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if rate_13_cycles is not None:
        print(f"1. Pregnancy chance within 13 cycles: {rate_13_cycles:.1%}")


if __name__ == "__main__":
    main()
