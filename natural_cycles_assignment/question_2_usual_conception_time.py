import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from natural_cycles_assignment.import_data import load_and_clean_data

# Set up plotting style
plt.style.use("seaborn-v0_8")
sns.set_palette("husl")


def calculate_usual_conception_time(df: pd.DataFrame):
    """Answer: How long does it usually take to get pregnant?"""
    print("\n" + "=" * 60)
    print("QUESTION 2: How long does it usually take to get pregnant?")
    print("=" * 60)

    # Only women who got pregnant are relevant for this question
    df_pregnant = df[df["pregnant"] == 1]

    if len(df_pregnant) == 0:
        print("No pregnant women found in the dataset!")
        return None

    # Calculate relevant statistics for conception time
    count_women_pregnant = len(df_pregnant)
    mean_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].mean()
    median_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].median()
    std_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].std()
    min_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].min()
    max_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].max()

    print(f"Nr of women who got pregnant: {count_women_pregnant}")
    print(f"Mean conception time: {mean_cycles_to_pregnancy:.2f} cycles")
    print(f"Median conception time: {median_cycles_to_pregnancy:.2f} cycles")
    print(f"Standard deviation: {std_cycles_to_pregnancy:.2f} cycles")
    print(f"Min conception time: {min_cycles_to_pregnancy} cycles")
    print(f"Max conception time: {max_cycles_to_pregnancy} cycles")

    # Percentiles
    # percentiles = [25, 50, 75, 90, 95]
    # for p in percentiles:
    #     value = np.percentile(cycles_to_pregnancy, p)
    #     print(f"{p}th percentile: {value:.1f} cycles")

    # Create visualizations
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.histplot(
        df_pregnant,
        x="n_cycles_trying",
        bins=max_cycles_to_pregnancy + 1,
        alpha=0.7,
        color="lightgreen",
        kde=True,
        discrete=True,
    )
    plt.axvline(
        mean_cycles_to_pregnancy,
        color="black",
        linestyle="--",
        label=f"Mean: {mean_cycles_to_pregnancy:.1f}",
    )
    plt.axvline(
        median_cycles_to_pregnancy,
        color="darkgreen",
        linestyle="--",
        label=f"Median: {median_cycles_to_pregnancy:.1f}",
    )
    plt.title("Histogram of conception time")
    plt.xlabel("Conception time [cycles]")
    plt.ylabel("Count")
    plt.legend()

    plt.subplot(1, 3, 2)
    # Cumulative distribution
    # sorted_cycles = np.sort(cycles_to_pregnancy)
    # cumulative_prob = np.arange(1, len(sorted_cycles) + 1) / len(sorted_cycles)
    # plt.plot(sorted_cycles, cumulative_prob, linewidth=2)
    sns.histplot(
        df_pregnant,
        x="n_cycles_trying",
        alpha=0.7,
        bins=max_cycles_to_pregnancy + 1,
        discrete=True,
        cumulative=True,
        stat="percent",
        thresh=90,
    )
    # Plot horizontal lines for 50%, 75% and 90%
    plt.axhline(y=90, color="red", linestyle="--", alpha=0.7, label="90%")
    plt.axhline(y=75, color="darkred", linestyle="--", alpha=0.7, label="75%")
    plt.axhline(y=50, color="black", linestyle="--", alpha=0.7, label="50%")
    plt.title("Cumulative Histogram of conception time")
    plt.xlabel("Conception time [cycles]")
    plt.ylabel("Cumulative Probability [%]")
    plt.legend()

    plt.subplot(1, 3, 3)
    # Box plot
    sns.violinplot(
        df_pregnant,
        y="n_cycles_trying",
        color="lightblue",
    )
    plt.title("Violin Plot of conception time")
    plt.ylabel("Conception time [cycles]")

    # Save figure
    plt.tight_layout()
    plt.savefig(
        "reports/figures/usual_conception_time.png", dpi=300, bbox_inches="tight"
    )
    plt.show()

    return df_pregnant["n_cycles_trying"]


def main():
    """Main analysis function"""
    print("Natural Cycles Assessment - Question 2 Analysis")
    print("=" * 60)

    csv_file = "data/external/ncdatachallenge-2021-v1.csv"

    # Load and clean data
    df = load_and_clean_data(csv_file=csv_file)

    # Answer question 2
    cycles_to_pregnancy = calculate_usual_conception_time(df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    if cycles_to_pregnancy is not None:
        print(f"2. Median time to pregnancy: {cycles_to_pregnancy.median():.1f} cycles")
        print(f"   Mean time to pregnancy: {cycles_to_pregnancy.mean():.1f} cycles")


if __name__ == "__main__":
    main()
