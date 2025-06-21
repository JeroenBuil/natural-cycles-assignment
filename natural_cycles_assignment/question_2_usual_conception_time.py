import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natural_cycles_assignment.import_data import load_and_clean_data
from natural_cycles_assignment.plotting import setup_plotting_style

# Set up plotting style
setup_plotting_style()


def calculate_usual_conception_time(df: pd.DataFrame):
    """Answer: How long does it usually take to get pregnant?
    Args:
        df: pd.DataFrame, the cleaned dataset
    Returns:
        df_pregnant["n_cycles_trying"]: pd.Series, the number of cycles trying to get pregnant for each participant
    """
    print("\n" + "=" * 60)
    print("QUESTION 2: How long does it usually take to get pregnant?")
    print("=" * 60)

    count_participants = len(df)
    # Only participants who got pregnant are relevant for this question
    df_pregnant = df[df["pregnant"] == 1]

    if len(df_pregnant) == 0:
        print("No pregnant participants found in the dataset!")
        return None

    # Calculate relevant statistics for conception time
    count_participants_pregnant = len(df_pregnant)
    median_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].median()
    mean_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].mean()
    std_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].std()
    min_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].min()
    max_cycles_to_pregnancy = df_pregnant["n_cycles_trying"].max()

    print(f"Total nr of participants: {count_participants}")
    print(
        f"Nr of participants who got pregnant: {count_participants_pregnant} ({(count_participants_pregnant/count_participants)*100:.2f}%)"
    )
    print(f"Mean conception time: {mean_cycles_to_pregnancy:.2f} cycles")
    print(f"Median conception time: {median_cycles_to_pregnancy:.2f} cycles")
    print(f"Standard deviation: {std_cycles_to_pregnancy:.2f} cycles")
    print(f"Min conception time: {min_cycles_to_pregnancy} cycles")
    print(f"Max conception time: {max_cycles_to_pregnancy} cycles")

    # Create visualizations
    plt.figure(figsize=(10, 4))

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
        label=f"Mean: {mean_cycles_to_pregnancy:.1f} ± {std_cycles_to_pregnancy:.1f}",
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

    # Cumulative distribution
    plt.subplot(1, 3, 2)
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

    # Box plot
    plt.subplot(1, 3, 3)

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
        "reports/figures/q2_usual_conception_time.png", dpi=300, bbox_inches="tight"
    )
    plt.show(block=False)

    return df_pregnant["n_cycles_trying"]


def main():
    """Main analysis function"""
    print("Natural Cycles Assessment - Question 2 Analysis")
    print("=" * 60)

    csv_file = "data/external/ncdatachallenge-2021-v1.csv"

    # Load and clean data
    df = load_and_clean_data(
        csv_file=csv_file, clean_outliers=False, remove_na=False
    )  # not necessary to clean data for this question

    # Answer question 2
    df_cycles_to_pregnancy = calculate_usual_conception_time(df)

    # Keep all figures open until user closes them
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - Close figure windows to exit")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    main()
