import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from natural_cycles_assignment.import_data import load_and_clean_data
from natural_cycles_assignment.plotting import setup_plotting_style

# Set up plotting style
setup_plotting_style()


def calculate_pregnancy_chance_within_13_cycles(df: pd.DataFrame):
    """Answer: What is the chance of getting pregnant within 13 cycles?
    Args:
        df: pd.DataFrame, the cleaned dataset
    Returns:
        all_participants_rate_within_13: float, the rate of participants who got pregnant within 13 cycles
    """
    print("\n" + "=" * 60)
    print("QUESTION 1: What is the chance of getting pregnant within 13 cycles?")
    print("=" * 60)

    # Calculate percentage of participants who got pregnant in the dataset
    total_pregnant = df["pregnant"].sum()
    total_participants = len(df)
    overall_pregnancy_rate = total_pregnant / total_participants

    print(
        f"Overall pregnancy rate:\t\t{overall_pregnancy_rate*100:.1f}%\t({total_pregnant} pregnancie / {total_participants} participants)"
    )

    # Calculate pregnancy rate for participants who tried for 13 cycles or less
    df_within_13_cycles = df[df["n_cycles_trying"] <= 13]
    pregnant_within_13 = df_within_13_cycles["pregnant"].sum()

    if total_pregnant > 0:
        all_participants_rate_within_13 = pregnant_within_13 / total_participants
        pregnant_participants_rate_within_13 = pregnant_within_13 / total_pregnant
        print(
            f"Of all participants\t\t{all_participants_rate_within_13*100:.1f}%\t({pregnant_within_13} pregnancies / {total_participants} participants) got pregnant <= 13 cycles"
        )
        print(
            f"Of all pregnant participants\t{pregnant_participants_rate_within_13*100:.1f}%\t({total_pregnant} pregnancies / {total_pregnant} participants) did this <= 13 cycles)"
        )
    else:
        print("No pregnant participants found in the dataset!")

    # PLOTTING
    plt.figure(figsize=(10, 6))
    # Create histogram of cycles trying
    plt.subplot(1, 2, 1)
    sns.histplot(
        df, x="n_cycles_trying", alpha=0.7, discrete=True, kde=True, color="lightgreen"
    )
    plt.axvline(
        x=13,
        color="red",
        linestyle="--",
        label=f"13 cycles ({all_participants_rate_within_13*100:.1f}% of participants)",
    )
    plt.title("ALL PARTICIPANTS:Histogram of Cycles needed to Conceive")
    plt.xlabel("Number of Cycles Trying")
    plt.ylabel("Count")
    plt.legend()

    # Cumulative histogram in percentage
    # BUG: the cumulative histogram is not correct, it should not total to 100%, because not all participants got pregnant!
    plt.subplot(1, 2, 2)
    sns.histplot(
        df[df["pregnant"] == 1],
        x="n_cycles_trying",
        alpha=0.7,
        discrete=True,
        cumulative=True,
        stat="percent",
        thresh=90,
    )
    plt.axvline(x=13, color="red", linestyle="--", label="13 cycles")
    plt.axhline(
        y=pregnant_participants_rate_within_13 * 100,
        color="black",
        linestyle="--",
        label=f"{pregnant_participants_rate_within_13*100:.1f}%",
    )
    plt.title("PREGNANT PARTICIPANTS: Cumulative Pregnancy Rate")
    plt.xlabel("Number of Cycles Trying")
    plt.ylabel("Pregnancy Rate [%]")
    plt.legend()

    # Save figure
    plt.tight_layout()
    plt.savefig(
        "reports/figures/q1_pregnancy_chance_13_cycles.png",
        dpi=300,
        bbox_inches="tight",
    )
    plt.show(block=False)

    return all_participants_rate_within_13


def main():
    """Main analysis function"""
    print("Natural Cycles Assessment - Question 1 Analysis")
    print("=" * 60)

    csv_file = "data/external/ncdatachallenge-2021-v1.csv"

    # Load and clean data
    df = load_and_clean_data(
        csv_file=csv_file, clean_outliers=False, remove_na=False
    )  # not necessary to clean data for this question

    # Answer question 1
    all_participants_rate_within_13 = calculate_pregnancy_chance_within_13_cycles(df)

    # Keep all figures open until user closes them
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE - Close figure windows to exit")
    print("=" * 60)
    plt.show()


if __name__ == "__main__":
    main()
