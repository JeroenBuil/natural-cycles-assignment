import pandas as pd
import numpy as np


def load_and_clean_data(csv_file: str, remove_na: bool = False):
    """Load and clean the dataset
    Args:
        csv_file: str, path to the csv file
        remove_na: bool, whether to remove rows with missing values (default: False)
    Returns:
        df: pd.DataFrame, the cleaned dataset
    """
    # Load the data
    df = pd.read_csv(csv_file, index_col=0)
    print(f"Original dataset shape: {df.shape}")

    # Clean the data
    # (Optional) Remove all rows with missing values
    if remove_na == True:
        print("Removing all rows with missing values")
        df = df.dropna()

    # Remove rows with missing outcome or n_cycles_trying
    df = df.dropna(subset=["outcome", "n_cycles_trying"])

    # Convert outcome to binary (1 for pregnant, 0 for not_pregnant)
    # This makes later analysis easier
    df["pregnant"] = (df["outcome"] == "pregnant").astype(int)

    # Clean BMI outliers (remove values that are likely errors)
    df = df[df["bmi"] > 10]  # Remove very low BMI values that are likely errors

    # Clean age outliers
    df = df[df["age"] >= 16]  # Remove ages under the consent age (up for discussion)

    print(f"Cleaned dataset shape: {df.shape}")
    return df
