import pandas as pd
import numpy as np


def load_and_clean_data(
    csv_file: str, clean_data: bool = True, remove_na: bool = False
):
    """Load and clean the dataset
    Args:
        csv_file: str, path to the csv file
        clean_data: bool, whether to clean the data (default: True)
        remove_na: bool, whether to remove rows with missing values (default: False)
    Returns:
        df: pd.DataFrame, the cleaned dataset
    """
    # Load the data
    df = pd.read_csv(csv_file, index_col=0)
    print(f"Original dataset shape: {df.shape}")


    # (Optional) Remove all rows with missing values
    if remove_na == True:
        print("Removing all rows with missing values")
        df = df.dropna()
    else:  # Remove rows with missing outcome or n_cycles_trying => essential for analysis => default
        df = df.dropna(subset=["outcome", "n_cycles_trying"])
    
    # Clean the data
    if clean_data == True:
        # Convert outcome to binary (1 for pregnant, 0 for not_pregnant)
        # This makes later analysis easier
        df["pregnant"] = (df["outcome"] == "pregnant").astype(int)
        df = df.drop(columns=["outcome"])

        # Clean BMI outliers (remove values that are likely errors)
        df = df[df["bmi"] > 10]  # Remove very low BMI values that are likely errors

        # Clean age outliers
        df = df[
            df["age"] >= 16
        ]  # Remove ages under the consent age (up for discussion)

    print(f"Cleaned dataset shape: {df.shape}")
    return df
