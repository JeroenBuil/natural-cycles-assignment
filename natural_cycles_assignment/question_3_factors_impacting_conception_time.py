import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from natural_cycles_assignment.import_data import load_and_clean_data
from natural_cycles_assignment.plotting import (
    create_factor_plot,
    create_factors_overview_plot,
    setup_plotting_style,
)
from sklearn.model_selection import train_test_split, KFold
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import pointbiserialr, f_oneway

# Set up plotting style
setup_plotting_style()


def analyze_categorical_factor(df, column, title, bins=None, labels=None):
    """Analyze a categorical factor's impact on cycles to pregnancy.

    Args:
        df: DataFrame with pregnant participants
        column: Column name to analyze
        title: Title for the analysis
        bins: Bins for pd.cut (if None, uses existing categories)
        labels: Labels for bins (if None, uses existing categories)

    Returns:
        tuple: (analysis_df, group_column_name)
    """
    print(f"\n{title}:")

    if bins is not None and labels is not None:
        # Create binned groups with ordered categories
        group_column = f"{column}_group"
        df[group_column] = pd.cut(
            df[column], bins=bins, labels=labels, include_lowest=True, ordered=True
        )
        # Convert to categorical with the specified order
        df[group_column] = pd.Categorical(
            df[group_column], categories=labels, ordered=True
        )
    elif labels is not None and bins is None:
        # For categorical variables with predefined order but no bins
        group_column = column
        # Only use categories that actually exist in the data
        existing_categories = df[column].dropna().unique()
        valid_labels = [label for label in labels if label in existing_categories]
        if len(valid_labels) > 0:
            # Convert to categorical with the valid order
            df[group_column] = pd.Categorical(
                df[column], categories=valid_labels, ordered=True
            )
        else:
            # If no expected labels match, use existing categories
            df[group_column] = pd.Categorical(
                df[column], categories=existing_categories, ordered=True
            )
    else:
        # Use existing categories, maintaining their order
        group_column = column
        if df[column].dtype == "object":
            # For object columns, preserve the order they appear in the data
            unique_values = df[column].dropna().unique()
            df[group_column] = pd.Categorical(
                df[column], categories=unique_values, ordered=True
            )

    analysis = (
        df.groupby(group_column, observed=True)["n_cycles_trying"]
        .agg(["mean", "median", "count"])
        .round(2)
    )
    print(analysis)

    return analysis, group_column


def perform_statistical_test(df, group_column, test_type="anova"):
    """Perform statistical test for group differences.

    Args:
        df: DataFrame with pregnant participants
        group_column: Column name for grouping
        test_type: Type of test ("anova" or "ttest")
    """
    groups = [
        group["n_cycles_trying"].values
        for name, group in df.groupby(group_column)
        if len(group) > 0
    ]

    if len(groups) < 2:
        return

    if test_type == "anova" and len(groups) > 1:
        f_stat, p_value = stats.f_oneway(*groups)
        print(f"ANOVA p-value for {group_column}: {p_value:.4f}")
    elif test_type == "ttest" and len(groups) == 2:
        t_stat, p_value = stats.ttest_ind(groups[0], groups[1])
        print(f"T-test p-value for {group_column}: {p_value:.4f}")


def calculate_associations(df, numeric_cols, categorical_vars):
    """Calculate associations between variables and cycles to pregnancy.

    Args:
        df: DataFrame with pregnant participants
        numeric_cols: List of numeric column names
        categorical_vars: List of tuples (name, series) for categorical variables

    Returns:
        DataFrame: Association strengths
    """
    associations = []

    # Numerical variables (Pearson correlation)
    correlation_matrix = df[numeric_cols + ["n_cycles_trying"]].corr()
    for col in numeric_cols:
        corr = correlation_matrix.loc[col, "n_cycles_trying"]
        associations.append(
            {"Variable": col, "Type": "Numerical (Pearson)", "Association": abs(corr)}
        )

    # Categorical variables
    for name, series in categorical_vars:
        if series.nunique() == 2:
            # Binary categorical: point-biserial correlation
            try:
                vals = pd.get_dummies(series, drop_first=True)
                corr, _ = pointbiserialr(vals.values.flatten(), df["n_cycles_trying"])
                associations.append(
                    {
                        "Variable": name,
                        "Type": "Categorical (point-biserial)",
                        "Association": abs(corr),
                    }
                )
            except Exception:
                associations.append(
                    {
                        "Variable": name,
                        "Type": "Categorical (point-biserial)",
                        "Association": np.nan,
                    }
                )
        elif series.nunique() > 1:
            # Multi-class: eta squared from ANOVA
            try:
                groups = [
                    df.loc[series == cat, "n_cycles_trying"].values
                    for cat in series.dropna().unique()
                ]
                f, p = f_oneway(*groups)
                # eta squared = SSB/SST
                grand_mean = df["n_cycles_trying"].mean()
                ssb = sum([len(g) * (np.mean(g) - grand_mean) ** 2 for g in groups])
                sst = sum((df["n_cycles_trying"] - grand_mean) ** 2)
                eta_sq = ssb / sst if sst > 0 else np.nan
                associations.append(
                    {
                        "Variable": name,
                        "Type": "Categorical (eta squared)",
                        "Association": eta_sq,
                    }
                )
            except Exception:
                associations.append(
                    {
                        "Variable": name,
                        "Type": "Categorical (eta squared)",
                        "Association": np.nan,
                    }
                )

    return pd.DataFrame(associations).sort_values("Association", ascending=False)


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

    # Only consider participants who got pregnant for this analysis
    df_pregnant = df[df["pregnant"] == 1].copy()

    if len(df_pregnant) == 0:
        print("No pregnant women found in the dataset!")
        return None

    # Define factor configurations
    factor_configs = [
        {
            "column": "age",
            "title": "1. AGE ANALYSIS",
            "bins": [18, 25, 30, 35, 50],
            "labels": ["18-25", "26-30", "31-35", "36-50"],
            "test_type": "anova",
            "plot_config": {
                "title": "Mean Cycles to Pregnancy by Age Group",
                "color": "skyblue",
                "xlabel": "Age Group",
            },
        },
        {
            "column": "bmi",
            "title": "2. BMI ANALYSIS",
            "bins": [0, 18.5, 24, 30, 50],
            "labels": [
                "Underweight (<18.5)",
                "Normal (18.5-24)",
                "Overweight (24-30)",
                "Obese (>30)",
            ],
            "test_type": "anova",
            "plot_config": {
                "title": "Mean Cycles to Pregnancy by BMI Group",
                "color": "lightcoral",
                "xlabel": "BMI Group",
            },
        },
        {
            "column": "been_pregnant_before",
            "title": "3. PREVIOUS PREGNANCY ANALYSIS",
            "bins": None,
            "labels": ["No, never", "Yes, once", "Yes, twice", "Yes 3 times or more"],
            "test_type": "anova",
            "plot_config": {
                "title": "Mean Cycles to Pregnancy by Previous Pregnancy",
                "color": "lightgreen",
                "xlabel": "Been Pregnant Before",
            },
        },
        {
            "column": "regular_cycle",
            "title": "4. CYCLE REGULARITY ANALYSIS",
            "bins": None,
            "labels": None,
            "test_type": "ttest",
            "plot_config": {
                "title": "Mean Cycles to Pregnancy by Cycle Regularity",
                "color": "gold",
                "xlabel": "Regular Cycle",
            },
        },
        {
            "column": "dedication",
            "title": "5. DEDICATION ANALYSIS",
            "bins": [0, 0.5, 0.8, 1.0],
            "labels": ["Low (<0.5)", "Medium (0.5-0.8)", "High (>0.8)"],
            "test_type": "anova",
            "plot_config": {
                "title": "Mean Cycles to Pregnancy by Dedication",
                "color": "plum",
                "xlabel": "Dedication Level",
            },
        },
        {
            "column": "intercourse_frequency",
            "title": "6. INTERCOURSE FREQUENCY ANALYSIS",
            "bins": [0, 0.05, 0.15, 0.3, 1.0],
            "labels": [
                "Very Low (<0.05)",
                "Low (0.05-0.15)",
                "Medium (0.15-0.3)",
                "High (>0.3)",
            ],
            "test_type": "anova",
            "plot_config": {
                "title": "Mean Cycles to Pregnancy by Intercourse Frequency",
                "color": "orange",
                "xlabel": "Intercourse Frequency",
            },
        },
        {
            "column": "sleeping_pattern",
            "title": "7. SLEEPING PATTERN ANALYSIS",
            "bins": None,
            "labels": [
                "Wake same every day",
                "Wake same every workday",
                "Shift work",
                "Several times during the night",
                "Late and snoozer",
            ],
            "test_type": "anova",
            "plot_config": {
                "title": "Mean Cycles to Pregnancy by Sleep Pattern",
                "color": "orange",
                "xlabel": "Sleep Pattern",
            },
        },
        {
            "column": "education",
            "title": "8. EDUCATION ANALYSIS",
            "bins": None,
            "labels": [
                "Elementary school",
                "High school",
                "Trade/technical/vocational training",
                "University",
                "PhD",
            ],
            "test_type": "anova",
            "plot_config": {
                "title": "Mean Cycles to Pregnancy by Education",
                "color": "purple",
                "xlabel": "Education",
            },
        },
    ]

    # Analyze all factors
    group_columns = []
    for i, config in enumerate(factor_configs):
        # Debug: Print unique values for education to see what's actually in the data
        if config["column"] == "education":
            print(f"\nDEBUG: Unique education values in data:")
            print(df_pregnant["education"].value_counts())
            print(f"Expected labels: {config['labels']}")

        analysis, group_col = analyze_categorical_factor(
            df_pregnant,
            config["column"],
            config["title"],
            config["bins"],
            config["labels"],
        )
        group_columns.append(group_col)

        # Perform statistical test
        perform_statistical_test(df_pregnant, group_col, config["test_type"])

    # Create visualizations
    plt.figure(figsize=(16, 10))

    for i, (config, group_col) in enumerate(zip(factor_configs, group_columns)):
        create_factor_plot(
            df_pregnant,
            group_col,
            config["plot_config"]["title"],
            i + 1,
            config["plot_config"]["color"],
            config["plot_config"]["xlabel"],
        )

    plt.tight_layout(pad=3.0)
    plt.savefig("reports/figures/q3_factors_impact.png", dpi=300, bbox_inches="tight")
    plt.show()

    # Correlation analysis
    print("\n9. CORRELATION ANALYSIS:")
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

    # Calculate associations for all variables
    categorical_vars = [(name, df_pregnant[name]) for name in group_columns]

    associations_df = calculate_associations(
        df_pregnant, numeric_cols, categorical_vars
    )

    # Create overview plot
    create_factors_overview_plot(
        associations_df, save_path="reports/figures/q3_overview_factors_correlation.png"
    )

    return df_pregnant


def main():
    """Main analysis function"""
    print("NATURAL CYCLES PREGNANCY ANALYSIS")
    print("=" * 60)

    csv_file = "data/external/ncdatachallenge-2021-v1.csv"

    # Load and clean data + remove na
    df = load_and_clean_data(csv_file=csv_file, clean_data=True, remove_na=True)

    # Answer question 3
    factors_analysis = question_3_factors_impacting_conception_time(df)

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY OF FINDINGS")
    print("=" * 60)

    print("3. Key factors affecting time to pregnancy:")
    print("   - Age (older participants may take longer)")
    print("   - Previous pregnancy history")
    print("   - BMI (both underweight and overweight)")
    print("   - Cycle regularity")
    print("   - App dedication and intercourse frequency")


if __name__ == "__main__":
    main()
