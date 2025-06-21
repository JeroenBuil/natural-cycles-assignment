"""
Plotting utilities for Natural Cycles assignment analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd


def setup_plotting_style():
    """Set up consistent plotting style for the project."""
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")


def create_factor_plot(df, group_column, title, subplot_pos, color, xlabel):
    """Create a bar plot for a factor's impact on cycles to pregnancy.

    Args:
        df: DataFrame with pregnant women
        group_column: Column name for grouping
        title: Plot title
        subplot_pos: Subplot position (1-8)
        color: Bar color
        xlabel: X-axis label
    """
    plt.subplot(2, 4, subplot_pos)

    # Calculate means and standard errors while preserving order
    group_stats = df.groupby(group_column, observed=True)["n_cycles_trying"].agg(
        ["mean", "std", "count"]
    )
    means = group_stats["mean"]
    # Standard error = std / sqrt(n)
    std_errors = group_stats["std"] / np.sqrt(group_stats["count"])

    # Create bar plot with error bars
    x_pos = np.arange(len(means))
    bars = plt.bar(
        x_pos, means.values, color=color, alpha=0.7, yerr=std_errors.values, capsize=5
    )

    plt.title(title, fontsize=10, pad=10)
    plt.xlabel(xlabel, fontsize=9)
    plt.ylabel("Mean Cycles", fontsize=9)
    plt.xticks(x_pos, means.index, rotation=45, fontsize=8)
    plt.yticks(fontsize=8)


def create_factors_overview_plot(
    associations_df,
    title="Association of Variables with Cycles to Pregnancy",
    save_path=None,
):
    """Create an overview plot showing association strengths for all factors.

    Args:
        associations_df: DataFrame with Variable, Type, and Association columns
        title: Plot title
        save_path: Path to save the plot
    """
    plt.figure(figsize=(8, 5))
    sns.barplot(
        data=associations_df, x="Association", y="Variable", hue="Type", dodge=False
    )
    plt.title(title)
    plt.xlabel("Association Strength (|correlation| or eta squared)")
    plt.ylabel("")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_predicted_vs_actual(
    y_true, y_pred, title="Predicted vs Actual", save_path=None, r2_score=None
):
    """
    Create a predicted vs actual scatter plot.

    Parameters:
    -----------
    y_true : array-like
        True values
    y_pred : array-like
        Predicted values
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    r2_score : float, optional
        R² score to display on the plot
    """
    plt.figure(figsize=(7, 7))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.plot(
        [y_true.min(), y_true.max()],
        [y_true.min(), y_true.max()],
        "r--",
        lw=2,
    )
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title(title)

    # Add R² score as text label if provided
    if r2_score is not None:
        plt.text(
            0.05,
            0.95,
            f"R² = {r2_score:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_feature_importance(
    feature_importance_df,
    title="Feature Importance",
    save_path=None,
    top_n=10,
    print_features=True,
):
    """
    Create a horizontal bar plot of feature importance and optionally print top features.

    Parameters:
    -----------
    feature_importance_df : pandas.DataFrame
        DataFrame with 'Feature' and 'Importance' columns
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    top_n : int
        Number of top features to display
    print_features : bool
        Whether to print the top features to console
    """
    if print_features:
        print(f"\n{title} (top {top_n}):")
        print(feature_importance_df.head(top_n))

    plt.figure(figsize=(10, 6))
    top_features = feature_importance_df.head(top_n)
    sns.barplot(
        data=top_features,
        y="Feature",
        x="Importance",
        orient="h",
    )
    plt.title(title)
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_model_comparison(metrics_dict, title="Model Comparison", save_path=None):
    """
    Create a comparison plot of different models.

    Parameters:
    -----------
    metrics_dict : dict
        Dictionary with model names as keys and lists of metrics as values
        Format: {'model_name': [r2_score, rmse_score]}
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    models = list(metrics_dict.keys())
    r2_scores = [metrics_dict[model][0] for model in models]
    rmse_scores = [metrics_dict[model][1] for model in models]
    colors = ["skyblue", "lightgreen"]

    plt.figure(figsize=(12, 5))

    # R² comparison
    plt.subplot(1, 2, 1)
    bars = plt.bar(models, r2_scores, color=colors, alpha=0.7)
    plt.title("R² Score Comparison")
    plt.ylabel("R² Score")
    plt.ylim(min(r2_scores) - 0.1, max(r2_scores) + 0.1)

    # Add value labels on bars
    for bar, score in zip(bars, r2_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    # RMSE comparison
    plt.subplot(1, 2, 2)
    bars = plt.bar(models, rmse_scores, color=colors, alpha=0.7)
    plt.title("RMSE Comparison")
    plt.ylabel("RMSE")
    plt.ylim(0, max(rmse_scores) * 1.1)

    # Add value labels on bars
    for bar, score in zip(bars, rmse_scores):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{score:.3f}",
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_correlation_matrix(df, title="Correlation Matrix", save_path=None):
    """
    Create a correlation matrix heatmap.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame to create correlation matrix for
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    """
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        center=0,
        square=True,
        fmt=".2f",
    )
    plt.title(title)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()


def plot_distribution(data, title="Distribution", save_path=None, bins=30):
    """
    Create a histogram of data distribution.

    Parameters:
    -----------
    data : array-like
        Data to plot
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    bins : int
        Number of bins for histogram
    """
    plt.figure(figsize=(8, 6))
    plt.hist(data, bins=bins, alpha=0.7, edgecolor="black")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
