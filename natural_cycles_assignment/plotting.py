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
    sns.set_palette("viridis")
    plt.rcParams["figure.figsize"] = (8, 6)
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 9
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["figure.titlesize"] = 14


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

    plt.title(title, fontsize=10, pad=5)
    plt.ylabel("Mean Cycles", fontsize=9)
    plt.xticks(x_pos, means.index, rotation=45, fontsize=8, ha="right")
    plt.yticks(fontsize=8)

    # Adjust subplot parameters to reduce whitespace
    plt.subplots_adjust(
        left=0.08, right=0.95, top=0.92, bottom=0.15, wspace=0.3, hspace=0.4
    )


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
    plt.figure(figsize=(6, 4))
    sns.barplot(
        data=associations_df, x="Association", y="Variable", hue="Type", dodge=False
    )
    plt.title(title)
    plt.xlabel("Association Strength (|correlation| or eta squared)")
    plt.ylabel("")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show(block=False)


def plot_predicted_vs_actual(
    y_true,
    y_pred,
    title="Predicted vs Actual",
    save_path=None,
    r2_score=None,
    model_params=None,
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
    model_params : dict, optional
        Dictionary of model parameters to display on the plot
    """
    # Convert to numpy arrays if they are lists
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    plt.figure(figsize=(5, 5))
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

    # Add model parameters as text label if provided
    if model_params is not None:
        # Format parameters for display
        param_text = "Model Parameters:\n"
        for key, value in model_params.items():
            if isinstance(value, float):
                param_text += f"{key}: {value:.3f}\n"
            else:
                param_text += f"{key}: {value}\n"

        plt.text(
            0.95,
            0.95,
            param_text.rstrip(),
            transform=plt.gca().transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show(block=False)


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

    plt.figure(figsize=(8, 5))
    top_features = feature_importance_df.head(top_n)
    sns.barplot(
        data=top_features,
        y="Feature",
        x="Importance",
        orient="h",
        palette="viridis",
    )
    plt.title(title, fontsize=12, pad=15)
    plt.xlabel("Importance Score", fontsize=10)
    plt.ylabel("Feature", fontsize=10)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout(pad=2.0)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show(block=False)


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

    plt.figure(figsize=(8, 3))

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
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show(block=False)


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
    plt.figure(figsize=(7, 6))
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
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

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
    plt.figure(figsize=(6, 4))
    plt.hist(data, bins=bins, alpha=0.7, edgecolor="black")
    plt.title(title)
    plt.xlabel("Value")
    plt.ylabel("Frequency")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show()


def plot_classification_heatmap(
    y_true,
    y_pred,
    title="Classification Results",
    save_path=None,
    model_params=None,
    class_names=None,
    accuracy=None,
    f1_score=None,
):
    """
    Create a classification heatmap (confusion matrix) for classification results.

    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    title : str
        Plot title
    save_path : str, optional
        Path to save the plot
    model_params : dict, optional
        Dictionary of model parameters to display on the plot
    class_names : list, optional
        List of class names for display
    accuracy : float, optional
        Accuracy score to display
    f1_score : float, optional
        F1 score to display
    """
    from sklearn.metrics import confusion_matrix
    import seaborn as sns

    # Convert to numpy arrays if they are lists
    if isinstance(y_true, list):
        y_true = np.array(y_true)
    if isinstance(y_pred, list):
        y_pred = np.array(y_pred)

    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages for better visualization
    cm_percent = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100

    plt.figure(figsize=(8, 6))

    # Add main title for the entire figure
    plt.suptitle(title, fontsize=14, y=0.98)

    # Create subplot for confusion matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="viridis",
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm)),
        annot_kws={"size": 8},
        square=True,
    )
    plt.title("Confusion Matrix (Counts)", fontsize=11, pad=15)
    plt.xlabel("Predicted", fontsize=10)
    plt.ylabel("Actual", fontsize=10)
    plt.xticks(fontsize=9, rotation=45, ha="right")
    plt.yticks(fontsize=9)

    # Create subplot for percentage confusion matrix
    plt.subplot(2, 2, 2)
    sns.heatmap(
        cm_percent,
        annot=True,
        fmt=".1f",
        cmap="viridis",
        xticklabels=class_names if class_names else range(len(cm)),
        yticklabels=class_names if class_names else range(len(cm)),
        annot_kws={"size": 8},
        square=True,
    )
    plt.title("Confusion Matrix (%)", fontsize=11, pad=15)
    plt.xlabel("Predicted", fontsize=10)
    plt.ylabel("Actual", fontsize=10)
    plt.xticks(fontsize=9, rotation=45, ha="right")
    plt.yticks(fontsize=9)

    # Create subplot for ROC curves (AUC)
    plt.subplot(2, 2, 3)
    from sklearn.metrics import roc_curve, auc
    from sklearn.preprocessing import label_binarize

    # Binarize the output for ROC curve calculation
    y_true_bin = label_binarize(y_true, classes=range(len(np.unique(y_true))))
    y_pred_bin = label_binarize(y_pred, classes=range(len(np.unique(y_pred))))

    # Calculate ROC curve and AUC for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(len(np.unique(y_true))):
        fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves with viridis colors
    colors = plt.cm.viridis([0.2, 0.5, 0.8])
    for i, color in enumerate(colors):
        if i < len(fpr):
            plt.plot(
                fpr[i],
                tpr[i],
                color=color,
                lw=2,
                label=f'{class_names[i] if class_names else f"Class {i}"} (AUC = {roc_auc[i]:.3f})',
            )

    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate", fontsize=10)
    plt.ylabel("True Positive Rate", fontsize=10)
    plt.title("ROC Curves (AUC)", fontsize=11, pad=15)
    plt.legend(loc="lower right", fontsize=8)
    plt.xticks(fontsize=9)
    plt.yticks(fontsize=9)

    # Create subplot for model parameters and performance metrics
    plt.subplot(2, 2, 4)
    if model_params is not None or accuracy is not None or f1_score is not None:
        # Format parameters and metrics for display
        info_text = ""

        if model_params is not None:
            info_text += "Model Parameters:\n"
            for key, value in model_params.items():
                if isinstance(value, float):
                    info_text += f"{key}: {value:.3f}\n"
                else:
                    info_text += f"{key}: {value}\n"
            info_text += "\n"

        if accuracy is not None:
            info_text += f"Accuracy: {accuracy:.3f}\n"

        if f1_score is not None:
            info_text += f"F1 Score: {f1_score:.3f}\n"

        plt.text(
            0.1,
            0.5,
            info_text.rstrip(),
            transform=plt.gca().transAxes,
            fontsize=9,
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
        )
        plt.title("Model Configuration & Performance", fontsize=11, pad=15)
    else:
        plt.text(
            0.5,
            0.5,
            "No parameters\navailable",
            transform=plt.gca().transAxes,
            ha="center",
            va="center",
            fontsize=10,
        )
        plt.title("Model Configuration", fontsize=11, pad=15)

    plt.axis("off")

    plt.tight_layout(pad=1.0, h_pad=0.5, w_pad=0.5)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    plt.show(block=False)
