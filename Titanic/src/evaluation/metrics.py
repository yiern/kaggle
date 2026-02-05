"""Evaluation metrics and visualization utilities."""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_score

try:
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    import seaborn as sns

    sns.set_style("whitegrid")
except ImportError:
    raise ImportError(
        "Visualization libraries not installed. Run: pip install matplotlib seaborn"
    )

logger = logging.getLogger(__name__)


def cross_validate(
    model,
    X: pd.DataFrame,
    y: Union[pd.Series, np.ndarray],
    cv_folds: int = 10,
    random_state: int = 42,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Perform stratified k-fold cross-validation.

    Args:
        model: Model instance with fit/predict methods
        X: Feature matrix
        y: Target labels
        cv_folds: Number of CV folds (default: 10)
        random_state: Random state for reproducibility
        verbose: Whether to log results

    Returns:
        Dictionary with CV scores and statistics
    """
    logger.info(f"Performing {cv_folds}-fold stratified cross-validation...")

    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)

    # Perform cross-validation
    scores = cross_val_score(model, X, y, cv=skf, scoring="accuracy", n_jobs=-1)

    results = {
        "scores": scores,
        "mean": scores.mean(),
        "std": scores.std(),
        "min": scores.min(),
        "max": scores.max(),
        "folds": cv_folds,
    }

    if verbose:
        logger.info(
            f"CV Accuracy: {results['mean']:.4f} (+/- {results['std'] * 2:.4f})"
        )
        logger.info(f"Min: {results['min']:.4f}, Max: {results['max']:.4f}")

    return results


def evaluate_model(
    model,
    X_train: pd.DataFrame,
    y_train: Union[pd.Series, np.ndarray],
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[Union[pd.Series, np.ndarray]] = None,
    verbose: bool = True,
) -> Dict[str, float]:
    """Evaluate model on train and validation sets.

    Args:
        model: Trained model with predict method
        X_train: Training features
        y_train: Training labels
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        verbose: Whether to print results

    Returns:
        Dictionary with evaluation metrics
    """
    results = {}

    # Training metrics
    train_pred = model.predict(X_train)
    results["train_accuracy"] = accuracy_score(y_train, train_pred)

    # Validation metrics
    if X_val is not None and y_val is not None:
        val_pred = model.predict(X_val)
        results["val_accuracy"] = accuracy_score(y_val, val_pred)

        if verbose:
            logger.info(f"Train Accuracy: {results['train_accuracy']:.4f}")
            logger.info(f"Val Accuracy: {results['val_accuracy']:.4f}")
            logger.info(
                f"Gap: {results['train_accuracy'] - results['val_accuracy']:.4f}"
            )

            # Classification report
            report = classification_report(
                y_val, val_pred, target_names=["Died", "Survived"]
            )
            logger.info(f"\nClassification Report:\n{report}")
    else:
        if verbose:
            logger.info(f"Train Accuracy: {results['train_accuracy']:.4f}")

    return results


def plot_confusion_matrix(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (8, 6),
    title: str = "Confusion Matrix",
):
    """Plot confusion matrix heatmap.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        save_path: Path to save figure (optional)
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    # Calculate confusion matrix
    cm = confusion_matrix(y_true, y_pred)

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Died", "Survived"],
        yticklabels=["Died", "Survived"],
        ax=ax,
        cbar_kws={"label": "Count"},
    )

    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("Actual", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add accuracy annotation
    accuracy = accuracy_score(y_true, y_pred)
    ax.text(
        0.5,
        -0.15,
        f"Accuracy: {accuracy:.4f}",
        transform=ax.transAxes,
        ha="center",
        fontsize=11,
        style="italic",
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Confusion matrix saved to {save_path}")

    return fig


def plot_feature_importance(
    importance_df: pd.DataFrame,
    save_path: Optional[Union[str, Path]] = None,
    top_n: int = 10,
    figsize: Tuple[int, int] = (10, 6),
    importance_type: str = "gain",
):
    """Plot feature importance bar chart.

    Args:
        importance_df: DataFrame with 'feature' and 'importance' columns
        save_path: Path to save figure (optional)
        top_n: Number of top features to show
        figsize: Figure size
        importance_type: Type of importance metric

    Returns:
        Matplotlib figure object
    """
    # Get top N features
    top_features = importance_df.head(top_n).copy()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot horizontal bar chart
    sns.barplot(
        data=top_features, y="feature", x="importance", palette="viridis", ax=ax
    )

    ax.set_xlabel(f"Importance ({importance_type})", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title(
        f"Top {top_n} Feature Importance ({importance_type})",
        fontsize=14,
        fontweight="bold",
    )

    # Add value labels
    for i, (idx, row) in enumerate(top_features.iterrows()):
        ax.text(
            row["importance"] + 0.01,
            i,
            f"{row['importance']:.3f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Feature importance plot saved to {save_path}")

    return fig


def plot_learning_curve(
    train_scores: np.ndarray,
    val_scores: np.ndarray,
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 6),
    title: str = "Learning Curve (10-Fold CV)",
):
    """Plot learning curve showing train vs validation scores.

    Args:
        train_scores: Training scores for each fold
        val_scores: Validation scores for each fold
        save_path: Path to save figure (optional)
        figsize: Figure size
        title: Plot title

    Returns:
        Matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)

    folds = np.arange(1, len(train_scores) + 1)

    # Plot scores
    ax.plot(folds, train_scores, "o-", label="Training", color="blue", linewidth=2)
    ax.plot(folds, val_scores, "o-", label="Validation", color="red", linewidth=2)

    # Add mean lines
    ax.axhline(
        y=train_scores.mean(),
        color="blue",
        linestyle="--",
        alpha=0.5,
        label=f"Train Mean: {train_scores.mean():.4f}",
    )
    ax.axhline(
        y=val_scores.mean(),
        color="red",
        linestyle="--",
        alpha=0.5,
        label=f"Val Mean: {val_scores.mean():.4f}",
    )

    ax.set_xlabel("Fold", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Add gap annotation
    gap = train_scores.mean() - val_scores.mean()
    ax.text(
        0.02,
        0.98,
        f"Gap: {gap:.4f}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Learning curve saved to {save_path}")

    return fig


def plot_hyperparameter_results(
    cv_results: Dict[str, Any],
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """Plot hyperparameter tuning results.

    Args:
        cv_results: GridSearchCV.cv_results_ dictionary
        save_path: Path to save figure (optional)
        figsize: Figure size

    Returns:
        Matplotlib figure object
    """
    results_df = pd.DataFrame(cv_results)

    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    fig.suptitle("Hyperparameter Tuning Results", fontsize=16, fontweight="bold")

    # Plot 1: Mean test score over iterations
    ax1 = axes[0, 0]
    ax1.plot(results_df["mean_test_score"], "o-", color="blue", alpha=0.7)
    ax1.set_xlabel("Parameter Set Index")
    ax1.set_ylabel("Mean CV Score")
    ax1.set_title("CV Score by Parameter Set")
    ax1.grid(True, alpha=0.3)

    # Highlight best score
    best_idx = results_df["mean_test_score"].idxmax()
    ax1.scatter(
        [best_idx],
        [results_df.loc[best_idx, "mean_test_score"]],
        color="red",
        s=100,
        zorder=5,
        label=f"Best: {results_df.loc[best_idx, 'mean_test_score']:.4f}",
    )
    ax1.legend()

    # Plot 2: n_estimators vs score
    ax2 = axes[0, 1]
    for n_est in results_df["param_n_estimators"].unique():
        if n_est is not None:
            mask = results_df["param_n_estimators"] == n_est
            ax2.scatter(
                results_df.loc[mask, "param_max_depth"],
                results_df.loc[mask, "mean_test_score"],
                label=f"n_est={n_est}",
                alpha=0.7,
                s=60,
            )
    ax2.set_xlabel("Max Depth")
    ax2.set_ylabel("Mean CV Score")
    ax2.set_title("Max Depth vs Score (colored by n_estimators)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: learning_rate vs score
    ax3 = axes[1, 0]
    for lr in results_df["param_learning_rate"].unique():
        if lr is not None:
            mask = results_df["param_learning_rate"] == lr
            ax3.scatter(
                results_df.loc[mask, "param_subsample"],
                results_df.loc[mask, "mean_test_score"],
                label=f"lr={lr}",
                alpha=0.7,
                s=60,
            )
    ax3.set_xlabel("Subsample")
    ax3.set_ylabel("Mean CV Score")
    ax3.set_title("Subsample vs Score (colored by learning_rate)")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Score distribution
    ax4 = axes[1, 1]
    ax4.hist(
        results_df["mean_test_score"],
        bins=15,
        alpha=0.7,
        color="green",
        edgecolor="black",
    )
    ax4.axvline(
        results_df["mean_test_score"].max(),
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Best: {results_df['mean_test_score'].max():.4f}",
    )
    ax4.set_xlabel("Mean CV Score")
    ax4.set_ylabel("Frequency")
    ax4.set_title("Score Distribution")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save if path provided
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Hyperparameter results saved to {save_path}")

    return fig
