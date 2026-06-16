# src/reporting.py
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, auc, roc_curve

from src.evaluation import EvaluationResults
from src.paths import ExperimentPaths


def save_json(data: dict, path: Path) -> None:
    """Save a dictionary as a formatted JSON file."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            indent=4,
            ensure_ascii=False,
        )


def plot_training_curves(
    history_dict: dict,
    paths: ExperimentPaths,
) -> None:
    """
    Save training curves for loss and accuracy as PNG files.
    """
    if "loss" in history_dict and "val_loss" in history_dict:
        plt.figure(figsize=(8, 6))
        plt.plot(history_dict["loss"], label="train_loss")
        plt.plot(history_dict["val_loss"], label="val_loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.fig_dir / f"{paths.run_id}_loss_curve.png", dpi=200)
        plt.close()

    acc_key = (
        "accuracy"
        if "accuracy" in history_dict
        else "sparse_categorical_accuracy"
        if "sparse_categorical_accuracy" in history_dict
        else None
    )

    val_acc_key = (
        "val_accuracy"
        if "val_accuracy" in history_dict
        else "val_sparse_categorical_accuracy"
        if "val_sparse_categorical_accuracy" in history_dict
        else None
    )

    if acc_key is not None and val_acc_key is not None:
        plt.figure(figsize=(8, 6))
        plt.plot(history_dict[acc_key], label="train_accuracy")
        plt.plot(history_dict[val_acc_key], label="val_accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(paths.fig_dir / f"{paths.run_id}_accuracy_curve.png", dpi=200)
        plt.close()


def plot_confusion_matrix(
    results: EvaluationResults,
    paths: ExperimentPaths,
) -> None:
    """
    Save the test confusion matrix as a PNG file.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(
        confusion_matrix=results.confusion_matrix,
        display_labels=results.summary["class_names"],
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )
    plt.title("Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(paths.fig_dir / f"{paths.run_id}_cm.png", dpi=200)
    plt.close(fig)


def plot_roc_curve(
    results: EvaluationResults,
    paths: ExperimentPaths,
) -> None:
    """
    Save the ROC curve (One-vs-Rest) for each class as a PNG file.
    """
    class_names = results.summary["class_names"]
    num_classes = len(class_names)
    y_true_onehot = np.eye(num_classes)[results.y_true.astype(int)]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], results.y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(paths.fig_dir / f"{paths.run_id}_roc_curve.png", dpi=200)
    plt.close(fig)


def plot_entropy_histogram(
    values: np.ndarray,
    paths: ExperimentPaths,
    filename: str,
    title: str,
    xlabel: str,
) -> None:
    """
    Save a histogram for entropy-based uncertainty values.
    """
    plt.figure(figsize=(8, 6))
    plt.hist(values, bins=30, edgecolor="black")
    plt.xlabel(xlabel)
    plt.ylabel("Number of Samples")
    plt.title(title)
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(paths.fig_dir / filename, dpi=200)
    plt.close()


def export_entropy_artifacts(
    results: EvaluationResults,
    paths: ExperimentPaths,
) -> None:
    """
    Export per-sample entropy values for downstream analysis.
    """
    test_entropy_df = {
        "y_true": results.y_true,
        "y_pred": results.y_pred,
        "entropy": results.entropy,
        "normalized_entropy": results.normalized_entropy,
    }

    import pandas as pd

    pd.DataFrame(test_entropy_df).to_csv(
        paths.met_dir / f"{paths.run_id}_test_entropy.csv",
        index=False,
    )


def export_evaluation_artifacts(
    results: EvaluationResults,
    paths: ExperimentPaths,
) -> None:
    """
    Export evaluation artifacts:
    - validation report as CSV
    - test report as CSV
    - test entropy values as CSV
    - summary as JSON
    """
    results.classification_report_df.to_csv(
        paths.met_dir / f"{paths.run_id}_test_report.csv",
        index=True,
    )

    results.val_classification_report_df.to_csv(
        paths.met_dir / f"{paths.run_id}_validation_report.csv",
        index=True,
    )

    export_entropy_artifacts(results, paths)

    save_json(
        results.summary,
        paths.met_dir / f"{paths.run_id}_summary.json",
    )


def create_reports(
    history_dict: dict,
    results: EvaluationResults,
    paths: ExperimentPaths,
) -> None:
    """
    Create all reports and visualizations for an experiment.
    """
    plot_training_curves(history_dict, paths)
    plot_confusion_matrix(results, paths)
    plot_roc_curve(results, paths)

    plot_entropy_histogram(
        values=results.entropy,
        paths=paths,
        filename=f"{paths.run_id}_test_entropy_histogram.png",
        title="Test Entropy Distribution",
        xlabel="Entropy",
    )

    plot_entropy_histogram(
        values=results.normalized_entropy,
        paths=paths,
        filename=f"{paths.run_id}_test_normalized_entropy_histogram.png",
        title="Test Normalized Entropy Distribution",
        xlabel="Normalized Entropy",
    )

    export_evaluation_artifacts(results, paths)