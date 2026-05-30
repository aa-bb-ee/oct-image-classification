# src/reporting.py
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from src.evaluation import EvaluationResults
from src.paths import ExperimentPaths


def save_json(data: dict, path: Path) -> None:
    """Speichert ein Dictionary formatiert als JSON-Datei."""
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
    Speichert Lernkurven für Loss und Accuracy als PNG-Dateien.
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
    Speichert die Confusion Matrix als PNG-Datei.
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


def export_evaluation_artifacts(
    results: EvaluationResults,
    paths: ExperimentPaths,
) -> None:
    """
    Exportiert Evaluationsartefakte:
    - Classification Report als CSV
    - Summary als JSON
    """
    results.classification_report_df.to_csv(
        paths.met_dir / f"{paths.run_id}_report.csv",
        index=True,
    )

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
    Erzeugt alle Reports und Visualisierungen eines Experiments.
    """
    plot_training_curves(history_dict, paths)
    plot_confusion_matrix(results, paths)
    export_evaluation_artifacts(results, paths)
