# src/evaluation.py
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from tensorflow import keras

from src.config import PipelineConfig
from src.data_loader import DatasetBundle
from src.paths import ExperimentPaths


@dataclass
class EvaluationResults:
    model: keras.Model
    y_true: np.ndarray
    y_pred: np.ndarray
    y_prob: np.ndarray
    entropy: np.ndarray
    normalized_entropy: np.ndarray
    confusion_matrix: np.ndarray
    classification_report_dict: dict
    classification_report_df: pd.DataFrame
    val_classification_report_dict: dict
    val_classification_report_df: pd.DataFrame
    val_auc: float
    val_auc_weighted: float
    eval_dict: dict[str, float]
    summary: dict


def _build_eval_metrics(num_classes: int) -> list[str]:
    """Build the built-in metrics for model.evaluate()."""
    metrics = ["sparse_categorical_accuracy"]

    if num_classes > 2:
        metrics.append("sparse_top_k_categorical_accuracy")

    return metrics


def _safe_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> tuple[float, float]:
    """
    Compute ROC-AUC.

    If it is not possible (for example because a class is missing in the split),
    return NaN.
    """
    try:
        y_true_onehot = tf.keras.utils.to_categorical(
            y_true,
            num_classes=num_classes,
        )

        macro = roc_auc_score(
            y_true_onehot,
            y_prob,
            multi_class="ovr",
            average="macro",
        )

        weighted = roc_auc_score(
            y_true_onehot,
            y_prob,
            multi_class="ovr",
            average="weighted",
        )

        return float(macro), float(weighted)

    except ValueError:
        return float("nan"), float("nan")


def _compute_entropy(
    y_prob: np.ndarray,
    num_classes: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute Shannon entropy per sample and normalized entropy in [0, 1].
    """
    eps = 1e-12
    clipped = np.clip(y_prob, eps, 1.0)

    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    normalized_entropy = entropy / np.log(num_classes)

    return entropy.astype(np.float32), normalized_entropy.astype(np.float32)


def _entropy_stats(
    entropy: np.ndarray,
    normalized_entropy: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    correct_mask = y_true == y_pred
    wrong_mask = ~correct_mask

    stats = {
        f"{prefix}_mean_entropy": float(entropy.mean()),
        f"{prefix}_mean_normalized_entropy": float(normalized_entropy.mean()),
        f"{prefix}_mean_entropy_correct": (
            float(entropy[correct_mask].mean())
            if correct_mask.any()
            else float("nan")
        ),
        f"{prefix}_mean_entropy_wrong": (
            float(entropy[wrong_mask].mean())
            if wrong_mask.any()
            else float("nan")
        ),
        f"{prefix}_mean_normalized_entropy_correct": (
            float(normalized_entropy[correct_mask].mean())
            if correct_mask.any()
            else float("nan")
        ),
        f"{prefix}_mean_normalized_entropy_wrong": (
            float(normalized_entropy[wrong_mask].mean())
            if wrong_mask.any()
            else float("nan")
        ),
    }
    return stats


def _build_summary(
    config: PipelineConfig,
    data: DatasetBundle,
    paths: ExperimentPaths,
    eval_dict: dict[str, float],
    report: dict,
    class_names: list[str],
    manual_accuracy: float,
    roc_auc_macro: float,
    roc_auc_weighted: float,
    val_report: dict,
    val_auc: float,
    val_auc_weighted: float,
    test_entropy: np.ndarray,
    test_normalized_entropy: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    val_entropy: np.ndarray,
    val_normalized_entropy: np.ndarray,
    val_y_true: np.ndarray,
    val_y_pred: np.ndarray,
) -> dict:
    """Build the final summary dictionary."""
    test_entropy_stats = _entropy_stats(
        entropy=test_entropy,
        normalized_entropy=test_normalized_entropy,
        y_true=y_true,
        y_pred=y_pred,
        prefix="test",
    )

    val_entropy_stats = _entropy_stats(
        entropy=val_entropy,
        normalized_entropy=val_normalized_entropy,
        y_true=val_y_true,
        y_pred=val_y_pred,
        prefix="val",
    )

    return {
        "run_id": paths.run_id,
        "run_name": config.run_name,
        "model_name": config.model_name,
        "class_names": class_names,
        "dataset": {
            "train_dir": str(config.train_dir),
            "test_dir": str(config.test_dir),
            "validation_split": config.val_split,
            "seed": config.seed,
            "train_samples_used": data.train_samples,
            "val_samples_used": data.val_samples,
            "test_samples_used": data.test_samples,
        },
        "test_results": {
            **eval_dict,
            "manual_test_accuracy": manual_accuracy,
            "roc_auc_ovr_macro": roc_auc_macro,
            "roc_auc_ovr_weighted": roc_auc_weighted,
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_precision": float(report["weighted avg"]["precision"]),
            "weighted_recall": float(report["weighted avg"]["recall"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            **test_entropy_stats,
        },
        "validation_results": {
            "val_roc_auc_ovr_macro": float(val_auc),
            "val_roc_auc_ovr_weighted": float(val_auc_weighted),
            "val_macro_precision": float(val_report["macro avg"]["precision"]),
            "val_macro_recall": float(val_report["macro avg"]["recall"]),
            "val_macro_f1": float(val_report["macro avg"]["f1-score"]),
            "val_weighted_precision": float(val_report["weighted avg"]["precision"]),
            "val_weighted_recall": float(val_report["weighted avg"]["recall"]),
            "val_weighted_f1": float(val_report["weighted avg"]["f1-score"]),
            **val_entropy_stats,
        },
        "paths": {
            "best_model": str(paths.best_model_path),
            "log_dir": str(paths.log_dir),
            "metrics_dir": str(paths.met_dir),
            "figures_dir": str(paths.fig_dir),
        },
    }


def evaluate_model(
    config: PipelineConfig,
    data: DatasetBundle,
    paths: ExperimentPaths,
) -> EvaluationResults:
    """
    Load the best model and compute full validation and test metrics.
    """
    model = keras.models.load_model(
        paths.best_model_path,
        compile=False,
    )

    eval_lr = (
        config.fine_tune_lr
        if config.fine_tune
        else config.learning_rate
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=eval_lr),
        loss="sparse_categorical_crossentropy",
        metrics=_build_eval_metrics(data.num_classes),
    )

    val_y_true_batches = []
    val_y_prob_batches = []

    for x_batch, y_batch in data.val_ds:
        preds = model.predict_on_batch(x_batch)
        val_y_prob_batches.append(preds)
        val_y_true_batches.append(y_batch.numpy())

    val_y_prob = np.concatenate(val_y_prob_batches, axis=0)
    val_y_true = np.concatenate(val_y_true_batches, axis=0)
    val_y_pred = np.argmax(val_y_prob, axis=1)

    val_report = classification_report(
        val_y_true,
        val_y_pred,
        labels=np.arange(data.num_classes),
        target_names=data.class_names,
        output_dict=True,
        zero_division=0,
    )
    val_report_df = pd.DataFrame(val_report).transpose()

    val_auc, val_auc_weighted = _safe_roc_auc(
        y_true=val_y_true,
        y_prob=val_y_prob,
        num_classes=data.num_classes,
    )

    val_entropy, val_normalized_entropy = _compute_entropy(
        y_prob=val_y_prob,
        num_classes=data.num_classes,
    )

    y_true_batches = []
    y_prob_batches = []

    for x_batch, y_batch in data.test_ds:
        preds = model.predict_on_batch(x_batch)
        y_prob_batches.append(preds)
        y_true_batches.append(y_batch.numpy())

    y_prob = np.concatenate(y_prob_batches, axis=0)
    y_true = np.concatenate(y_true_batches, axis=0)
    y_pred = np.argmax(y_prob, axis=1)

    eval_results = model.evaluate(
        data.test_ds,
        verbose=0,
    )

    eval_dict = {
        name: float(value)
        for name, value in zip(
            model.metrics_names,
            eval_results,
        )
    }

    manual_accuracy = float((y_pred == y_true).mean())

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(data.num_classes),
        target_names=data.class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(data.num_classes),
    )

    roc_auc_macro, roc_auc_weighted = _safe_roc_auc(
        y_true=y_true,
        y_prob=y_prob,
        num_classes=data.num_classes,
    )

    test_entropy, test_normalized_entropy = _compute_entropy(
        y_prob=y_prob,
        num_classes=data.num_classes,
    )

    summary = _build_summary(
        config=config,
        data=data,
        paths=paths,
        eval_dict=eval_dict,
        report=report,
        val_report=val_report,
        class_names=data.class_names,
        manual_accuracy=manual_accuracy,
        roc_auc_macro=roc_auc_macro,
        roc_auc_weighted=roc_auc_weighted,
        val_auc=val_auc,
        val_auc_weighted=val_auc_weighted,
        test_entropy=test_entropy,
        test_normalized_entropy=test_normalized_entropy,
        y_true=y_true,
        y_pred=y_pred,
        val_entropy=val_entropy,
        val_normalized_entropy=val_normalized_entropy,
        val_y_true=val_y_true,
        val_y_pred=val_y_pred,
    )

    return EvaluationResults(
        model=model,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        entropy=test_entropy,
        normalized_entropy=test_normalized_entropy,
        confusion_matrix=cm,
        classification_report_dict=report,
        classification_report_df=report_df,
        val_classification_report_dict=val_report,
        val_classification_report_df=val_report_df,
        val_auc=val_auc,
        val_auc_weighted=val_auc_weighted,
        eval_dict=eval_dict,
        summary=summary,
    )