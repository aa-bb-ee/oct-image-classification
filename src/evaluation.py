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
    confusion_matrix: np.ndarray
    classification_report_dict: dict
    classification_report_df: pd.DataFrame
    eval_dict: dict[str, float]
    summary: dict


def _build_eval_metrics(num_classes: int) -> list[str]:
    """Erzeugt Built-in-Metriken für evaluate()."""
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
    Berechnet ROC-AUC.
    Falls nicht möglich (z.B. fehlende Klasse im Testset),
    wird NaN zurückgegeben.
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
) -> dict:
    """Erstellt das finale Summary-Dictionary."""
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
        "results": {
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
    Lädt das beste Modell und berechnet vollständige Testmetriken.
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

    # Labels und Predictions gemeinsam einsammeln
    y_true_batches = []
    y_prob_batches = []

    for x_batch, y_batch in data.test_ds:
        preds = model.predict_on_batch(x_batch)

        y_prob_batches.append(preds)
        y_true_batches.append(y_batch.numpy())

    y_prob = np.concatenate(y_prob_batches, axis=0)
    y_true = np.concatenate(y_true_batches, axis=0)

    y_pred = np.argmax(y_prob, axis=1)

    # echte Keras evaluate-Werte behalten
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

    manual_accuracy = float(
        (y_pred == y_true).mean()
    )

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

    summary = _build_summary(
        config=config,
        data=data,
        paths=paths,
        eval_dict=eval_dict,
        report=report,
        class_names=data.class_names,
        manual_accuracy=manual_accuracy,
        roc_auc_macro=roc_auc_macro,
        roc_auc_weighted=roc_auc_weighted,
    )

    return EvaluationResults(
        model=model,
        y_true=y_true,
        y_pred=y_pred,
        y_prob=y_prob,
        confusion_matrix=cm,
        classification_report_dict=report,
        classification_report_df=report_df,
        eval_dict=eval_dict,
        summary=summary,
    )
