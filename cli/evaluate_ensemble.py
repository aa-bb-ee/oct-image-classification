# cli/evaluate_ensemble.py
from __future__ import annotations

import argparse
import gc
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
    log_loss,
)
from tensorflow import keras

from src.config import PipelineConfig
from src.data_loader import build_datasets, build_patient_split_datasets
from src.gpu import configure_gpu
from src.helpers import save_json

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a soft-voting ensemble of trained OCT models.",
    )

    parser.add_argument(
        "--model_paths",
        type=Path,
        nargs="+",
        required=True,
        help="Paths to best_model.keras files.",
    )
    parser.add_argument(
        "--weights",
        type=float,
        nargs="*",
        default=None,
        help="Optional model weights. Must match number of models.",
    )
    parser.add_argument("--ensemble_name", type=str, default="soft_voting_ensemble")
    parser.add_argument("--output_root", type=Path, default=Path("experiment_outputs"))

    parser.add_argument("--data_dir", type=Path, default=Path("data/OCT"))
    parser.add_argument("--train_subdir", type=str, default="train")
    parser.add_argument("--test_subdir", type=str, default="test")
    parser.add_argument("--img_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu_index", type=int, default=-1)

    parser.add_argument(
        "--patient_split",
        action="store_true",
        help="Use patient-level train/validation split.",
    )

    return parser.parse_args()


def _safe_name(value: str) -> str:
    return (
        value.strip()
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
        .replace("\\", "_")
    )


def create_output_dirs(output_root: Path, ensemble_name: str) -> dict[str, Path | str]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"{timestamp}_{_safe_name(ensemble_name)}"

    run_dir = PROJECT_ROOT / output_root / run_id
    fig_dir = run_dir / "reports" / "figures"
    met_dir = run_dir / "reports" / "metrics"

    fig_dir.mkdir(parents=True, exist_ok=True)
    met_dir.mkdir(parents=True, exist_ok=True)

    return {
        "run_id": run_id,
        "run_dir": run_dir,
        "fig_dir": fig_dir,
        "met_dir": met_dir,
    }


def build_config(args: argparse.Namespace) -> PipelineConfig:
    return PipelineConfig(
        data_dir=args.data_dir,
        train_subdir=args.train_subdir,
        test_subdir=args.test_subdir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        val_split=args.val_split,
        seed=args.seed,
        gpu_index=args.gpu_index,
        model_name="ensemble",
        train_mode="stage1", # dummy value; no training is performed
        run_name=args.ensemble_name,
        cache=True,
        mixed_precision=False,
    )


def collect_labels(ds: tf.data.Dataset) -> np.ndarray:
    labels = []
    for _, y_batch in ds:
        labels.append(y_batch.numpy())
    return np.concatenate(labels, axis=0)


def predict_probs(model: keras.Model, ds: tf.data.Dataset) -> np.ndarray:
    probs = []
    for x_batch, _ in ds:
        probs.append(model.predict_on_batch(x_batch))
    return np.concatenate(probs, axis=0)


def normalize_weights(weights: list[float] | None, num_models: int) -> np.ndarray:
    if not weights:
        return np.ones(num_models, dtype=np.float32) / num_models

    if len(weights) != num_models:
        raise ValueError("--weights must match number of --model_paths.")

    arr = np.asarray(weights, dtype=np.float32)

    if np.any(arr < 0):
        raise ValueError("--weights must not contain negative values.")

    if arr.sum() <= 0:
        raise ValueError("At least one weight must be > 0.")

    return arr / arr.sum()


def compute_entropy(y_prob: np.ndarray, num_classes: int) -> tuple[np.ndarray, np.ndarray]:
    eps = 1e-12
    clipped = np.clip(y_prob, eps, 1.0)
    entropy = -(clipped * np.log(clipped)).sum(axis=1)
    normalized_entropy = entropy / np.log(num_classes)
    return entropy, normalized_entropy


def safe_roc_auc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
) -> tuple[float, float]:
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


def top2_accuracy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    top2 = np.argsort(y_prob, axis=1)[:, -2:]
    hits = [y_true[i] in top2[i] for i in range(len(y_true))]
    return float(np.mean(hits))


def entropy_stats(
    entropy: np.ndarray,
    normalized_entropy: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prefix: str,
) -> dict[str, float]:
    correct = y_true == y_pred
    wrong = ~correct

    return {
        f"{prefix}_mean_entropy": float(entropy.mean()),
        f"{prefix}_mean_normalized_entropy": float(normalized_entropy.mean()),
        f"{prefix}_mean_entropy_correct": (
            float(entropy[correct].mean()) if correct.any() else float("nan")
        ),
        f"{prefix}_mean_entropy_wrong": (
            float(entropy[wrong].mean()) if wrong.any() else float("nan")
        ),
        f"{prefix}_mean_normalized_entropy_correct": (
            float(normalized_entropy[correct].mean())
            if correct.any()
            else float("nan")
        ),
        f"{prefix}_mean_normalized_entropy_wrong": (
            float(normalized_entropy[wrong].mean())
            if wrong.any()
            else float("nan")
        ),
    }


def evaluate_split(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    prefix: str,
) -> tuple[dict[str, Any], pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    num_classes = len(class_names)
    y_pred = np.argmax(y_prob, axis=1)

    report = classification_report(
        y_true,
        y_pred,
        labels=np.arange(num_classes),
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    report_df = pd.DataFrame(report).transpose()

    roc_auc_macro, roc_auc_weighted = safe_roc_auc(
        y_true=y_true,
        y_prob=y_prob,
        num_classes=num_classes,
    )

    entropy, normalized_entropy = compute_entropy(y_prob, num_classes)
    accuracy = float((y_pred == y_true).mean())

    loss = float(
        log_loss(
            y_true,
            y_prob,
            labels=np.arange(num_classes),
        )
    )

    if prefix == "test":
        metrics = {
            "manual_test_accuracy": accuracy,
            "roc_auc_ovr_macro": roc_auc_macro,
            "roc_auc_ovr_weighted": roc_auc_weighted,
            "macro_precision": float(report["macro avg"]["precision"]),
            "macro_recall": float(report["macro avg"]["recall"]),
            "macro_f1": float(report["macro avg"]["f1-score"]),
            "weighted_precision": float(report["weighted avg"]["precision"]),
            "weighted_recall": float(report["weighted avg"]["recall"]),
            "weighted_f1": float(report["weighted avg"]["f1-score"]),
            "sparse_top_k_categorical_accuracy": top2_accuracy(y_true, y_prob),
            "loss": loss,
            **entropy_stats(
                entropy=entropy,
                normalized_entropy=normalized_entropy,
                y_true=y_true,
                y_pred=y_pred,
                prefix="test",
            ),
        }
    else:
        metrics = {
            "val_accuracy": accuracy,
            "val_roc_auc_ovr_macro": roc_auc_macro,
            "val_roc_auc_ovr_weighted": roc_auc_weighted,
            "val_macro_precision": float(report["macro avg"]["precision"]),
            "val_macro_recall": float(report["macro avg"]["recall"]),
            "val_macro_f1": float(report["macro avg"]["f1-score"]),
            "val_weighted_precision": float(report["weighted avg"]["precision"]),
            "val_weighted_recall": float(report["weighted avg"]["recall"]),
            "val_weighted_f1": float(report["weighted avg"]["f1-score"]),
            "val_loss": loss,
            **entropy_stats(
                entropy=entropy,
                normalized_entropy=normalized_entropy,
                y_true=y_true,
                y_pred=y_pred,
                prefix="val",
            ),
        }

    return metrics, report_df, y_pred, entropy, normalized_entropy


def summarize_member_model(
    model_name: str,
    model_path: Path,
    weight: float,
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
) -> dict[str, Any]:
    metrics, _, _, _, _ = evaluate_split(
        y_true=y_true,
        y_prob=y_prob,
        class_names=class_names,
        prefix="test",
    )

    return {
        "model_name": model_name,
        "model_path": str(model_path),
        "ensemble_weight": float(weight),
        "test_loss": metrics["loss"],
        "test_accuracy": metrics["manual_test_accuracy"],
        "test_macro_precision": metrics["macro_precision"],
        "test_macro_recall": metrics["macro_recall"],
        "test_macro_f1": metrics["macro_f1"],
        "test_weighted_f1": metrics["weighted_f1"],
        "test_roc_auc_macro": metrics["roc_auc_ovr_macro"],
        "test_roc_auc_weighted": metrics["roc_auc_ovr_weighted"],
        "test_mean_entropy": metrics["test_mean_entropy"],
        "test_mean_normalized_entropy": metrics["test_mean_normalized_entropy"],
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    fig_dir: Path,
    run_id: str,
) -> None:
    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=np.arange(len(class_names)),
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    ).plot(
        ax=ax,
        cmap="Blues",
        colorbar=False,
    )

    plt.title("Ensemble Test Confusion Matrix")
    plt.tight_layout()
    plt.savefig(fig_dir / f"{run_id}_cm.png", dpi=200)
    plt.close(fig)


def plot_roc_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: list[str],
    fig_dir: Path,
    run_id: str,
) -> None:
    num_classes = len(class_names)
    y_true_onehot = np.eye(num_classes)[y_true.astype(int)]

    fig, ax = plt.subplots(figsize=(8, 6))

    for i, name in enumerate(class_names):
        try:
            fpr, tpr, _ = roc_curve(y_true_onehot[:, i], y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f"{name} (AUC = {roc_auc:.2f})")
        except ValueError:
            continue

    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Ensemble ROC Curve (One-vs-Rest)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(fig_dir / f"{run_id}_roc_curve.png", dpi=200)
    plt.close(fig)


def plot_entropy_histogram(
    entropy: np.ndarray,
    fig_dir: Path,
    run_id: str,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.hist(entropy, bins=30, edgecolor="black")
    plt.xlabel("Entropy")
    plt.ylabel("Number of Samples")
    plt.title("Ensemble Test Entropy Distribution")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(fig_dir / f"{run_id}_test_entropy_histogram.png", dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch_size must be greater than 0.")

    tf.keras.utils.set_random_seed(args.seed)

    gpu_info = configure_gpu(args.gpu_index)

    config = build_config(args)
    output = create_output_dirs(args.output_root, args.ensemble_name)

    run_id = str(output["run_id"])
    run_dir = output["run_dir"]
    fig_dir = output["fig_dir"]
    met_dir = output["met_dir"]

    data = (
        build_patient_split_datasets(config)
        if args.patient_split
        else build_datasets(config)
    )

    class_names = data.class_names
    num_classes = data.num_classes
    weights = normalize_weights(args.weights, len(args.model_paths))

    val_y_true = collect_labels(data.val_ds)
    test_y_true = collect_labels(data.test_ds)

    val_prob_sum: np.ndarray | None = None
    test_prob_sum: np.ndarray | None = None

    member_rows: list[dict[str, Any]] = []

    print()
    print("=" * 80)
    print("Ensemble Evaluation")
    print("=" * 80)
    print(f"Run ID       : {run_id}")
    print(f"Run Dir      : {run_dir}")
    print(f"GPU          : {gpu_info}")
    print(f"Classes      : {', '.join(class_names)}")
    print(f"Models       : {len(args.model_paths)}")
    print(f"Val samples  : {len(val_y_true)}")
    print(f"Test samples : {len(test_y_true)}")
    print()

    for idx, model_path in enumerate(args.model_paths):
        model_path = model_path.resolve()

        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        print(f"[{idx + 1}/{len(args.model_paths)}] {model_path}")
        print(f"Weight: {weights[idx]:.4f}")

        model = keras.models.load_model(model_path, compile=False)

        output_size = int(model.output_shape[-1])
        input_size = int(model.input_shape[1])

        if output_size != num_classes:
            raise ValueError(
                f"Model output size ({output_size}) does not match "
                f"dataset classes ({num_classes}): {model_path}"
            )

        if input_size != config.img_size:
            raise ValueError(
                f"Model input size ({input_size}) does not match "
                f"--img_size ({config.img_size}): {model_path}"
            )

        val_probs = predict_probs(model, data.val_ds)
        test_probs = predict_probs(model, data.test_ds)

        member_rows.append(
            summarize_member_model(
                model_name=model_path.parents[1].name,
                model_path=model_path,
                weight=float(weights[idx]),
                y_true=test_y_true,
                y_prob=test_probs,
                class_names=class_names,
            )
        )

        if val_prob_sum is None:
            val_prob_sum = weights[idx] * val_probs
            test_prob_sum = weights[idx] * test_probs
        else:
            val_prob_sum += weights[idx] * val_probs
            test_prob_sum += weights[idx] * test_probs

        del model
        keras.backend.clear_session()
        gc.collect()

    if val_prob_sum is None or test_prob_sum is None:
        raise RuntimeError("No ensemble predictions were created.")

    pd.DataFrame(member_rows).to_csv(
        met_dir / f"{run_id}_ensemble_member_metrics.csv",
        index=False,
    )

    val_metrics, val_report_df, val_y_pred, val_entropy, val_norm_entropy = evaluate_split(
        y_true=val_y_true,
        y_prob=val_prob_sum,
        class_names=class_names,
        prefix="val",
    )

    test_metrics, test_report_df, test_y_pred, test_entropy, test_norm_entropy = evaluate_split(
        y_true=test_y_true,
        y_prob=test_prob_sum,
        class_names=class_names,
        prefix="test",
    )

    summary = {
        "run_id": run_id,
        "run_name": args.ensemble_name,
        "model_name": "ensemble_soft_voting",
        "class_names": class_names,
        "ensemble": {
            "method": "soft_voting_probability_average",
            "model_paths": [str(path.resolve()) for path in args.model_paths],
            "weights": [float(w) for w in weights],
        },
        "dataset": {
            "train_dir": str(config.train_dir),
            "test_dir": str(config.test_dir),
            "validation_split": config.val_split,
            "seed": config.seed,
            "train_samples_used": data.train_samples,
            "val_samples_used": data.val_samples,
            "test_samples_used": data.test_samples,
            "split_description": data.split_description,
        },
        "test_results": test_metrics,
        "validation_results": val_metrics,
    }

    run_config = {
        "run_name": args.ensemble_name,
        "model_name": "ensemble_soft_voting",
        "train_mode": "ensemble",
        "fine_tune": None,
        "use_class_weights": None,
        "use_augmentation": None,
        "unfreeze_last_n": None,
        "dropout": None,
        "learning_rate": None,
        "fine_tune_lr": None,
        "batch_size": config.batch_size,
        "epochs": None,
        "fine_tune_epochs": None,
        "img_size": config.img_size,
        "seed": config.seed,
        "val_split": config.val_split,
        "patient_split": args.patient_split,
        "model_paths": [str(path.resolve()) for path in args.model_paths],
        "weights": [float(w) for w in weights],
    }

    test_report_df.to_csv(
        met_dir / f"{run_id}_test_report.csv",
        index=True,
    )
    val_report_df.to_csv(
        met_dir / f"{run_id}_validation_report.csv",
        index=True,
    )

    pd.DataFrame(
        {
            "y_true": test_y_true,
            "y_pred": test_y_pred,
            "entropy": test_entropy,
            "normalized_entropy": test_norm_entropy,
        }
    ).to_csv(
        met_dir / f"{run_id}_test_entropy.csv",
        index=False,
    )
    pd.DataFrame(
        {
            "y_true": val_y_true,
            "y_pred": val_y_pred,
            "entropy": val_entropy,
            "normalized_entropy": val_norm_entropy,
        }
    ).to_csv(
        met_dir / f"{run_id}_validation_entropy.csv",
        index=False,
    )

    save_json(met_dir / f"{run_id}_summary.json", summary)
    save_json(met_dir / "run_config.json", run_config)

    plot_confusion_matrix(
        y_true=test_y_true,
        y_pred=test_y_pred,
        class_names=class_names,
        fig_dir=fig_dir,
        run_id=run_id,
    )
    plot_roc_curve(
        y_true=test_y_true,
        y_prob=test_prob_sum,
        class_names=class_names,
        fig_dir=fig_dir,
        run_id=run_id,
    )
    plot_entropy_histogram(
        entropy=test_entropy,
        fig_dir=fig_dir,
        run_id=run_id,
    )

    print()
    print("=" * 80)
    print("Ensemble Results")
    print("=" * 80)
    print(f"TEST Accuracy    : {test_metrics['manual_test_accuracy']:.4f}")
    print(f"TEST Macro F1    : {test_metrics['macro_f1']:.4f}")
    print(f"TEST Macro Recall: {test_metrics['macro_recall']:.4f}")
    print(f"TEST ROC-AUC     : {test_metrics['roc_auc_ovr_macro']:.4f}")
    print()
    print(f"Metrics saved to : {met_dir}")
    print(f"Figures saved to : {fig_dir}")


if __name__ == "__main__":
    main()