# cli/train.py
from __future__ import annotations

import argparse
import sys
from dataclasses import fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import tensorflow as tf

from src.config import PipelineConfig
from src.data_loader import build_datasets
from src.evaluation import evaluate_model
from src.gpu import configure_gpu
from src.helpers import print_kv, print_section
from src.paths import ExperimentPaths
from src.reporting import create_reports
from src.training import train_model


def parse_args() -> argparse.Namespace:
    """Parse all CLI arguments for an OCT training run."""
    parser = argparse.ArgumentParser(
        description="OCT training pipeline"
    )

    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--train_subdir", type=str, default=None)
    parser.add_argument("--test_subdir", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)

    # NEU:
    # train_mode steuert die gesamte Trainingslogik.
    # Default kommt aus PipelineConfig.
    parser.add_argument(
        "--train_mode",
        type=str,
        choices=["stage1", "full", "finetune"],
        default=None,
    )

    # NEU:
    # Für train_mode='finetune'
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
    )

    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--fine_tune_epochs", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--fine_tune_lr", type=float, default=None)

    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--unfreeze_last_n", type=int, default=None)

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--gpu_index", type=int, default=None)

    parser.add_argument("--train_take", type=int, default=None)
    parser.add_argument("--val_take", type=int, default=None)
    parser.add_argument("--test_take", type=int, default=None)

    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--val_split", type=float, default=None)

    parser.add_argument("--cache", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--mixed_precision", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--fine_tune", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_class_weights", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--use_augmentation", action=argparse.BooleanOptionalAction, default=None)

    parser.add_argument(
        "--dry_run_name",
        action="store_true",
        help="Only generate and print the run name, then exit.",
    )

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """
    Create config defaults from PipelineConfig and override
    only CLI-provided values.
    """
    config_fields = {field.name for field in fields(PipelineConfig)}

    overrides = {
        key: value
        for key, value in vars(args).items()
        if key in config_fields and value is not None
    }

    return PipelineConfig(**overrides)


def print_class_distribution(data) -> None:
    print_section("Class Distribution")

    for split_name, counts in (
        ("Train", data.class_counts),
        ("Validation", data.val_class_counts),
        ("Test", data.test_class_counts),
    ):
        if counts is None:
            continue

        print(split_name)
        for idx, class_name in enumerate(data.class_names):
            print_kv(f"  {class_name}", int(counts[idx]))

    if data.class_weights is not None:
        print_kv("Computed Weights", data.class_weights)



def print_classification_table(
    title: str,
    report_df,
    class_names: list[str],
) -> None:
    cols = ["precision", "recall", "f1-score", "support"]

    print()
    print(f"=== {title} ===")
    print()

    rows = class_names + ["accuracy", "macro avg", "weighted avg"]

    print(
        report_df.loc[
            rows,
            cols,
        ].round(4).to_string()
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if not (0.0 < config.val_split < 1.0):
        raise ValueError("--val_split must be between 0 and 1.")

    paths = ExperimentPaths.from_config(config)

    if args.dry_run_name:
        print(f"Run Name: {config.run_name}")
        print(f"Run ID : {paths.run_id}")
        print(f"Output : {paths.output_root}")
        return

    tf.keras.utils.set_random_seed(config.seed)

    if config.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    paths.create_directories()

    print_section("Run Initialization")
    print_kv("Run ID", paths.run_id)
    print_kv("Run Name", config.run_name)
    print_kv("Model", config.model_name)
    print_kv("Data Directory", config.data_dir)
    print_kv("Train Directory", config.train_dir)
    print_kv("Test Directory", config.test_dir)
    print_kv("Image Size", config.img_size)
    print_kv("Batch Size", config.batch_size)
    print_kv("Train Mode", config.train_mode)

    if config.train_mode == "stage1":
        print_kv("Stage 1 Epochs", config.epochs)
        print_kv("Fine-Tuning Epochs", 0)
        print_kv("Fine-Tune LR", "-")

    elif config.train_mode == "full":
        print_kv("Stage 1 Epochs", config.epochs)
        print_kv("Fine-Tuning Epochs", config.fine_tune_epochs)
        print_kv("Fine-Tune LR", config.fine_tune_lr)

    elif config.train_mode == "finetune":
        print_kv("Base Model Path", config.base_model_path)
        print_kv("Stage 1 Epochs", 0)
        print_kv("Fine-Tuning Epochs", config.fine_tune_epochs)
        print_kv("Fine-Tune LR", config.fine_tune_lr)

    print_kv("Learning Rate", config.learning_rate)

    print_kv("Validation Split", config.val_split)
    print_kv("Seed", config.seed)
    print_kv("Data Augmentation", config.use_augmentation)
    print_kv("Class Weights", config.use_class_weights)
    print_kv("Mixed Precision", config.mixed_precision)

    gpu_info = configure_gpu(config.gpu_index)
    print_kv("Compute Device", gpu_info)

    print_section("Loading Data")
    data = build_datasets(config)

    print_kv("Classes", ", ".join(data.class_names))
    print_kv("Number of Classes", data.num_classes)

    print_section("Dataset Split")
    print_kv("Train Samples", data.train_samples)
    print_kv("Validation Samples", data.val_samples)
    print_kv("Test Samples", data.test_samples)
    print_kv("Validation Order", "deterministic (shuffle=False)")
    print_class_distribution(data)

    print_section("Training")
    _, history_dict = train_model(
        config=config,
        data=data,
        paths=paths,
    )

    if config.train_mode == "stage1":
        print("[INFO] Starting Stage 1 only: frozen backbone feature extraction.")

    elif config.train_mode == "full":
        print("[INFO] Starting full training: Stage 1 followed by Stage 2 fine-tuning.")

    elif config.train_mode == "finetune":
        print("[INFO] Starting Fine-Tune only: loading existing model and running Stage 2.")

    print()

    print_section("Evaluation")
    results = evaluate_model(
        config=config,
        data=data,
        paths=paths,
    )

    create_reports(
        history_dict=history_dict,
        results=results,
        paths=paths,
    )

    test_results = results.summary["test_results"]
    val_results = results.summary["validation_results"]

    test_loss = test_results.get("loss", float("nan"))
    if not isinstance(test_loss, (int, float)):
        test_loss = float("nan")

    print_section("Results Overview")

    print_classification_table(
        title="VALIDATION",
        report_df=results.val_classification_report_df,
        class_names=data.class_names,
    )
    print()
    print_kv("VAL Macro ROC-AUC OvR", f"{results.val_auc:.4f}")
    print_kv("VAL Weighted ROC-AUC OvR", f"{results.val_auc_weighted:.4f}")
    print_kv("VAL Mean Normalized Entropy", f"{val_results['val_mean_normalized_entropy']:.4f}")
    print_kv("VAL Mean Entropy Correct", f"{val_results['val_mean_entropy_correct']:.4f}")
    print_kv("VAL Mean Entropy Wrong",f"{val_results['val_mean_entropy_wrong']:.4f}",)

    print_classification_table(
        title="TEST",
        report_df=results.classification_report_df,
        class_names=data.class_names,
    )
    print()
    print_kv("TEST Loss", f"{test_loss:.4f}")
    print_kv("TEST Accuracy", f"{test_results['manual_test_accuracy']:.4f}")
    print_kv("TEST Macro ROC-AUC OvR", f"{test_results['roc_auc_ovr_macro']:.4f}")
    print_kv("TEST Weighted ROC-AUC OvR", f"{test_results['roc_auc_ovr_weighted']:.4f}")

    if "sparse_top_k_categorical_accuracy" in test_results:
        print_kv("TEST Top-K Accuracy", f"{test_results['sparse_top_k_categorical_accuracy']:.4f}")

    print_kv("TEST Mean Normalized Entropy", f"{test_results['test_mean_normalized_entropy']:.4f}")
    print_kv("TEST Mean Entropy Correct", f"{test_results['test_mean_entropy_correct']:.4f}")
    print_kv("TEST Mean Entropy Wrong", f"{test_results['test_mean_entropy_wrong']:.4f}")

    print_section("Saved Artifacts")
    print_kv("Best Model", paths.best_model_path)
    print_kv("Log Directory", paths.log_dir)
    print_kv("Metrics Directory", paths.met_dir)
    print_kv("Figures Directory", paths.fig_dir)

    print()
    print(f"[INFO] Results saved to: {paths.output_root}")

    # Compact summary for copy & paste
    print()
    print("---- SUMMARY ----")
    print_kv("Model ID", paths.run_id)
    print_kv("Train Mode", config.train_mode)
    print_kv("TEST Accuracy", f"{results.summary['test_results']['manual_test_accuracy']:.4f}")
    print_kv("TEST Loss", f"{test_loss:.4f}")
    print_kv("TEST Macro F1", f"{results.summary['test_results']['macro_f1']:.4f}")
    print_kv("TEST Macro ROC-AUC OvR", f"{results.summary['test_results']['roc_auc_ovr_macro']:.4f}")
    print_kv("TEST Mean Normalized Entropy", f"{results.summary['test_results']['test_mean_normalized_entropy']:.4f}")

if __name__ == "__main__":
    main()