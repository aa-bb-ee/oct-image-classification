# cli/train.py
from __future__ import annotations

import argparse

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
    """Parst alle CLI-Argumente für einen Trainingslauf."""
    parser = argparse.ArgumentParser(
        description="OCT training pipeline"
    )

    parser.add_argument("--data_dir", type=str, default="data/OCT")
    parser.add_argument("--img_size", type=int, default=299)
    parser.add_argument("--batch_size", type=int, default=32)

    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--fine_tune_epochs", type=int, default=10)

    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5)

    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--unfreeze_last_n", type=int, default=50)

    parser.add_argument("--run_name", type=str, default="oct_exp")
    parser.add_argument("--model_name", type=str, default="inceptionv3")
    parser.add_argument("--gpu_index", type=int, default=-1)

    parser.add_argument("--train_take", type=int, default=-1)
    parser.add_argument("--val_take", type=int, default=-1)
    parser.add_argument("--test_take", type=int, default=-1)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val_split", type=float, default=0.1)

    parser.add_argument("--cache", action="store_true")
    parser.add_argument("--mixed_precision", action="store_true")
    parser.add_argument("--fine_tune", action="store_true")
    parser.add_argument("--use_class_weights", action="store_true")
    parser.add_argument("--use_augmentation", action=argparse.BooleanOptionalAction, default=True)

    return parser.parse_args()


def build_config(args: argparse.Namespace) -> PipelineConfig:
    """Erzeugt aus argparse-Argumenten eine PipelineConfig."""
    return PipelineConfig(
        data_dir=args.data_dir,
        img_size=args.img_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        fine_tune_epochs=args.fine_tune_epochs,
        learning_rate=args.learning_rate,
        fine_tune_lr=args.fine_tune_lr,
        dropout=args.dropout,
        unfreeze_last_n=args.unfreeze_last_n,
        run_name=args.run_name,
        model_name=args.model_name,
        gpu_index=args.gpu_index,
        train_take=args.train_take,
        val_take=args.val_take,
        test_take=args.test_take,
        seed=args.seed,
        val_split=args.val_split,
        cache=args.cache,
        mixed_precision=args.mixed_precision,
        fine_tune=args.fine_tune,
        use_class_weights=args.use_class_weights,
        use_augmentation=args.use_augmentation,
    )


def main() -> None:
    args = parse_args()
    config = build_config(args)

    if not (0.0 < config.val_split < 1.0):
        raise ValueError("--val_split muss zwischen 0 und 1 liegen.")

    tf.keras.utils.set_random_seed(config.seed)

    if config.mixed_precision:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    paths = ExperimentPaths.from_config(config)
    paths.create_directories()

    print_section("Laufinitialisierung")
    print_kv("Run-ID", paths.run_id)
    print_kv("Run-Name", config.run_name)
    print_kv("Modell", config.model_name)
    print_kv("Datenverzeichnis", config.data_dir)
    print_kv("Bildgröße", config.img_size)
    print_kv("Batch Size", config.batch_size)
    print_kv("Epochen Stage 1", config.epochs)
    print_kv("Epochen Fine-Tuning", config.fine_tune_epochs if config.fine_tune else 0)
    print_kv("Learning Rate", config.learning_rate)
    print_kv("Fine-Tune LR", config.fine_tune_lr if config.fine_tune else "-")
    print_kv("Validation Split", config.val_split)
    print_kv("Seed", config.seed)
    print_kv("Data Augmentation", config.use_augmentation)
    print_kv("Class Weights", config.use_class_weights)
    print_kv("Mixed Precision", config.mixed_precision)

    gpu_info = configure_gpu(config.gpu_index)
    print_kv("Rechengerät", gpu_info)

    print_section("Daten laden")
    data = build_datasets(config)

    print_kv("Klassen", ", ".join(data.class_names))
    print_kv("Anzahl Klassen", data.num_classes)

    print_section("Datensplit")
    print_kv("Train Samples", data.train_samples)
    print_kv("Validation Samples", data.val_samples)
    print_kv("Test Samples", data.test_samples)
    print_kv("Validation Reihenfolge", "deterministisch (shuffle=False)")

    if data.class_counts is not None:
        print_section("Klassengewichte")
        for idx, class_name in enumerate(data.class_names):
            print_kv(f"Train {class_name}", int(data.class_counts[idx]))
        print_kv("Berechnete Gewichte", data.class_weights)

    print_section("Training")
    _, history_dict = train_model(
        config=config,
        data=data,
        paths=paths,
    )

    print_section("Evaluation auf Testset")
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

    test_loss = results.summary["results"].get("loss", float("nan"))
    if not isinstance(test_loss, (int, float)):
        test_loss = float("nan")

    print_section("Ergebnisübersicht")
    print_kv("Bestes Modell", paths.best_model_path.name)
    print_kv("Test Loss", f"{test_loss:.4f}")
    print_kv("Test Accuracy", f"{results.summary['results']['manual_test_accuracy']:.4f}")

    if "sparse_top_k_categorical_accuracy" in results.summary["results"]:
        print_kv(
            "Test Top-2 Accuracy",
            f"{results.summary['results']['sparse_top_k_categorical_accuracy']:.4f}",
        )

    print_kv("ROC-AUC OVR Macro", f"{results.summary['results']['roc_auc_ovr_macro']:.4f}")
    print_kv("ROC-AUC OVR Weighted", f"{results.summary['results']['roc_auc_ovr_weighted']:.4f}")
    print_kv("Macro Precision", f"{results.summary['results']['macro_precision']:.4f}")
    print_kv("Macro Recall", f"{results.summary['results']['macro_recall']:.4f}")
    print_kv("Macro F1", f"{results.summary['results']['macro_f1']:.4f}")
    print_kv("Weighted Precision", f"{results.summary['results']['weighted_precision']:.4f}")
    print_kv("Weighted Recall", f"{results.summary['results']['weighted_recall']:.4f}")
    print_kv("Weighted F1", f"{results.summary['results']['weighted_f1']:.4f}")
    print_kv("Modellpfad", paths.best_model_path)
    print_kv("Logverzeichnis", paths.log_dir)
    print_kv("Metriken", paths.met_dir)
    print_kv("Abbildungen", paths.fig_dir)

    print()
    print("Klassenbezogene Kennzahlen:")
    class_report_cols = ["precision", "recall", "f1-score", "support"]
    print(
        results.classification_report_df.loc[
            data.class_names,
            class_report_cols,
        ].round(4).to_string()
    )

    print()
    print(f"[INFO] Ergebnisse gespeichert unter: {paths.output_root}")


if __name__ == "__main__":
    main()
