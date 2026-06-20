# cli/hyperparameter_tuning.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import optuna

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter tuning entry point for OCT models.",
    )
    parser.add_argument("--mode", choices=["stage1", "finetune", "full"], default="stage1")
    parser.add_argument("--model_name", default="inceptionv3")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--fine_tune_epochs", type=int, default=5)
    parser.add_argument("--batch_size_choices", type=int, nargs="+", default=[16, 32])
    parser.add_argument("--unfreeze_choices", type=int, nargs="+", default=[20, 50, 70])
    parser.add_argument("--train_take", type=int, default=1000)
    parser.add_argument("--val_take", type=int, default=300)
    parser.add_argument("--test_take", type=int, default=300)
    parser.add_argument("--gpu_index", type=int, default=-1)
    parser.add_argument("--study_name", default="oct_tuning")
    parser.add_argument("--base_model_path", default=None)
    return parser.parse_args()


def objective(args: argparse.Namespace):
    import tensorflow as tf

    from src.config import PipelineConfig
    from src.data_loader import build_datasets
    from src.evaluation import evaluate_model
    from src.paths import ExperimentPaths
    from src.reporting import create_reports
    from src.training import train_model

    def _objective(trial: optuna.Trial) -> float:
        tf.keras.backend.clear_session()

        batch_size = trial.suggest_categorical("batch_size", args.batch_size_choices)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        use_class_weights = True
        use_augmentation = False

        learning_rate = 1e-5
        fine_tune_lr = 1e-5
        unfreeze_last_n = 0
        fine_tune_epochs = 0
        train_mode = args.mode

        if args.mode == "stage1":
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            fine_tune_epochs = 0
            train_mode = "stage1"

        elif args.mode == "finetune":
            fine_tune_lr = trial.suggest_float("fine_tune_lr", 1e-6, 1e-4, log=True)
            unfreeze_last_n = trial.suggest_categorical("unfreeze_last_n", args.unfreeze_choices)
            fine_tune_epochs = args.fine_tune_epochs
            train_mode = "finetune"

        elif args.mode == "full":
            learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
            fine_tune_lr = trial.suggest_float("fine_tune_lr", 1e-6, 1e-4, log=True)
            unfreeze_last_n = trial.suggest_categorical("unfreeze_last_n", args.unfreeze_choices)
            fine_tune_epochs = args.fine_tune_epochs
            train_mode = "full"

        config = PipelineConfig(
            model_name=args.model_name,
            train_mode=train_mode,
            epochs=args.epochs,
            fine_tune_epochs=fine_tune_epochs,
            fine_tune=args.mode in {"finetune", "full"},
            learning_rate=learning_rate,
            fine_tune_lr=fine_tune_lr,
            batch_size=batch_size,
            dropout=dropout,
            unfreeze_last_n=unfreeze_last_n,
            use_class_weights=use_class_weights,
            use_augmentation=use_augmentation,
            train_take=args.train_take,
            val_take=args.val_take,
            test_take=args.test_take,
            gpu_index=args.gpu_index,
            base_model_path=args.base_model_path,
            run_name=f"optuna_{args.mode}_trial{trial.number}_{args.model_name}",
        )

        paths = ExperimentPaths.from_config(config)
        paths.create_directories()

        data = build_datasets(config)
        _, history = train_model(config=config, data=data, paths=paths)
        results = evaluate_model(config=config, data=data, paths=paths)
        create_reports(history_dict=history, results=results, paths=paths)

        score = float(results.summary["validation_results"]["val_macro_f1"])
        trial.set_user_attr("run_id", paths.run_id)
        trial.set_user_attr("test_accuracy", results.summary["test_results"].get("manual_test_accuracy"))
        trial.set_user_attr("mode", args.mode)
        trial.set_user_attr("batch_size", batch_size)
        trial.set_user_attr("unfreeze_last_n", unfreeze_last_n)
        return score

    return _objective


def main() -> None:
    args = parse_args()

    if args.n_trials <= 0:
        raise ValueError("--n_trials must be a positive integer.")
    if args.epochs <= 0:
        raise ValueError("--epochs must be a positive integer.")
    if args.fine_tune_epochs < 0:
        raise ValueError("--fine_tune_epochs must be non-negative.")
    if not args.batch_size_choices or any(x <= 0 for x in args.batch_size_choices):
        raise ValueError("--batch_size_choices must contain positive integers.")
    if not args.unfreeze_choices or any(x < 0 for x in args.unfreeze_choices):
        raise ValueError("--unfreeze_choices must contain non-negative integers.")

    from src.gpu import configure_gpu

    configure_gpu(args.gpu_index)

    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name,
    )
    study.optimize(objective(args), n_trials=args.n_trials)

    print("Best trial")
    print(f"  value : {study.best_trial.value}")
    print(f"  params: {study.best_trial.params}")
    print(f"  attrs : {study.best_trial.user_attrs}")


if __name__ == "__main__":
    main()